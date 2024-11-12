
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


import itertools


import torch


import collections


import abc


import numpy as np


import torch.utils.data


import collections.abc


import enum


import re


import copy


import random


import torch.nn.functional as F


from torch import nn


import math


import torch.nn as nn


from typing import Tuple


from typing import List


import functools


import time


import logging


from typing import Optional


from torch.utils.data.dataset import Dataset


from typing import Any


from typing import Dict


from typing import NewType


from torch import Tensor


import warnings


from typing import Callable


from typing import NamedTuple


from typing import Union


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import Sampler


from torch.utils.data.sampler import SequentialSampler


class Attention(torch.nn.Module):

    def __init__(self, pointer):
        super().__init__()
        self.pointer = pointer
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, values, attn_mask=None):
        attn_logits = self.pointer(query, values, attn_mask)
        attn = self.softmax(attn_logits)
        output = torch.bmm(attn.unsqueeze(1), values)
        output = output.squeeze(1)
        return output, attn


def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert all(a == 1 or b == 1 or a == b for a, b in zip(attn.shape[::-1], attn_mask.shape[::-1])), 'Attention mask shape {} should be broadcastable with attention shape {}'.format(attn_mask.shape, attn.shape)
        attn.data.masked_fill_(attn_mask, -float('inf'))


class ScaledDotProductPointer(torch.nn.Module):

    def __init__(self, query_size, key_size):
        super().__init__()
        self.query_proj = torch.nn.Linear(query_size, key_size)
        self.temp = np.power(key_size, 0.5)

    def forward(self, query, keys, attn_mask=None):
        proj_query = self.query_proj(query).unsqueeze(2)
        attn_logits = torch.bmm(keys, proj_query).squeeze(2) / self.temp
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


class ScaledDotProductAttention(Attention):

    def __init__(self, query_size, value_size):
        super().__init__(ScaledDotProductPointer(query_size, value_size))


class BahdanauPointer(torch.nn.Module):

    def __init__(self, query_size, key_size, proj_size):
        super().__init__()
        self.compute_scores = torch.nn.Sequential(torch.nn.Linear(query_size + key_size, proj_size), torch.nn.Tanh(), torch.nn.Linear(proj_size, 1))

    def forward(self, query: 'torch.Tensor', keys: 'torch.Tensor', attn_mask=None):
        query_expanded = query.unsqueeze(1).expand(-1, keys.shape[1], -1)
        attn_logits = self.compute_scores(torch.cat((query_expanded, keys), dim=2))
        attn_logits = attn_logits.squeeze(2)
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


class BahdanauAttention(Attention):

    def __init__(self, query_size, value_size, proj_size):
        super().__init__(BahdanauPointer(query_size, value_size, proj_size))


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module_fn, N):
    return nn.ModuleList([module_fn() for _ in range(N)])


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda : nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        if query.dim() == 3:
            x = x.squeeze(1)
        return self.linears[-1](x)


class ZippedDataset(torch.utils.data.Dataset):

    def __init__(self, *components):
        assert len(components) >= 1
        lengths = [len(c) for c in components]
        assert all(lengths[0] == other for other in lengths[1:]), "Lengths don't match: {}".format(lengths)
        self.components = components

    def __getitem__(self, idx):
        return tuple(c[idx] for c in self.components)

    def __len__(self):
        return len(self.components[0])


def get_field_presence_info(ast_wrapper, node, field_infos):
    present = []
    for field_info in field_infos:
        field_value = node.get(field_info.name)
        is_present = field_value is not None and field_value != []
        maybe_missing = field_info.opt or field_info.seq
        is_builtin_type = field_info.type in ast_wrapper.primitive_types
        if maybe_missing and is_builtin_type:
            present.append(is_present and type(field_value).__name__)
        elif maybe_missing and not is_builtin_type:
            present.append(is_present)
        elif not maybe_missing and is_builtin_type:
            present.append(type(field_value).__name__)
        elif not maybe_missing and not is_builtin_type:
            assert is_present
            present.append(True)
    return tuple(present)


def accumulate_logprobs(d, keys_and_logprobs):
    for key, logprob in keys_and_logprobs:
        existing = d.get(key)
        if existing is None:
            d[key] = logprob
        else:
            d[key] = torch.logsumexp(torch.stack((logprob, existing), dim=0), dim=0)


def maybe_stack(items, dim=None):
    to_stack = [item for item in items if item is not None]
    if not to_stack:
        return None
    elif len(to_stack) == 1:
        return to_stack[0].unsqueeze(dim)
    else:
        return torch.stack(to_stack, dim)


PUNKS = set(a for a in string.punctuation)


def compute_cell_value_linking(tokens, schema, db_dir):

    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_match(word, column, table, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or {column} like '% {word} %'  or {column} like '{word}'"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except:
            return False
    db_name = schema.db_id
    db_path = os.path.join(db_dir, db_name, db_name + '.sqlite')
    num_date_match = {}
    cell_match = {}
    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS or word in PUNKS:
            continue
        num_flag = isnumber(word)
        for col_id, column in enumerate(schema.columns):
            if col_id == 0:
                assert column.orig_name == '*'
                continue
            if num_flag:
                if column.type in ['number', 'time']:
                    num_date_match[f'{q_id},{col_id}'] = column.type.upper()
            else:
                ret = db_word_match(word, column.orig_name, column.table.orig_name, db_path)
                if ret:
                    cell_match[f'{q_id},{col_id}'] = 'CELLMATCH'
    cv_link = {'num_date_match': num_date_match, 'cell_match': cell_match, 'normalized_token': tokens}
    return cv_link


def compute_schema_linking(question, column, table):

    def partial_match(x_list, y_list):
        x_str = ' '.join(x_list)
        y_str = ' '.join(y_list)
        if x_str in STOPWORDS or x_str in PUNKS:
            return False
        if re.match(f'\\b{re.escape(x_str)}\\b', y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        x_str = ' '.join(x_list)
        y_str = ' '.join(y_list)
        if x_str == y_str:
            return True
        else:
            return False
    q_col_match = dict()
    q_tab_match = dict()
    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0:
            continue
        col_id2list[col_id] = col_item
    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item
    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram = ' '.join(n_gram_list)
            if len(n_gram.strip()) == 0:
                continue
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        q_col_match[f'{q_id},{col_id}'] = 'CEM'
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        q_tab_match[f'{q_id},{tab_id}'] = 'TEM'
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        if f'{q_id},{col_id}' not in q_col_match:
                            q_col_match[f'{q_id},{col_id}'] = 'CPM'
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        if f'{q_id},{tab_id}' not in q_tab_match:
                            q_tab_match[f'{q_id},{tab_id}'] = 'TPM'
        n -= 1
    return {'q_col_match': q_col_match, 'q_tab_match': q_tab_match}


class Bertokens:

    def __init__(self, pieces):
        self.pieces = pieces
        self.normalized_pieces = None
        self.idx_map = None
        self.normalize_toks()

    def normalize_toks(self):
        """
        If the token is not a word piece, then find its lemma
        If it is, combine pieces into a word, and then find its lemma
        E.g., a ##b ##c will be normalized as "abc", "", ""
        NOTE: this is only used for schema linking
        """
        self.startidx2pieces = dict()
        self.pieces2startidx = dict()
        cache_start = None
        for i, piece in enumerate(self.pieces + ['']):
            if piece.startswith('##'):
                if cache_start is None:
                    cache_start = i - 1
                self.pieces2startidx[i] = cache_start
                self.pieces2startidx[i - 1] = cache_start
            else:
                if cache_start is not None:
                    self.startidx2pieces[cache_start] = i
                cache_start = None
        assert cache_start is None
        combined_word = {}
        for start, end in self.startidx2pieces.items():
            assert end - start + 1 < 10
            pieces = [self.pieces[start]] + [self.pieces[_id].strip('##') for _id in range(start + 1, end)]
            word = ''.join(pieces)
            combined_word[start] = word
        idx_map = {}
        new_toks = []
        for i, piece in enumerate(self.pieces):
            if i in combined_word:
                idx_map[len(new_toks)] = i
                new_toks.append(combined_word[i])
            elif i in self.pieces2startidx:
                pass
            else:
                idx_map[len(new_toks)] = i
                new_toks.append(piece)
        self.idx_map = idx_map
        normalized_toks = []
        for i, tok in enumerate(new_toks):
            ann = corenlp.annotate(tok, annotators=['tokenize', 'ssplit', 'lemma'])
            lemmas = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
            lemma_word = ' '.join(lemmas)
            normalized_toks.append(lemma_word)
        self.normalized_pieces = normalized_toks

    def bert_schema_linking(self, columns, tables):
        question_tokens = self.normalized_pieces
        column_tokens = [c.normalized_pieces for c in columns]
        table_tokens = [t.normalized_pieces for t in tables]
        sc_link = compute_schema_linking(question_tokens, column_tokens, table_tokens)
        new_sc_link = {}
        for m_type in sc_link:
            _match = {}
            for ij_str in sc_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(',')
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = self.idx_map[q_id]
                _match[f'{real_q_id},{col_tab_id}'] = sc_link[m_type][ij_str]
            new_sc_link[m_type] = _match
        return new_sc_link


def preprocess_schema_uncached(schema, tokenize_func, include_table_name_in_column, fix_issue_16_primary_keys, bert=False):
    """If it's bert, we also cache the normalized version of 
    question/column/table for schema linking"""
    r = PreprocessedSchema()
    if bert:
        assert not include_table_name_in_column
    last_table_id = None
    for i, column in enumerate(schema.columns):
        col_toks = tokenize_func(column.name, column.unsplit_name)
        type_tok = '<type: {}>'.format(column.type)
        if bert:
            column_name = col_toks + [type_tok]
            r.normalized_column_names.append(Bertokens(col_toks))
        else:
            column_name = [type_tok] + col_toks
        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']
            else:
                table_name = tokenize_func(column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name
        r.column_names.append(column_name)
        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i)] = table_id
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])
            columns.append(i)
        if last_table_id != table_id:
            r.table_bounds.append(i)
            last_table_id = table_id
        if column.foreign_key_for is not None:
            r.foreign_keys[str(column.id)] = column.foreign_key_for.id
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)
    r.table_bounds.append(len(schema.columns))
    assert len(r.table_bounds) == len(schema.tables) + 1
    for i, table in enumerate(schema.tables):
        table_toks = tokenize_func(table.name, table.unsplit_name)
        r.table_names.append(table_toks)
        if bert:
            r.normalized_table_names.append(Bertokens(table_toks))
    last_table = schema.tables[-1]
    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    r.primary_keys = [column.id for table in schema.tables for column in table.primary_keys] if fix_issue_16_primary_keys else [column.id for column in last_table.primary_keys for table in schema.tables]
    return r


class AverageSpanExtractor(nn.Module):

    def __init__(self):
        super(AverageSpanExtractor, self).__init__()

    def forward(self, sequence_tensor: 'torch.FloatTensor', span_indices: 'torch.LongTensor', sequence_mask: 'torch.LongTensor'=None, span_indices_mask: 'torch.LongTensor'=None) ->torch.FloatTensor:
        span_starts, span_ends = span_indices.split(1, dim=-1)
        span_ends = span_ends - 1
        span_widths = span_ends - span_starts
        max_batch_span_width = span_widths.max().item() + 1
        global_average_logits = torch.ones(sequence_tensor.size()[:2] + (1,)).float()
        max_span_range_indices = utils.get_range_vector(max_batch_span_width, sequence_tensor.device).view(1, 1, -1)
        span_mask = (max_span_range_indices <= span_widths).float()
        raw_span_indices = span_ends - max_span_range_indices
        span_mask = span_mask * (raw_span_indices >= 0).float()
        span_indices = torch.relu(raw_span_indices.float()).long()
        flat_span_indices = utils.flatten_and_batch_shift_indices(span_indices, sequence_tensor.size(1))
        span_embeddings = utils.batched_index_select(sequence_tensor, span_indices, flat_span_indices)
        span_attention_logits = utils.batched_index_select(global_average_logits, span_indices, flat_span_indices).squeeze(-1)
        span_attention_weights = utils.masked_softmax(span_attention_logits, span_mask)
        attended_text_embeddings = utils.weighted_sum(span_embeddings, span_attention_weights)
        if span_indices_mask is not None:
            return attended_text_embeddings * span_indices_mask.unsqueeze(-1).float()
        return attended_text_embeddings


BART_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use BartTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.BartTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for translation and summarization training. By default, the model will create this tensor by shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            If you want to change padding behavior, you should read :func:`~transformers.modeling_bart._prepare_decoder_inputs` and modify.
            See diagram 1 in the paper for more info on the default strategy
"""


BART_START_DOCSTRING = """

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""


def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = 'encoder_decoder' if self.encoder_decoder_attention else 'self'

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(self, query, key: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, layer_state: 'Optional[Dict[str, Optional[Tensor]]]'=None, attn_mask: 'Optional[Tensor]'=None, need_weights=False) ->Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: 'bool' = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if layer_state is not None:
            saved_state = layer_state.get(self.cache_key, {})
            if 'prev_key' in saved_state:
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}
        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)
        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)
        layer_state[self.cache_key] = {'prev_key': k.view(bsz, self.num_heads, -1, self.head_dim), 'prev_value': v.view(bsz, self.num_heads, -1, self.head_dim), 'prev_key_padding_mask': key_padding_mask if not static_kv else None}
        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)
        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        if 'prev_key' in saved_state:
            _prev_key = saved_state['prev_key']
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if 'prev_value' in saved_state:
            _prev_value = saved_state['prev_value']
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: 'Optional[Tensor]' = saved_state.get('prev_key_padding_mask', None)
        key_padding_mask = self._cat_prev_key_padding_mask(key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv)
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(key_padding_mask: 'Optional[Tensor]', prev_key_padding_mask: 'Optional[Tensor]', batch_size: 'int', src_len: 'int', static_kv: 'bool') ->Optional[Tensor]:
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class DecoderLayer(nn.Module):

    def __init__(self, config: 'BartConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout, encoder_decoder_attention=True)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_hidden_states, encoder_attn_mask=None, layer_state=None, causal_mask=None, decoder_padding_mask=None):
        residual = x
        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, self_attn_weights = self.self_attn(query=x, key=x, layer_state=layer_state, key_padding_mask=decoder_padding_mask, attn_mask=causal_mask, need_weights=self.output_attentions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(query=x, key=encoder_hidden_states, key_padding_mask=encoder_attn_mask, layer_state=layer_state)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, self_attn_weights, layer_state


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', padding_idx: 'int'):
        assert padding_idx is not None
        num_embeddings += padding_idx + 1
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if use_cache:
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)
        else:
            positions = create_position_ids_from_input_ids(input, self.padding_idx)
        return super().forward(positions), positions


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f'odd embedding_dim {embedding_dim} not supported')
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: 'nn.Parameter'):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
            The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
        out[:, 0:dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, dim // 2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)
        else:
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


def invert_mask(attention_mask):
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: 'BartConfig', embed_tokens: 'nn.Embedding'):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings, config.d_model, config.pad_token_id)
        else:
            self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, config.d_model, self.padding_idx)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(self, input_ids, input_embed, encoder_hidden_states, encoder_padding_mask, decoder_padding_mask, decoder_causal_mask, decoder_cached_states=None, use_cache=False, **unused):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        if encoder_padding_mask is not None:
            encoder_padding_mask = invert_mask(encoder_padding_mask)
        positions, p_idx = self.embed_positions(input_ids, use_cache=use_cache)
        if use_cache:
            input_ids = input_ids[:, -1:]
            input_embed = input_embed[:, -1:]
            positions = positions[:, -1:]
        x = input_embed * self.embed_scale
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states += x,
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:
                continue
            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None
            x, layer_self_attn, layer_past = decoder_layer(x, encoder_hidden_states, encoder_attn_mask=encoder_padding_mask, decoder_padding_mask=decoder_padding_mask, layer_state=layer_state, causal_mask=decoder_causal_mask)
            if use_cache:
                next_decoder_cache.append(layer_past.copy())
            if self.layer_norm and idx == len(self.layers) - 1:
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += layer_self_attn,
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        if use_cache:
            next_cache = (encoder_hidden_states, encoder_padding_mask), next_decoder_cache
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


class EncoderLayer(nn.Module):

    def __init__(self, config: 'BartConfig'):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask, need_weights=self.output_attentions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: 'BartConfig', embed_tokens):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos, p_idx = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)
            if self.output_attentions:
                all_attentions.append(attn)
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)
        return x, encoder_states, all_attentions


BART_PRETRAINED_MODEL_ARCHIVE_MAP = {'facebook/bart-large': 'https://cdn.huggingface.co/facebook/bart-large/pytorch_model.bin', 'facebook/bart-large-mnli': 'https://cdn.huggingface.co/facebook/bart-large-mnli/pytorch_model.bin', 'facebook/bart-large-cnn': 'https://cdn.huggingface.co/facebook/bart-large-cnn/pytorch_model.bin', 'facebook/bart-large-xsum': 'https://cdn.huggingface.co/facebook/bart-large-xsum/pytorch_model.bin', 'facebook/mbart-large-en-ro': 'https://cdn.huggingface.co/facebook/mbart-large-en-ro/pytorch_model.bin'}


def _filter_out_falsey_values(tup) ->Tuple:
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def _prepare_bart_decoder_inputs(config, input_ids, pad_token_id, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1)
    return decoder_input_ids, decoder_padding_mask, causal_mask


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [(1 if i != dim else -1) for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


class BartTokens:

    def __init__(self, text, tokenizer):
        self.text = text
        self.tokenizer = tokenizer
        self.normalized_pieces = None
        self.idx_map = None
        self.normalize_toks()

    def normalize_toks(self):
        tokens = nltk.word_tokenize(self.text.replace("'", " ' ").replace('"', ' " '))
        self.idx_map = {}
        toks = []
        for i, tok in enumerate(tokens):
            self.idx_map[i] = len(toks)
            toks.extend(self.tokenizer.tokenize(tok, add_prefix_space=True))
        normalized_toks = []
        for i, tok in enumerate(tokens):
            ann = corenlp.annotate(tok, annotators=['tokenize', 'ssplit', 'lemma'])
            lemmas = [tok.lemma.lower() for sent in ann.sentence for tok in sent.token]
            lemma_word = ' '.join(lemmas)
            normalized_toks.append(lemma_word)
        self.normalized_pieces = normalized_toks

    def bart_schema_linking(self, columns, tables):
        question_tokens = self.normalized_pieces
        column_tokens = [c.normalized_pieces for c in columns]
        table_tokens = [t.normalized_pieces for t in tables]
        sc_link = compute_schema_linking(question_tokens, column_tokens, table_tokens)
        new_sc_link = {}
        for m_type in sc_link:
            _match = {}
            for ij_str in sc_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(',')
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = self.idx_map[q_id]
                _match[f'{real_q_id},{col_tab_id}'] = sc_link[m_type][ij_str]
            new_sc_link[m_type] = _match
        return new_sc_link

    def bart_cv_linking(self, schema, db_path):
        question_tokens = self.normalized_pieces
        cv_link = compute_cell_value_linking(question_tokens, schema, db_path)
        new_cv_link = {}
        for m_type in cv_link:
            if m_type != 'normalized_token':
                _match = {}
                for ij_str in cv_link[m_type]:
                    q_id_str, col_tab_id_str = ij_str.split(',')
                    q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                    real_q_id = self.idx_map[q_id]
                    _match[f'{real_q_id},{col_tab_id}'] = cv_link[m_type][ij_str]
                new_cv_link[m_type] = _match
            else:
                new_cv_link[m_type] = cv_link[m_type]
        return new_cv_link


def preprocess_schema_uncached_bart(schema, tokenizer, tokenize_func, include_table_name_in_column, fix_issue_16_primary_keys, bart=False):
    """If it's bert, we also cache the normalized version of
    question/column/table for schema linking"""
    r = PreprocessedSchema()
    if bart:
        assert not include_table_name_in_column
    last_table_id = None
    for i, column in enumerate(schema.columns):
        col_toks = tokenize_func(column.name, column.unsplit_name)
        type_tok = '<type: {}>'.format(column.type)
        if bart:
            column_name = col_toks + [type_tok]
            r.normalized_column_names.append(BartTokens(column.unsplit_name, tokenizer))
        else:
            column_name = [type_tok] + col_toks
        if include_table_name_in_column:
            if column.table is None:
                table_name = ['<any-table>']
            else:
                table_name = tokenize_func(column.table.name, column.table.unsplit_name)
            column_name += ['<table-sep>'] + table_name
        r.column_names.append(column_name)
        table_id = None if column.table is None else column.table.id
        r.column_to_table[str(i)] = table_id
        if table_id is not None:
            columns = r.table_to_columns.setdefault(str(table_id), [])
            columns.append(i)
        if last_table_id != table_id:
            r.table_bounds.append(i)
            last_table_id = table_id
        if column.foreign_key_for is not None:
            r.foreign_keys[str(column.id)] = column.foreign_key_for.id
            r.foreign_keys_tables[str(column.table.id)].add(column.foreign_key_for.table.id)
    r.table_bounds.append(len(schema.columns))
    assert len(r.table_bounds) == len(schema.tables) + 1
    for i, table in enumerate(schema.tables):
        table_toks = tokenize_func(table.name, table.unsplit_name)
        r.table_names.append(table_toks)
        if bart:
            r.normalized_table_names.append(BartTokens(table.unsplit_name, tokenizer))
    last_table = schema.tables[-1]
    r.foreign_keys_tables = serialization.to_dict_with_sorted_values(r.foreign_keys_tables)
    r.primary_keys = [column.id for table in schema.tables for column in table.primary_keys] if fix_issue_16_primary_keys else [column.id for column in last_table.primary_keys for table in schema.tables]
    return r


class LookupEmbeddings(torch.nn.Module):

    def __init__(self, device, vocab, embedder, emb_size, learnable_words=[]):
        super().__init__()
        self._device = device
        self.vocab = vocab
        self.embedder = embedder
        self.emb_size = emb_size
        self.embedding = torch.nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=emb_size)
        if self.embedder:
            assert emb_size == self.embedder.dim
        self.learnable_words = learnable_words
        init_embed_list = []
        for i, word in enumerate(self.vocab):
            if self.embedder.contains(word):
                init_embed_list.append(self.embedder.lookup(word))
            else:
                init_embed_list.append(self.embedding.weight[i])
        init_embed_weight = torch.stack(init_embed_list, 0)
        self.embedding.weight = nn.Parameter(init_embed_weight)

    def forward_unbatched(self, token_lists):
        embs = []
        for tokens in token_lists:
            token_indices = torch.tensor(self.vocab.indices(tokens), device=self._device).unsqueeze(0)
            emb = self.embedding(token_indices)
            emb = emb.transpose(0, 1)
            embs.append(emb)
        all_embs = torch.cat(embs, dim=0)
        boundaries = np.cumsum([0] + [emb.shape[0] for emb in embs])
        return all_embs, boundaries

    def _compute_boundaries(self, token_lists):
        boundaries = [np.cumsum([0] + [len(token_list) for token_list in token_lists_for_item]) for token_lists_for_item in token_lists]
        return boundaries

    def _embed_token(self, token, batch_idx):
        if token in self.learnable_words or not self.embedder.contains(token):
            return self.embedding.weight[self.vocab.index(token)]
        else:
            emb = self.embedder.lookup(token)
            return emb

    def forward(self, token_lists):
        all_embs = batched_sequence.PackedSequencePlus.from_lists(lists=[[token for token_list in token_lists_for_item for token in token_list] for token_lists_for_item in token_lists], item_shape=(self.emb_size,), device=self._device, item_to_tensor=self._embed_token)
        all_embs = all_embs.apply(lambda d: d)
        return all_embs, self._compute_boundaries(token_lists)

    def _embed_words_learned(self, token_lists):
        indices = batched_sequence.PackedSequencePlus.from_lists(lists=[[token for token_list in token_lists_for_item for token in token_list] for token_lists_for_item in token_lists], item_shape=(1,), tensor_type=torch.LongTensor, item_to_tensor=lambda token, batch_idx, out: out.fill_(self.vocab.index(token)))
        indices = indices.apply(lambda d: d)
        all_embs = indices.apply(lambda x: self.embedding(x.squeeze(-1)))
        return all_embs, self._compute_boundaries(token_lists)


class EmbLinear(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, input_):
        all_embs, boundaries = input_
        all_embs = all_embs.apply(lambda d: self.linear(d))
        return all_embs, boundaries


class BiLSTM(torch.nn.Module):

    def __init__(self, input_size, output_size, dropout, summarize, use_native=False):
        super().__init__()
        if use_native:
            self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=output_size // 2, bidirectional=True, dropout=dropout)
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.lstm = variational_lstm.LSTM(input_size=input_size, hidden_size=int(output_size // 2), bidirectional=True, dropout=dropout)
        self.summarize = summarize
        self.use_native = use_native

    def forward_unbatched(self, input_):
        all_embs, boundaries = input_
        new_boundaries = [0]
        outputs = []
        for left, right in zip(boundaries, boundaries[1:]):
            if self.use_native:
                inp = self.dropout(all_embs[left:right])
                output, (h, c) = self.lstm(inp)
            else:
                output, (h, c) = self.lstm(all_embs[left:right])
            if self.summarize:
                seq_emb = torch.cat((h[0], h[1]), dim=-1).unsqueeze(0)
                new_boundaries.append(new_boundaries[-1] + 1)
            else:
                seq_emb = output
                new_boundaries.append(new_boundaries[-1] + output.shape[0])
            outputs.append(seq_emb)
        return torch.cat(outputs, dim=0), new_boundaries

    def forward(self, input_):
        all_embs, boundaries = input_
        desc_lengths = []
        batch_desc_to_flat_map = {}
        for batch_idx, boundaries_for_item in enumerate(boundaries):
            for desc_idx, (left, right) in enumerate(zip(boundaries_for_item, boundaries_for_item[1:])):
                desc_lengths.append((batch_idx, desc_idx, right - left))
                batch_desc_to_flat_map[batch_idx, desc_idx] = len(batch_desc_to_flat_map)
        remapped_ps_indices = []

        def rearranged_all_embs_map_index(desc_lengths_idx, seq_idx):
            batch_idx, desc_idx, _ = desc_lengths[desc_lengths_idx]
            return batch_idx, boundaries[batch_idx][desc_idx] + seq_idx

        def rearranged_all_embs_gather_from_indices(indices):
            batch_indices, seq_indices = zip(*indices)
            remapped_ps_indices[:] = all_embs.raw_index(batch_indices, seq_indices)
            return all_embs.ps.data[torch.LongTensor(remapped_ps_indices)]
        rearranged_all_embs = batched_sequence.PackedSequencePlus.from_gather(lengths=[length for _, _, length in desc_lengths], map_index=rearranged_all_embs_map_index, gather_from_indices=rearranged_all_embs_gather_from_indices)
        rev_remapped_ps_indices = tuple(x[0] for x in sorted(enumerate(remapped_ps_indices), key=operator.itemgetter(1)))
        if self.use_native:
            rearranged_all_embs = rearranged_all_embs.apply(self.dropout)
        output, (h, c) = self.lstm(rearranged_all_embs.ps)
        if self.summarize:
            h = torch.cat((h[0], h[1]), dim=-1)
            new_all_embs = batched_sequence.PackedSequencePlus.from_gather(lengths=[(len(boundaries_for_item) - 1) for boundaries_for_item in boundaries], map_index=lambda batch_idx, desc_idx: rearranged_all_embs.sort_to_orig[batch_desc_to_flat_map[batch_idx, desc_idx]], gather_from_indices=lambda indices: h[torch.LongTensor(indices)])
            new_boundaries = [list(range(len(boundaries_for_item))) for boundaries_for_item in boundaries]
        else:
            new_all_embs = all_embs.apply(lambda _: output.data[torch.LongTensor(rev_remapped_ps_indices)])
            new_boundaries = boundaries
        return new_all_embs, new_boundaries


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, layer_size, N, tie_layers=False):
        super(Encoder, self).__init__()
        if tie_layers:
            self.layer = layer()
            self.layers = [self.layer for _ in range(N)]
        else:
            self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer_size)

    def forward(self, x, relation, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, relation, mask)
        return self.norm(x)


def relative_attention_logits(query, key, relation):
    qk_matmul = torch.matmul(query, key.transpose(-2, -1))
    q_t = query.permute(0, 2, 1, 3)
    r_t = relation.transpose(-2, -1)
    q_tr_t_matmul = torch.matmul(q_t, r_t)
    q_tr_tmatmul_t = q_tr_t_matmul.permute(0, 2, 1, 3)
    return (qk_matmul + q_tr_tmatmul_t) / math.sqrt(query.shape[-1])


def relative_attention_values(weight, value, relation):
    wv_matmul = torch.matmul(weight, value)
    w_t = weight.permute(0, 2, 1, 3)
    w_tr_matmul = torch.matmul(w_t, relation)
    w_tr_matmul_t = w_tr_matmul.permute(0, 2, 1, 3)
    return wv_matmul + w_tr_matmul_t


def attention_with_relations(query, key, value, relation_k, relation_v, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = relative_attention_logits(query, key, relation_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn_orig = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn_orig)
    return relative_attention_values(p_attn, value, relation_v), p_attn_orig


class MultiHeadedAttentionWithRelations(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttentionWithRelations, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(lambda : nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, relation_k, relation_v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention_with_relations(query, key, value, relation_k, relation_v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PointerWithRelations(nn.Module):

    def __init__(self, hidden_size, num_relation_kinds, dropout=0.2):
        super(PointerWithRelations, self).__init__()
        self.hidden_size = hidden_size
        self.linears = clones(lambda : nn.Linear(hidden_size, hidden_size), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.relation_k_emb = nn.Embedding(num_relation_kinds, self.hidden_size)
        self.relation_v_emb = nn.Embedding(num_relation_kinds, self.hidden_size)

    def forward(self, query, key, value, relation, mask=None):
        relation_k = self.relation_k_emb(relation)
        relation_v = self.relation_v_emb(relation)
        if mask is not None:
            mask = mask.unsqueeze(0)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, 1, self.hidden_size).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        _, self.attn = attention_with_relations(query, key, value, relation_k, relation_v, mask=mask, dropout=self.dropout)
        return self.attn[0, 0]


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def clamp(value, abs_max):
    value = max(-abs_max, value)
    value = min(abs_max, value)
    return value


class RelationalTransformerUpdate(torch.nn.Module):

    def __init__(self, num_layers, num_heads, hidden_size, ff_size=None, dropout=0.1, merge_types=False, tie_layers=False, qq_max_dist=2, cc_foreign_key=True, cc_table_match=True, cc_max_dist=2, ct_foreign_key=True, ct_table_match=True, tc_table_match=True, tc_foreign_key=True, tt_max_dist=2, tt_foreign_key=True, sc_link=False, cv_link=False):
        super().__init__()
        self.num_heads = num_heads
        self.qq_max_dist = qq_max_dist
        self.cc_foreign_key = cc_foreign_key
        self.cc_table_match = cc_table_match
        self.cc_max_dist = cc_max_dist
        self.ct_foreign_key = ct_foreign_key
        self.ct_table_match = ct_table_match
        self.tc_table_match = tc_table_match
        self.tc_foreign_key = tc_foreign_key
        self.tt_max_dist = tt_max_dist
        self.tt_foreign_key = tt_foreign_key
        self.relation_ids = {}

        def add_relation(name):
            self.relation_ids[name] = len(self.relation_ids)

        def add_rel_dist(name, max_dist):
            for i in range(-max_dist, max_dist + 1):
                add_relation((name, i))
        add_rel_dist('qq_dist', qq_max_dist)
        add_relation('qc_default')
        add_relation('qt_default')
        add_relation('cq_default')
        add_relation('cc_default')
        if cc_foreign_key:
            add_relation('cc_foreign_key_forward')
            add_relation('cc_foreign_key_backward')
        if cc_table_match:
            add_relation('cc_table_match')
        add_rel_dist('cc_dist', cc_max_dist)
        add_relation('ct_default')
        if ct_foreign_key:
            add_relation('ct_foreign_key')
        if ct_table_match:
            add_relation('ct_primary_key')
            add_relation('ct_table_match')
            add_relation('ct_any_table')
        add_relation('tq_default')
        add_relation('tc_default')
        if tc_table_match:
            add_relation('tc_primary_key')
            add_relation('tc_table_match')
            add_relation('tc_any_table')
        if tc_foreign_key:
            add_relation('tc_foreign_key')
        add_relation('tt_default')
        if tt_foreign_key:
            add_relation('tt_foreign_key_forward')
            add_relation('tt_foreign_key_backward')
            add_relation('tt_foreign_key_both')
        add_rel_dist('tt_dist', tt_max_dist)
        if sc_link:
            add_relation('qcCEM')
            add_relation('cqCEM')
            add_relation('qtTEM')
            add_relation('tqTEM')
            add_relation('qcCPM')
            add_relation('cqCPM')
            add_relation('qtTPM')
            add_relation('tqTPM')
        if cv_link:
            add_relation('qcNUMBER')
            add_relation('cqNUMBER')
            add_relation('qcTIME')
            add_relation('cqTIME')
            add_relation('qcCELLMATCH')
            add_relation('cqCELLMATCH')
        if merge_types:
            assert not cc_foreign_key
            assert not cc_table_match
            assert not ct_foreign_key
            assert not ct_table_match
            assert not tc_foreign_key
            assert not tc_table_match
            assert not tt_foreign_key
            assert cc_max_dist == qq_max_dist
            assert tt_max_dist == qq_max_dist
            add_relation('xx_default')
            self.relation_ids['qc_default'] = self.relation_ids['xx_default']
            self.relation_ids['qt_default'] = self.relation_ids['xx_default']
            self.relation_ids['cq_default'] = self.relation_ids['xx_default']
            self.relation_ids['cc_default'] = self.relation_ids['xx_default']
            self.relation_ids['ct_default'] = self.relation_ids['xx_default']
            self.relation_ids['tq_default'] = self.relation_ids['xx_default']
            self.relation_ids['tc_default'] = self.relation_ids['xx_default']
            self.relation_ids['tt_default'] = self.relation_ids['xx_default']
            if sc_link:
                self.relation_ids['qcCEM'] = self.relation_ids['xx_default']
                self.relation_ids['qcCPM'] = self.relation_ids['xx_default']
                self.relation_ids['qtTEM'] = self.relation_ids['xx_default']
                self.relation_ids['qtTPM'] = self.relation_ids['xx_default']
                self.relation_ids['cqCEM'] = self.relation_ids['xx_default']
                self.relation_ids['cqCPM'] = self.relation_ids['xx_default']
                self.relation_ids['tqTEM'] = self.relation_ids['xx_default']
                self.relation_ids['tqTPM'] = self.relation_ids['xx_default']
            if cv_link:
                self.relation_ids['qcNUMBER'] = self.relation_ids['xx_default']
                self.relation_ids['cqNUMBER'] = self.relation_ids['xx_default']
                self.relation_ids['qcTIME'] = self.relation_ids['xx_default']
                self.relation_ids['cqTIME'] = self.relation_ids['xx_default']
                self.relation_ids['qcCELLMATCH'] = self.relation_ids['xx_default']
                self.relation_ids['cqCELLMATCH'] = self.relation_ids['xx_default']
            for i in range(-qq_max_dist, qq_max_dist + 1):
                self.relation_ids['cc_dist', i] = self.relation_ids['qq_dist', i]
                self.relation_ids['tt_dist', i] = self.relation_ids['tt_dist', i]
        if ff_size is None:
            ff_size = hidden_size * 4
        self.encoder = Encoder(lambda : EncoderLayer(hidden_size, MultiHeadedAttentionWithRelations(num_heads, hidden_size, dropout), PositionwiseFeedForward(hidden_size, ff_size, dropout), len(self.relation_ids), dropout), hidden_size, num_layers, tie_layers)
        self.align_attn = PointerWithRelations(hidden_size, len(self.relation_ids), dropout)

    def create_align_mask(self, num_head, q_length, c_length, t_length):
        all_length = q_length + c_length + t_length
        mask_1 = torch.ones(num_head - 1, all_length, all_length, device=next(self.parameters()).device)
        mask_2 = torch.zeros(1, all_length, all_length, device=next(self.parameters()).device)
        for i in range(q_length):
            for j in range(q_length, q_length + c_length):
                mask_2[0, i, j] = 1
                mask_2[0, j, i] = 1
        mask = torch.cat([mask_1, mask_2], 0)
        return mask

    def forward_unbatched(self, desc, q_enc, c_enc, c_boundaries, t_enc, t_boundaries):
        enc = torch.cat((q_enc, c_enc, t_enc), dim=0)
        enc = enc.transpose(0, 1)
        relations = self.compute_relations(desc, enc_length=enc.shape[1], q_enc_length=q_enc.shape[0], c_enc_length=c_enc.shape[0], c_boundaries=c_boundaries, t_boundaries=t_boundaries)
        relations_t = torch.LongTensor(relations)
        enc_new = self.encoder(enc, relations_t, mask=None)
        c_base = q_enc.shape[0]
        t_base = q_enc.shape[0] + c_enc.shape[0]
        q_enc_new = enc_new[:, :c_base]
        c_enc_new = enc_new[:, c_base:t_base]
        t_enc_new = enc_new[:, t_base:]
        m2c_align_mat = self.align_attn(enc_new, enc_new[:, c_base:t_base], enc_new[:, c_base:t_base], relations_t[:, c_base:t_base])
        m2t_align_mat = self.align_attn(enc_new, enc_new[:, t_base:], enc_new[:, t_base:], relations_t[:, t_base:])
        return q_enc_new, c_enc_new, t_enc_new, (m2c_align_mat, m2t_align_mat)

    def compute_relations(self, desc, enc_length, q_enc_length, c_enc_length, c_boundaries, t_boundaries):
        sc_link = desc.get('sc_link', {'q_col_match': {}, 'q_tab_match': {}})
        cv_link = desc.get('cv_link', {'num_date_match': {}, 'cell_match': {}})
        loc_types = {}
        for i in range(q_enc_length):
            loc_types[i] = 'question',
        c_base = q_enc_length
        for c_id, (c_start, c_end) in enumerate(zip(c_boundaries, c_boundaries[1:])):
            for i in range(c_start + c_base, c_end + c_base):
                loc_types[i] = 'column', c_id
        t_base = q_enc_length + c_enc_length
        for t_id, (t_start, t_end) in enumerate(zip(t_boundaries, t_boundaries[1:])):
            for i in range(t_start + t_base, t_end + t_base):
                loc_types[i] = 'table', t_id
        relations = np.empty((enc_length, enc_length), dtype=np.int64)
        for i, j in itertools.product(range(enc_length), repeat=2):

            def set_relation(name):
                relations[i, j] = self.relation_ids[name]
            i_type, j_type = loc_types[i], loc_types[j]
            if i_type[0] == 'question':
                if j_type[0] == 'question':
                    set_relation(('qq_dist', clamp(j - i, self.qq_max_dist)))
                elif j_type[0] == 'column':
                    j_real = j - c_base
                    if f'{i},{j_real}' in sc_link['q_col_match']:
                        set_relation('qc' + sc_link['q_col_match'][f'{i},{j_real}'])
                    elif f'{i},{j_real}' in cv_link['cell_match']:
                        set_relation('qc' + cv_link['cell_match'][f'{i},{j_real}'])
                    elif f'{i},{j_real}' in cv_link['num_date_match']:
                        set_relation('qc' + cv_link['num_date_match'][f'{i},{j_real}'])
                    else:
                        set_relation('qc_default')
                elif j_type[0] == 'table':
                    j_real = j - t_base
                    if f'{i},{j_real}' in sc_link['q_tab_match']:
                        set_relation('qt' + sc_link['q_tab_match'][f'{i},{j_real}'])
                    else:
                        set_relation('qt_default')
            elif i_type[0] == 'column':
                if j_type[0] == 'question':
                    i_real = i - c_base
                    if f'{j},{i_real}' in sc_link['q_col_match']:
                        set_relation('cq' + sc_link['q_col_match'][f'{j},{i_real}'])
                    elif f'{j},{i_real}' in cv_link['cell_match']:
                        set_relation('cq' + cv_link['cell_match'][f'{j},{i_real}'])
                    elif f'{j},{i_real}' in cv_link['num_date_match']:
                        set_relation('cq' + cv_link['num_date_match'][f'{j},{i_real}'])
                    else:
                        set_relation('cq_default')
                elif j_type[0] == 'column':
                    col1, col2 = i_type[1], j_type[1]
                    if col1 == col2:
                        set_relation(('cc_dist', clamp(j - i, self.cc_max_dist)))
                    else:
                        set_relation('cc_default')
                        if self.cc_foreign_key:
                            if desc['foreign_keys'].get(str(col1)) == col2:
                                set_relation('cc_foreign_key_forward')
                            if desc['foreign_keys'].get(str(col2)) == col1:
                                set_relation('cc_foreign_key_backward')
                        if self.cc_table_match and desc['column_to_table'][str(col1)] == desc['column_to_table'][str(col2)]:
                            set_relation('cc_table_match')
                elif j_type[0] == 'table':
                    col, table = i_type[1], j_type[1]
                    set_relation('ct_default')
                    if self.ct_foreign_key and self.match_foreign_key(desc, col, table):
                        set_relation('ct_foreign_key')
                    if self.ct_table_match:
                        col_table = desc['column_to_table'][str(col)]
                        if col_table == table:
                            if col in desc['primary_keys']:
                                set_relation('ct_primary_key')
                            else:
                                set_relation('ct_table_match')
                        elif col_table is None:
                            set_relation('ct_any_table')
            elif i_type[0] == 'table':
                if j_type[0] == 'question':
                    i_real = i - t_base
                    if f'{j},{i_real}' in sc_link['q_tab_match']:
                        set_relation('tq' + sc_link['q_tab_match'][f'{j},{i_real}'])
                    else:
                        set_relation('tq_default')
                elif j_type[0] == 'column':
                    table, col = i_type[1], j_type[1]
                    set_relation('tc_default')
                    if self.tc_foreign_key and self.match_foreign_key(desc, col, table):
                        set_relation('tc_foreign_key')
                    if self.tc_table_match:
                        col_table = desc['column_to_table'][str(col)]
                        if col_table == table:
                            if col in desc['primary_keys']:
                                set_relation('tc_primary_key')
                            else:
                                set_relation('tc_table_match')
                        elif col_table is None:
                            set_relation('tc_any_table')
                elif j_type[0] == 'table':
                    table1, table2 = i_type[1], j_type[1]
                    if table1 == table2:
                        set_relation(('tt_dist', clamp(j - i, self.tt_max_dist)))
                    else:
                        set_relation('tt_default')
                        if self.tt_foreign_key:
                            forward = table2 in desc['foreign_keys_tables'].get(str(table1), ())
                            backward = table1 in desc['foreign_keys_tables'].get(str(table2), ())
                            if forward and backward:
                                set_relation('tt_foreign_key_both')
                            elif forward:
                                set_relation('tt_foreign_key_forward')
                            elif backward:
                                set_relation('tt_foreign_key_backward')
        return relations

    @classmethod
    def match_foreign_key(cls, desc, col, table):
        foreign_key_for = desc['foreign_keys'].get(str(col))
        if foreign_key_for is None:
            return False
        foreign_table = desc['column_to_table'][str(foreign_key_for)]
        return desc['column_to_table'][str(col)] == foreign_table


class SublayerConnection(nn.Module):
    """
  A residual connection followed by a layer norm.
  Note for code simplicity the norm is first as opposed to last.
  """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class RecurrentDropoutLSTMCell(torch.jit.ScriptModule):
    __constants__ = ['hidden_size']

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(RecurrentDropoutLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.W_i = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_i = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_f = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_f = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_c = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_c = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_o = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.U_o = torch.nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.empty(4 * hidden_size))
        self._input_dropout_mask = torch.jit.Attribute(torch.empty((), requires_grad=False), torch.Tensor)
        self._h_dropout_mask = torch.jit.Attribute(torch.empty((), requires_grad=False), torch.Tensor)
        super(torch.jit.ScriptModule, self)._register_state_dict_hook(self._hook_remove_dropout_masks_from_state_dict)
        super(torch.jit.ScriptModule, self)._register_load_state_dict_pre_hook(self._hook_add_dropout_masks_to_state_dict)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.orthogonal_(self.W_i)
        torch.nn.init.orthogonal_(self.U_i)
        torch.nn.init.orthogonal_(self.W_f)
        torch.nn.init.orthogonal_(self.U_f)
        torch.nn.init.orthogonal_(self.W_c)
        torch.nn.init.orthogonal_(self.U_c)
        torch.nn.init.orthogonal_(self.W_o)
        torch.nn.init.orthogonal_(self.U_o)
        self.bias_ih.data.fill_(0.0)
        self.bias_ih.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)
        self.bias_hh.data.fill_(0.0)

    def set_dropout_masks(self, batch_size):

        def constant_mask(v):
            return torch.tensor(v).reshape(1, 1, 1).expand(4, batch_size, -1)
        if self.dropout:
            if self.training:
                new_tensor = self.W_i.data.new
                self._input_dropout_mask = torch.bernoulli(new_tensor(4, batch_size, self.input_size).fill_(1 - self.dropout))
                self._h_dropout_mask = torch.bernoulli(new_tensor(4, batch_size, self.hidden_size).fill_(1 - self.dropout))
            else:
                mask = constant_mask(1 - self.dropout)
                self._input_dropout_mask = mask
                self._h_dropout_mask = mask
        else:
            mask = constant_mask(1.0)
            self._input_dropout_mask = mask
            self._h_dropout_mask = mask

    @classmethod
    def _hook_remove_dropout_masks_from_state_dict(cls, instance, state_dict, prefix, local_metadata):
        del state_dict[prefix + '_input_dropout_mask']
        del state_dict[prefix + '_h_dropout_mask']

    def _hook_add_dropout_masks_to_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + '_input_dropout_mask'] = self._input_dropout_mask
        state_dict[prefix + '_h_dropout_mask'] = self._h_dropout_mask

    @torch.jit.script_method
    def forward(self, input: 'torch.Tensor', hidden_state: 'Tuple[torch.Tensor, torch.Tensor]'):
        h_tm1, c_tm1 = hidden_state
        xi_t = torch.nn.functional.linear(input * self._input_dropout_mask[0, :input.shape[0]], self.W_i)
        xf_t = torch.nn.functional.linear(input * self._input_dropout_mask[1, :input.shape[0]], self.W_f)
        xc_t = torch.nn.functional.linear(input * self._input_dropout_mask[2, :input.shape[0]], self.W_c)
        xo_t = torch.nn.functional.linear(input * self._input_dropout_mask[3, :input.shape[0]], self.W_o)
        hi_t = torch.nn.functional.linear(h_tm1 * self._h_dropout_mask[0, :input.shape[0]], self.U_i)
        hf_t = torch.nn.functional.linear(h_tm1 * self._h_dropout_mask[1, :input.shape[0]], self.U_f)
        hc_t = torch.nn.functional.linear(h_tm1 * self._h_dropout_mask[2, :input.shape[0]], self.U_c)
        ho_t = torch.nn.functional.linear(h_tm1 * self._h_dropout_mask[3, :input.shape[0]], self.U_o)
        i_t = torch.sigmoid(xi_t + self.bias_ih[:self.hidden_size] + hi_t + self.bias_hh[:self.hidden_size])
        f_t = torch.sigmoid(xf_t + self.bias_ih[self.hidden_size:2 * self.hidden_size] + hf_t + self.bias_hh[self.hidden_size:2 * self.hidden_size])
        c_t = f_t * c_tm1 + i_t * torch.tanh(xc_t + self.bias_ih[2 * self.hidden_size:3 * self.hidden_size] + hc_t + self.bias_hh[2 * self.hidden_size:3 * self.hidden_size])
        o_t = torch.sigmoid(xo_t + self.bias_ih[3 * self.hidden_size:4 * self.hidden_size] + ho_t + self.bias_hh[3 * self.hidden_size:4 * self.hidden_size])
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTM(torch.jit.ScriptModule):

    def __init__(self, input_size, hidden_size, bidirectional=False, dropout=0.0, cell_factory=RecurrentDropoutLSTMCell):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.cell_factory = cell_factory
        num_directions = 2 if bidirectional else 1
        self.lstm_cells = []
        for direction in range(num_directions):
            cell = cell_factory(input_size, hidden_size, dropout=dropout)
            self.lstm_cells.append(cell)
            suffix = '_reverse' if direction == 1 else ''
            cell_name = 'cell{}'.format(suffix)
            self.add_module(cell_name, cell)

    def forward(self, input, hidden_state=None):
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if not is_packed:
            raise NotImplementedError
        max_batch_size = input.batch_sizes[0]
        for cell in self.lstm_cells:
            cell.set_dropout_masks(max_batch_size)
        if hidden_state is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.data.new_zeros(num_directions, max_batch_size, self.hidden_size, requires_grad=False)
            hidden_state = hx, hx
        forward_hidden_state = tuple(v[0] for v in hidden_state)
        if self.bidirectional:
            reverse_hidden_state = tuple(v[1] for v in hidden_state)
            forward_output, (forward_h, forward_c) = self._forward_packed(input.data, input.batch_sizes, forward_hidden_state)
            reverse_output, (reverse_h, reverse_c) = self._reverse_packed(input.data, input.batch_sizes, reverse_hidden_state)
            return torch.nn.utils.rnn.PackedSequence(torch.cat((forward_output, reverse_output), dim=-1), input.batch_sizes, input.sorted_indices, input.unsorted_indices), (torch.stack((forward_h, reverse_h), dim=0), torch.stack((forward_c, reverse_c), dim=0))
        output, next_hidden = self._forward_packed(input.data, input.batch_sizes, forward_hidden_state)
        return torch.nn.utils.rnn.PackedSequence(output, input.batch_sizes, input.sorted_indices, input.unsorted_indices), next_hidden

    @torch.jit.script_method
    def _forward_packed(self, input: 'torch.Tensor', batch_sizes: 'torch.Tensor', hidden_state: 'Tuple[torch.Tensor, torch.Tensor]'):
        step_outputs = []
        hs = []
        cs = []
        input_offset = torch.zeros((), dtype=torch.int64)
        num_steps = batch_sizes.shape[0]
        last_batch_size = batch_sizes[0]
        h, c = hidden_state
        for i in range(num_steps):
            batch_size = batch_sizes[i]
            step_input = input.narrow(0, input_offset, batch_size)
            input_offset += batch_size
            dec = last_batch_size - batch_size
            if dec > 0:
                hs.append(h[last_batch_size - dec:last_batch_size])
                cs.append(c[last_batch_size - dec:last_batch_size])
                h = h[:last_batch_size - dec]
                c = c[:last_batch_size - dec]
            last_batch_size = batch_size
            h, c = self.cell(step_input, (h, c))
            step_outputs.append(h)
        hs.append(h)
        cs.append(c)
        hs.reverse()
        cs.reverse()
        concat_h = torch.cat(hs)
        concat_c = torch.cat(cs)
        return torch.cat(step_outputs, dim=0), (concat_h, concat_c)

    @torch.jit.script_method
    def _reverse_packed(self, input: 'torch.Tensor', batch_sizes: 'torch.Tensor', hidden_state: 'Tuple[torch.Tensor, torch.Tensor]'):
        step_outputs = []
        input_offset = torch.zeros((), dtype=torch.int64)
        num_steps = batch_sizes.shape[0]
        last_batch_size = batch_sizes[-1]
        h, c = hidden_state
        input_h, input_c = hidden_state
        h = h[:batch_sizes[-1]]
        c = c[:batch_sizes[-1]]
        i = num_steps - 1
        while i > -1:
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                h = torch.cat((h, input_h[last_batch_size:batch_size]))
                c = torch.cat((c, input_c[last_batch_size:batch_size]))
            step_input = input.narrow(0, input_offset - batch_size, batch_size)
            input_offset -= batch_size
            last_batch_size = batch_size
            h, c = self.cell_reverse(step_input, (h, c))
            step_outputs.append(h)
            i -= 1
        step_outputs.reverse()
        return torch.cat(step_outputs, dim=0), (h, c)


BART_GENERATION_EXAMPLE = """
    Examples::

        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        # see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example
        model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
        ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

"""


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


WEIGHTS_NAME = 'pytorch_model.bin'


logger = logging.getLogger(__name__)


class MultiDModel(nn.Module):
    """
  output: tuple: (loss, ) in training
  """

    def __init__(self):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    def forward(self, *input, **kwargs):
        input_ids = kwargs.pop('input_ids')
        pad_token_id = kwargs.pop('pad_token_id')
        attention_mask = (input_ids != pad_token_id).long()
        if self.training:
            output_ids = kwargs.pop('labels')
            y_ids = output_ids[:, :-1].contiguous()
            lm_labels = output_ids[:, 1:].clone()
            lm_labels[output_ids[:, 1:] == pad_token_id] = -100
            outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            return outputs[0],
        else:
            label_eos_id = kwargs.pop('label_eos_id')
            label_bos_id = kwargs.pop('label_bos_id')
            label_padding_id = kwargs.pop('label_padding_id')
            generated_ids = self.bert.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=3, max_length=60, length_penalty=2.0, early_stopping=True, use_cache=True, decoder_start_token_id=label_bos_id, eos_token_id=label_eos_id, pad_token_id=label_padding_id)
            output_ids = kwargs.pop('labels')
            y_ids = output_ids[:, :-1].contiguous()
            lm_labels = output_ids[:, 1:].clone()
            lm_labels[output_ids[:, 1:] == pad_token_id] = -100
            outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            return outputs[0].detach(), generated_ids

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
    """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info('Model weights saved in {}'.format(output_model_file))


class RelationalBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: 'BartConfig', embed_tokens):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = SinusoidalPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None
        num_heads = 8
        hidden_size = 1024
        num_layers = 8
        self.relational_transformer = RelationalTransformerUpdate(num_layers=num_layers, num_heads=num_heads, hidden_size=hidden_size, sc_link=True, cv_link=True)
        self.use_relation_transformer = True

    def forward(self, input_ids, attention_mask, example_info_list):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos, p_idx = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)
            if self.output_attentions:
                all_attentions.append(attn)
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)
        max_q_length = max([(example_info['question_end'] - example_info['question_start']) for example_info in example_info_list])
        max_column_length = max([len(example_info['column_start']) for example_info in example_info_list])
        max_table_length = max([len(example_info['table_start']) for example_info in example_info_list])
        batch_size, dim = x.size(0), x.size(-1)
        batch_q_enc = x.new_zeros((batch_size, max_q_length, dim))
        batch_q_enc_mask = x.new_zeros((batch_size, max_q_length))
        batch_col_enc = x.new_zeros((batch_size, max_column_length, dim))
        batch_col_enc_mask = x.new_zeros((batch_size, max_column_length))
        batch_tab_enc = x.new_zeros((batch_size, max_table_length, dim))
        batch_tab_enc_mask = x.new_zeros((batch_size, max_table_length))
        for batch_idx, example_info in enumerate(example_info_list):
            q_enc = x[batch_idx][example_info['question_start']:example_info['question_end']]
            col_enc_start = x[batch_idx][example_info['column_start']]
            tab_enc_start = x[batch_idx][example_info['table_start']]
            col_enc_end = x[batch_idx][example_info['column_end'] - 1]
            tab_enc_end = x[batch_idx][example_info['table_end'] - 1]
            col_enc = (col_enc_start + col_enc_end) / 2.0
            tab_enc = (tab_enc_start + tab_enc_end) / 2.0
            if self.use_relation_transformer:
                c_boundary = list(range(len(example_info['column_start']) + 1))
                t_boundary = list(range(len(example_info['table_start']) + 1))
                q_enc_new, c_enc_new, t_enc_new, _ = self.relational_transformer.forward_unbatched(example_info, q_enc.unsqueeze(1), col_enc.unsqueeze(1), c_boundary, tab_enc.unsqueeze(1), t_boundary)
            else:
                q_enc_new, c_enc_new, t_enc_new = q_enc, col_enc, tab_enc
            batch_q_enc[batch_idx, :q_enc.size(0)] = q_enc_new
            batch_q_enc_mask[batch_idx, :q_enc.size(0)] = 1
            batch_col_enc[batch_idx, :col_enc.size(0)] = c_enc_new
            batch_col_enc_mask[batch_idx, :col_enc.size(0)] = 1
            batch_tab_enc[batch_idx, :tab_enc.size(0)] = t_enc_new
            batch_tab_enc_mask[batch_idx, :tab_enc.size(0)] = 1
        return ((batch_q_enc, batch_q_enc_mask), (batch_col_enc, batch_col_enc_mask), (batch_tab_enc, batch_tab_enc_mask)), encoder_states, all_attentions


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BeamHypotheses(object):

    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1000000000.0

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            return True
        if len(tokens) > len(prev_input_ids):
            return False
        if prev_tokens[-len(tokens):] == tokens:
            return True
        else:
            return False
    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []
        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, 'Banned words token sequences {} cannot have an empty list'.format(bad_words_ids)
            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                continue
            banned_tokens_slice.append(banned_token_seq[-1])
        banned_tokens.append(banned_tokens_slice)
    return banned_tokens


def calc_banned_ngram_tokens(prev_input_ids: 'Tensor', num_hypos: 'int', no_repeat_ngram_size: 'int', cur_len: 'int') ->None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])
    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf'), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class RelationalBARTParser(nn.Module):
    """
  output: tuple: (loss, ) in training
  """

    def __init__(self):
        super().__init__()
        self.bert: 'RelationalBartForTextToSQL' = RelationalBartForTextToSQL.from_pretrained('facebook/bart-large')
        self.bert.model.encoder.use_relation_transformer = True

    def forward(self, *input, **kwargs):
        input_token_ids = kwargs.pop('input_ids')
        input_padding_id = kwargs.pop('input_padding_id')
        attention_mask = (input_token_ids != input_padding_id).long()
        example_info_list = kwargs.pop('example_info_list')
        if self.training:
            label_ids = kwargs.pop('labels')
            label_padding_id = kwargs.pop('label_padding_id')
            y_ids = label_ids[:, :-1].contiguous()
            lm_labels = label_ids[:, 1:].clone()
            lm_labels[label_ids[:, 1:] == label_padding_id] = -100
            outputs = self.bert(input_token_ids, example_info_list=example_info_list, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            return outputs[0],
        else:
            label_eos_id = kwargs.pop('label_eos_id')
            label_bos_id = kwargs.pop('label_bos_id')
            label_padding_id = kwargs.pop('label_padding_id')
            generated_ids = self.bert.generate(input_ids=input_token_ids, example_info_list=example_info_list, attention_mask=attention_mask, num_beams=1, max_length=30, length_penalty=2.0, early_stopping=True, use_cache=True, decoder_start_token_id=label_bos_id, eos_token_id=label_eos_id, pad_token_id=label_padding_id, vocab_size=len(KEYWORDS))
            return torch.zeros(1), generated_ids

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
    """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info('Model weights saved in {}'.format(output_model_file))


class BARTParser(nn.Module):
    """
  output: tuple: (loss, ) in training
  """

    def __init__(self):
        super().__init__()
        self.bert = BartForTextToSQL.from_pretrained('facebook/bart-large')

    def forward(self, *input, **kwargs):
        input_token_ids = kwargs.pop('input_ids')
        column_spans = kwargs.pop('column_spans')
        input_padding_id = kwargs.pop('input_padding_id')
        copy_span = None
        attention_mask = (input_token_ids != input_padding_id).long()
        if self.training:
            label_ids = kwargs.pop('labels')
            label_padding_id = kwargs.pop('label_padding_id')
            y_ids = label_ids[:, :-1].contiguous()
            lm_labels = label_ids[:, 1:].clone()
            lm_labels[label_ids[:, 1:] == label_padding_id] = -100
            outputs = self.bert(input_token_ids, column_spans=column_spans, copy_span=copy_span, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            return outputs[0],
        else:
            label_eos_id = kwargs.pop('label_eos_id')
            label_bos_id = kwargs.pop('label_bos_id')
            label_padding_id = kwargs.pop('label_padding_id')
            generated_ids = self.bert.generate(input_ids=input_token_ids, column_spans=column_spans, copy_span=copy_span, attention_mask=attention_mask, num_beams=1, max_length=30, length_penalty=2.0, early_stopping=True, use_cache=True, decoder_start_token_id=label_bos_id, eos_token_id=label_eos_id, pad_token_id=label_padding_id, vocab_size=len(KEYWORDS))
            return torch.zeros(1), generated_ids

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
    """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info('Model weights saved in {}'.format(output_model_file))


def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


class LogicalTaBARTModel(nn.Module):
    """
  output: tuple: (loss, ) in training
  """

    def __init__(self, task):
        super().__init__()
        tasks = task.replace(',', '+').split('+')
        self.bert_for_texttosql = BartForTextToSQL.from_pretrained('facebook/bart-large')
        self.bert = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        for name in self.bert_for_texttosql.state_dict().keys():
            if name != 'model.keyword_embedding.weight' and not any(['model.decoder' in name]):
                rsetattr(self.bert, name, rgetattr(self.bert_for_texttosql, name))
        self.average_span_extractor = AverageSpanExtractor()
        self.column_mlp = nn.Linear(self.bert.config.d_model, self.bert.config.d_model)
        if 'col_type' in tasks:
            self.column_to_prob = nn.Linear(self.bert.config.d_model, 3)
        else:
            self.column_to_prob = nn.Linear(self.bert.config.d_model, 1)
        self.value_column_mlp = nn.Linear(self.bert.config.d_model * 2, self.bert.config.d_model)
        self.value_column_to_prob = nn.Linear(self.bert.config.d_model, 1)
        self.table_pred_mlp = nn.Linear(self.bert.config.d_model, 2)

    def column_prediction(self, input_ids, attention_mask, column_spans):
        column_mask = (column_spans[:, :, 0] > 0).long()
        features = self.bert.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0].contiguous()
        column_features = self.average_span_extractor(sequence_tensor=features, span_indices=column_spans, span_indices_mask=column_mask)
        column_selection_logits = self.column_to_prob(torch.relu(self.column_mlp(column_features)))
        column_selection_prob = torch.sigmoid(column_selection_logits)
        return column_selection_prob

    def column_classification(self, input_ids, attention_mask, column_spans):
        column_mask = (column_spans[:, :, 0] > 0).long()
        features = self.bert.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0].contiguous()
        column_features = self.average_span_extractor(sequence_tensor=features, span_indices=column_spans, span_indices_mask=column_mask)
        column_selection_logits = self.column_to_prob(torch.relu(self.column_mlp(column_features)))
        return column_selection_logits

    def value_prediction(self, input_ids, attention_mask, column_spans, value_spans):
        column_mask = (column_spans[:, :, 0] > 0).long()
        features = self.bert.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0].contiguous()
        column_features = self.average_span_extractor(sequence_tensor=features, span_indices=column_spans, span_indices_mask=column_mask)
        value_feature = self.average_span_extractor(sequence_tensor=features, span_indices=value_spans)
        column_features = torch.cat([column_features, value_feature.expand(column_features.size())], dim=-1)
        column_selection_logits = self.value_column_to_prob(torch.relu(self.value_column_mlp(column_features)))
        column_selection_prob = torch.sigmoid(column_selection_logits)
        return column_selection_prob

    def table_pred(self, input_ids, attention_mask):
        features = self.bert.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0].contiguous()
        logits = self.table_pred_mlp(features[:, 0])
        return logits

    def forward(self, *input, **kwargs):
        input_ids = kwargs.pop('input_ids')
        pad_token_id = kwargs.pop('pad_token_id')
        attention_mask = (input_ids != pad_token_id).long()
        if self.training:
            task = kwargs.pop('task')
            if task == 'text2sql':
                copy_span = None
                column_spans = kwargs.pop('column_spans')
                label_ids = kwargs.pop('labels')
                label_padding_id = kwargs.pop('label_padding_id')
                y_ids = label_ids[:, :-1].contiguous()
                lm_labels = label_ids[:, 1:].clone()
                lm_labels[label_ids[:, 1:] == label_padding_id] = -100
                outputs = self.bert_for_texttosql(input_ids, column_spans=column_spans, copy_span=copy_span, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
                return outputs[0],
            if task == 'mlm' or task == 'col_rev':
                output_ids = kwargs.pop('labels')
                y_ids = output_ids[:, :-1].contiguous()
                lm_labels = output_ids[:, 1:].clone()
                lm_labels[output_ids[:, 1:] == pad_token_id] = -100
                outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
                return outputs[0],
            if task == 'recurring_mlm':
                y_ids = kwargs.pop('y_ids')
                output_ids = kwargs.pop('labels')
                lm_labels = output_ids[:, 1:].clone()
                lm_labels[output_ids[:, 1:] == pad_token_id] = -100
                outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
                return outputs[0],
            if task == 'col_pred':
                label_ids = kwargs.pop('labels')
                column_spans = kwargs.pop('column_spans')
                column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)
                label_mask = column_spans.view(-1, 2)[:, 0] > 0
                column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask], label_ids.view(-1)[label_mask].float(), reduction='sum') / label_ids.size(0)
                return column_selection_loss,
            if task == 'col_type':
                label_ids = kwargs.pop('labels')
                column_spans = kwargs.pop('column_spans')
                column_selection_prob = self.column_classification(input_ids, attention_mask, column_spans)
                label_mask = column_spans.view(-1, 2)[:, 0] > 0
                column_selection_loss = F.cross_entropy(column_selection_prob.view(-1, 3)[label_mask], label_ids.view(-1)[label_mask], reduction='sum') / label_ids.size(0)
                return column_selection_loss,
            if task == 'value_pred':
                label_ids = kwargs.pop('labels')
                column_spans = kwargs.pop('column_spans')
                value_spans = kwargs.pop('value_spans')
                column_selection_prob = self.value_prediction(input_ids, attention_mask, column_spans, value_spans)
                label_mask = column_spans.view(-1, 2)[:, 0] > 0
                column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask], label_ids.view(-1)[label_mask].float(), reduction='sum') / label_ids.size(0)
                return column_selection_loss,
            if task == 'table_pred':
                label_ids = kwargs.pop('labels')
                table_prediction_prob = self.table_pred(input_ids, attention_mask)
                table_prediction_loss = F.cross_entropy(table_prediction_prob.view(-1, 2), label_ids.view(-1), reduction='sum') / label_ids.size(0)
                return table_prediction_loss,
            raise NotImplementedError('Unknown task {}'.format(task))
        else:
            task = kwargs.pop('task')
            if task == 'text2sql':
                copy_span = None
                column_spans = kwargs.pop('column_spans')
                label_eos_id = kwargs.pop('label_eos_id')
                label_bos_id = kwargs.pop('label_bos_id')
                label_padding_id = kwargs.pop('label_padding_id')
                generated_ids = self.bert_for_texttosql.generate(input_ids=input_ids, column_spans=column_spans, copy_span=copy_span, attention_mask=attention_mask, num_beams=1, max_length=30, length_penalty=2.0, early_stopping=True, use_cache=True, decoder_start_token_id=label_bos_id, eos_token_id=label_eos_id, pad_token_id=label_padding_id, vocab_size=len(KEYWORDS))
                output_ids = kwargs.pop('labels')
                y_ids = output_ids[:, :-1].contiguous()
                lm_labels = output_ids[:, 1:].clone()
                lm_labels[output_ids[:, 1:] == label_padding_id] = -100
                outputs = self.bert_for_texttosql(input_ids, column_spans=column_spans, attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
                return outputs[0].detach(), generated_ids
            if task == 'recurring_mlm':
                label_eos_id = kwargs.pop('label_eos_id')
                label_bos_id = kwargs.pop('label_bos_id')
                label_padding_id = kwargs.pop('label_padding_id')
                generated_ids = self.bert.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=3, max_length=input_ids.size(1) + 5, length_penalty=2.0, early_stopping=True, use_cache=True, decoder_start_token_id=label_bos_id, eos_token_id=label_eos_id, pad_token_id=label_padding_id)
                generated_ids = generated_ids[:, 1:].contiguous()
                y_ids = kwargs.pop('y_ids')
                output_ids = kwargs.pop('labels')
                lm_labels = output_ids[:, 1:].clone()
                lm_labels[output_ids[:, 1:] == pad_token_id] = -100
                outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
                return outputs[0].detach(), generated_ids
            if task == 'mlm' or task == 'col_rev':
                label_eos_id = kwargs.pop('label_eos_id')
                label_bos_id = kwargs.pop('label_bos_id')
                label_padding_id = kwargs.pop('label_padding_id')
                generated_ids = self.bert.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=3, max_length=input_ids.size(1) + 5, length_penalty=2.0, early_stopping=True, use_cache=True, decoder_start_token_id=label_bos_id, eos_token_id=label_eos_id, pad_token_id=label_padding_id)
                generated_ids = generated_ids[:, 1:].contiguous()
                output_ids = kwargs.pop('labels')
                y_ids = output_ids[:, :-1].contiguous()
                lm_labels = output_ids[:, 1:].clone()
                lm_labels[output_ids[:, 1:] == label_padding_id] = -100
                outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, labels=lm_labels)
                return outputs[0].detach(), generated_ids
            if task == 'col_pred':
                label_ids = kwargs.pop('labels')
                column_spans = kwargs.pop('column_spans')
                column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)
                generated_ids = (column_selection_prob.squeeze(-1) > 0.5).long()
                generated_ids[column_spans[:, :, 0] == 0] = -100
                label_mask = column_spans.view(-1, 2)[:, 0] > 0
                column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask], label_ids.view(-1)[label_mask].float(), reduction='sum') / label_ids.size(0)
                return column_selection_loss.detach(), generated_ids
            if task == 'col_type':
                label_ids = kwargs.pop('labels')
                column_spans = kwargs.pop('column_spans')
                column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)
                generated_ids = column_selection_prob.argmax(dim=-1)
                generated_ids[column_spans[:, :, 0] == 0] = -100
                label_mask = column_spans.view(-1, 2)[:, 0] > 0
                column_selection_loss = F.cross_entropy(column_selection_prob.view(-1, 3)[label_mask], label_ids.view(-1)[label_mask], reduction='sum') / label_ids.size(0)
                return column_selection_loss.detach(), generated_ids
            if task == 'value_pred':
                label_ids = kwargs.pop('labels')
                column_spans = kwargs.pop('column_spans')
                value_spans = kwargs.pop('value_spans')
                column_selection_prob = self.value_prediction(input_ids, attention_mask, column_spans, value_spans)
                generated_ids = (column_selection_prob.squeeze(-1) > 0.5).long()
                generated_ids[column_spans[:, :, 0] == 0] = -100
                label_mask = column_spans.view(-1, 2)[:, 0] > 0
                column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask], label_ids.view(-1)[label_mask].float(), reduction='sum') / label_ids.size(0)
                return column_selection_loss.detach(), generated_ids
            if task == 'table_pred':
                label_ids = kwargs.pop('labels')
                table_prediction_prob = self.table_pred(input_ids, attention_mask)
                generated_ids = table_prediction_prob.argmax(dim=-1).unsqueeze(-1)
                table_prediction_loss = F.cross_entropy(table_prediction_prob.view(-1, 2), label_ids.view(-1), reduction='sum') / label_ids.size(0)
                return table_prediction_loss.detach(), generated_ids
            raise NotImplementedError()

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
    """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info('Model weights saved in {}'.format(output_model_file))


class TaBARTModel(nn.Module):
    """
  output: tuple: (loss, ) in training
  """

    def __init__(self):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        self.average_span_extractor = AverageSpanExtractor()
        self.column_mlp = nn.Linear(self.bert.config.d_model, self.bert.config.d_model)
        self.column_to_prob = nn.Linear(self.bert.config.d_model, 1)

    def column_prediction(self, input_ids, attention_mask, column_spans):
        column_mask = (column_spans[:, :, 0] > 0).long()
        features = self.bert.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0].contiguous()
        column_features = self.average_span_extractor(sequence_tensor=features, span_indices=column_spans, span_indices_mask=column_mask)
        column_selection_logits = self.column_to_prob(torch.relu(self.column_mlp(column_features)))
        column_selection_prob = torch.sigmoid(column_selection_logits)
        return column_selection_prob

    def forward(self, *input, **kwargs):
        input_ids = kwargs.pop('input_ids')
        pad_token_id = kwargs.pop('pad_token_id')
        attention_mask = (input_ids != pad_token_id).long()
        if self.training:
            task = kwargs.pop('task')
            if task == 'mlm':
                output_ids = kwargs.pop('labels')
                y_ids = output_ids[:, :-1].contiguous()
                lm_labels = output_ids[:, 1:].clone()
                lm_labels[output_ids[:, 1:] == pad_token_id] = -100
                outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
                return outputs[0],
            elif task == 'col_pred':
                label_ids = kwargs.pop('labels')
                column_spans = kwargs.pop('column_spans')
                column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)
                label_mask = column_spans.view(-1, 2)[:, 0] > 0
                column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask], label_ids.view(-1)[label_mask].float(), reduction='sum') / label_ids.size(0)
                return column_selection_loss,
            else:
                raise NotImplementedError('Unknown task {}'.format(task))
        else:
            task = kwargs.pop('task')
            if task == 'mlm':
                label_eos_id = kwargs.pop('label_eos_id')
                label_bos_id = kwargs.pop('label_bos_id')
                label_padding_id = kwargs.pop('label_padding_id')
                generated_ids = self.bert.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=3, max_length=input_ids.size(1) + 5, length_penalty=2.0, early_stopping=True, use_cache=True, decoder_start_token_id=label_bos_id, eos_token_id=label_eos_id, pad_token_id=label_padding_id)
                output_ids = kwargs.pop('labels')
                y_ids = output_ids[:, :-1].contiguous()
                lm_labels = output_ids[:, 1:].clone()
                lm_labels[output_ids[:, 1:] == pad_token_id] = -100
                outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
                return outputs[0].detach(), generated_ids
            elif task == 'col_pred':
                label_ids = kwargs.pop('labels')
                column_spans = kwargs.pop('column_spans')
                column_selection_prob = self.column_prediction(input_ids, attention_mask, column_spans)
                generated_ids = (column_selection_prob.squeeze(-1) > 0.5).long()
                generated_ids[column_spans[:, :, 0] == 0] = -100
                label_mask = column_spans.view(-1, 2)[:, 0] > 0
                column_selection_loss = F.binary_cross_entropy(column_selection_prob.view(-1)[label_mask], label_ids.view(-1)[label_mask].float(), reduction='sum') / label_ids.size(0)
                return column_selection_loss.detach(), generated_ids
            else:
                raise NotImplementedError()

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
    """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info('Model weights saved in {}'.format(output_model_file))


class SQL2TextModel(nn.Module):
    """
  output: tuple: (loss, ) in training
  """

    def __init__(self):
        super().__init__()
        self.bert = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    def forward(self, *input, **kwargs):
        input_ids = kwargs.pop('input_ids')
        pad_token_id = kwargs.pop('pad_token_id')
        attention_mask = (input_ids != pad_token_id).long()
        if self.training:
            output_ids = kwargs.pop('labels')
            y_ids = output_ids[:, :-1].contiguous()
            lm_labels = output_ids[:, 1:].clone()
            lm_labels[output_ids[:, 1:] == pad_token_id] = -100
            outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            return outputs[0],
        else:
            label_eos_id = kwargs.pop('label_eos_id')
            label_bos_id = kwargs.pop('label_bos_id')
            label_padding_id = kwargs.pop('label_padding_id')
            generated_ids = self.bert.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=3, max_length=60, length_penalty=2.0, early_stopping=True, use_cache=True, decoder_start_token_id=label_bos_id, eos_token_id=label_eos_id, pad_token_id=label_padding_id)
            output_ids = kwargs.pop('labels')
            y_ids = output_ids[:, :-1].contiguous()
            lm_labels = output_ids[:, 1:].clone()
            lm_labels[output_ids[:, 1:] == pad_token_id] = -100
            outputs = self.bert(input_ids, attention_mask=attention_mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
            return outputs[0].detach(), generated_ids

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory: directory to which to save.
    """
        assert os.path.isdir(save_directory), 'Saving path should be a directory where the model and configuration can be saved'
        model_to_save = self.module if hasattr(self, 'module') else self
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info('Model weights saved in {}'.format(output_model_file))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BahdanauAttention,
     lambda: ([], {'query_size': 4, 'value_size': 4, 'proj_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {})),
    (BahdanauPointer,
     lambda: ([], {'query_size': 4, 'key_size': 4, 'proj_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {})),
    (BartClassificationHead,
     lambda: ([], {'input_dim': 4, 'inner_dim': 4, 'num_classes': 4, 'pooler_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LearnedPositionalEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4, 'padding_idx': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiHeadedAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SublayerConnection,
     lambda: ([], {'size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.nn.ReLU()], {})),
]

