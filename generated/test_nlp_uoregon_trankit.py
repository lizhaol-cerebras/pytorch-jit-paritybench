
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


import torch.nn.functional as F


from copy import deepcopy


from collections import defaultdict


import logging


import re


import math


from numbers import Number


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import time


from collections import Counter


import random


import torch


import torch.nn as nn


import numpy as np


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def set_cuda(var, cuda):
    if cuda:
        return var
    return var


class CRFLoss(nn.Module):
    """
    Calculate log-space crf loss, given unary potentials, a transition matrix
    and gold tag sequences.
    """

    def __init__(self, num_tag, batch_average=True):
        super().__init__()
        self._transitions = nn.Parameter(torch.zeros(num_tag, num_tag))
        self._batch_average = batch_average

    def forward(self, inputs, masks, tag_indices):
        """
        inputs: batch_size x seq_len x num_tags
        masks: batch_size x seq_len
        tag_indices: batch_size x seq_len
        @return:
            loss: CRF negative log likelihood on all instances.
            transitions: the transition matrix
        """
        self.bs, self.sl, self.nc = inputs.size()
        unary_scores = self.crf_unary_score(inputs, masks, tag_indices)
        binary_scores = self.crf_binary_score(inputs, masks, tag_indices)
        log_norm = self.crf_log_norm(inputs, masks, tag_indices)
        log_likelihood = unary_scores + binary_scores - log_norm
        loss = torch.sum(-log_likelihood)
        if self._batch_average:
            loss = loss / self.bs
        else:
            total = masks.eq(0).sum()
            loss = loss / (total + 1e-08)
        return loss, self._transitions

    def crf_unary_score(self, inputs, masks, tag_indices):
        """
        @return:
            unary_scores: batch_size
        """
        flat_inputs = inputs.view(self.bs, -1)
        flat_tag_indices = tag_indices + set_cuda(torch.arange(self.sl).long().unsqueeze(0) * self.nc, tag_indices.is_cuda)
        unary_scores = torch.gather(flat_inputs, 1, flat_tag_indices).view(self.bs, -1)
        unary_scores.masked_fill_(masks, 0)
        return unary_scores.sum(dim=1)

    def crf_binary_score(self, inputs, masks, tag_indices):
        """
        @return:
            binary_scores: batch_size
        """
        nt = tag_indices.size(-1) - 1
        start_indices = tag_indices[:, :nt]
        end_indices = tag_indices[:, 1:]
        flat_transition_indices = start_indices * self.nc + end_indices
        flat_transition_indices = flat_transition_indices.view(-1)
        flat_transition_matrix = self._transitions.view(-1)
        binary_scores = torch.gather(flat_transition_matrix, 0, flat_transition_indices).view(self.bs, -1)
        score_masks = masks[:, 1:]
        binary_scores.masked_fill_(score_masks, 0)
        return binary_scores.sum(dim=1)

    def crf_log_norm(self, inputs, masks, tag_indices):
        """
        Calculate the CRF partition in log space for each instance, following:
            http://www.cs.columbia.edu/~mcollins/fb.pdf
        @return:
            log_norm: batch_size
        """
        start_inputs = inputs[:, 0, :]
        rest_inputs = inputs[:, 1:, :]
        rest_masks = masks[:, 1:]
        alphas = start_inputs
        trans = self._transitions.unsqueeze(0)
        for i in range(rest_inputs.size(1)):
            transition_scores = alphas.unsqueeze(2) + trans
            new_alphas = rest_inputs[:, i, :] + log_sum_exp(transition_scores, dim=1)
            m = rest_masks[:, i].unsqueeze(1).expand_as(new_alphas)
            new_alphas.masked_scatter_(m, alphas.masked_select(m))
            alphas = new_alphas
        log_norm = log_sum_exp(alphas, dim=1)
        return log_norm


INFINITY_NUMBER = 65504


class BasicAttention(nn.Module):
    """
    A basic MLP attention layer.
    """

    def __init__(self, dim):
        super(BasicAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input)
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        attn = target.unsqueeze(1).expand_as(context) + source
        attn = self.tanh(attn)
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)
        if mask is not None:
            attn.masked_fill_(mask, -INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)
        attn = torch.bmm(context, target).squeeze(2)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LinearAttention(nn.Module):
    """ A linear attention form, inspired by BiDAF:
        a = W (u; v; u o v)
    """

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.linear = nn.Linear(dim * 3, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)
        v = context.contiguous().view(-1, dim)
        attn_in = torch.cat((u, v, u.mul(v)), 1)
        attn = self.linear(attn_in).view(batch_size, source_len)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(batch_size, 1, source_len)
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class DeepAttention(nn.Module):
    """ A deep attention form, invented by Robert:
        u = ReLU(Wx)
        v = ReLU(Wy)
        a = V.(u o v)
    """

    def __init__(self, dim):
        super(DeepAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)
        u = self.relu(self.linear_in(u))
        v = self.relu(self.linear_in(context.contiguous().view(-1, dim)))
        attn = self.linear_v(u.mul(v)).view(batch_size, source_len)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(batch_size, 1, source_len)
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LSTMAttention(nn.Module):
    """A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='soft'):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        if attn_type == 'soft':
            self.attention_layer = SoftDotAttention(hidden_size)
        elif attn_type == 'mlp':
            self.attention_layer = BasicAttention(hidden_size)
        elif attn_type == 'linear':
            self.attention_layer = LinearAttention(hidden_size)
        elif attn_type == 'deep':
            self.attention_layer = DeepAttention(hidden_size)
        else:
            raise Exception('Unsupported LSTM attention type: {}'.format(attn_type))

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        if self.batch_first:
            input = input.transpose(0, 1)
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.lstm_cell(input[i], hidden)
            hy, cy = hidden
            h_tilde, alpha = self.attention_layer(hy, ctx, mask=ctx_mask)
            output.append(h_tilde)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, hidden


EOS_ID = 3


PAD_ID = 0


SOS_ID = 2


class Beam:
    """
     Adapted and modified from the OpenNMT project.

     Class for managing the internals of the beam search process.


             hyp1-hyp1---hyp1 -hyp1
                     \\             /
             hyp2 \\-hyp2 /-hyp2hyp2
                                   /                   hyp3-hyp3---hyp3 -hyp3
             ========================

     Takes care of beams, back pointers, and scores.
    """

    def __init__(self, size, cuda=False):
        self.size = size
        self.done = False
        self.tt = torch.cuda if cuda else torch
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []
        self.prevKs = []
        self.nextYs = [self.tt.LongTensor(size).fill_(PAD_ID)]
        self.nextYs[0][0] = SOS_ID
        self.copy = []

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.nextYs[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prevKs[-1]

    def advance(self, wordLk, copy_indices=None):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `copy_indices` - copy indices (K x ctx_len)

        Returns: True if beam search is complete.
        """
        if self.done:
            return True
        numWords = wordLk.size(1)
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        if copy_indices is not None:
            self.copy.append(copy_indices.index_select(0, prevK))
        if self.nextYs[-1][0] == EOS_ID:
            self.done = True
            self.allScores.append(self.scores)
        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """
        Walk back to construct the full hypothesis.

        Parameters:

             * `k` - the position in the beam to construct.

         Returns: The hypothesis
        """
        hyp = []
        cpy = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            if len(self.copy) > 0:
                cpy.append(self.copy[j][k])
            k = self.prevKs[j][k]
        hyp = hyp[::-1]
        cpy = cpy[::-1]
        for i, cidx in enumerate(cpy):
            if cidx >= 0:
                hyp[i] = -(cidx + 1)
        return hyp


EMB_INIT_RANGE = 1.0


def prune_hyp(hyp):
    """
    Prune a decoded hypothesis
    """
    if EOS_ID in hyp:
        idx = hyp.index(EOS_ID)
        return hyp[:idx]
    else:
        return hyp


class Seq2SeqModel(nn.Module):
    """
    A complete encoder-decoder model, with optional attention.
    """

    def __init__(self, args, emb_matrix=None, use_cuda=False, training_mode=False):
        super().__init__()
        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.hidden_dim = args['hidden_dim']
        self.nlayers = args['num_layers']
        self.emb_dropout = args.get('emb_dropout', 0.0)
        self.dropout = args['dropout']
        self.pad_token = PAD_ID
        self.max_dec_len = args['max_dec_len']
        self.use_cuda = use_cuda
        self.training_mode = training_mode
        self.top = args.get('top', 10000000000.0)
        self.args = args
        self.emb_matrix = emb_matrix
        self.num_directions = 2
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim
        self.use_pos = args.get('pos', False)
        self.pos_dim = args.get('pos_dim', 0)
        self.pos_vocab_size = args.get('pos_vocab_size', 0)
        self.pos_dropout = args.get('pos_dropout', 0)
        self.edit = args.get('edit', False)
        self.num_edit = args.get('num_edit', 0)
        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.drop = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)
        self.encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, self.nlayers, bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.decoder = LSTMAttention(self.emb_dim, self.dec_hidden_dim, batch_first=True, attn_type=self.args['attn_type'])
        self.dec2vocab = nn.Linear(self.dec_hidden_dim, self.vocab_size)
        if self.use_pos and self.pos_dim > 0:
            self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_dim, self.pad_token)
            self.pos_drop = nn.Dropout(self.pos_dropout)
        if self.edit:
            edit_hidden = self.hidden_dim // 2
            self.edit_clf = nn.Sequential(nn.Linear(self.hidden_dim, edit_hidden), nn.ReLU(), nn.Linear(edit_hidden, self.num_edit))
        self.SOS_tensor = torch.LongTensor([SOS_ID])
        self.SOS_tensor = self.SOS_tensor if self.use_cuda else self.SOS_tensor
        self.init_weights()

    def init_weights(self):
        init_range = EMB_INIT_RANGE
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == (self.vocab_size, self.emb_dim), 'Input embedding matrix must match size: {} x {}'.format(self.vocab_size, self.emb_dim)
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            self.embedding.weight.data.uniform_(-init_range, init_range)
        if self.use_pos:
            self.pos_embedding.weight.data.uniform_(-init_range, init_range)

    def cuda(self):
        super()
        self.use_cuda = True

    def cpu(self):
        super().cpu()
        self.use_cuda = False

    def zero_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(self.encoder.num_layers * 2, batch_size, self.enc_hidden_dim, requires_grad=False)
        c0 = torch.zeros(self.encoder.num_layers * 2, batch_size, self.enc_hidden_dim, requires_grad=False)
        if self.use_cuda:
            if self.training_mode:
                return h0, c0
            else:
                return h0.half(), c0.half()
        return h0, c0

    def encode(self, enc_inputs, lens):
        """ Encode source sequence. """
        self.h0, self.c0 = self.zero_state(enc_inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(enc_inputs, lens, batch_first=True)
        packed_h_in, (hn, cn) = self.encoder(packed_inputs, (self.h0, self.c0))
        h_in, _ = nn.utils.rnn.pad_packed_sequence(packed_h_in, batch_first=True)
        hn = torch.cat((hn[-1], hn[-2]), 1)
        cn = torch.cat((cn[-1], cn[-2]), 1)
        return h_in, (hn, cn)

    def decode(self, dec_inputs, hn, cn, ctx, ctx_mask=None):
        """ Decode a step, based on context encoding and source context states."""
        dec_hidden = hn, cn
        h_out, dec_hidden = self.decoder(dec_inputs, dec_hidden, ctx, ctx_mask)
        h_out_reshape = h_out.contiguous().view(h_out.size(0) * h_out.size(1), -1)
        decoder_logits = self.dec2vocab(h_out_reshape)
        decoder_logits = decoder_logits.view(h_out.size(0), h_out.size(1), -1)
        log_probs = self.get_log_prob(decoder_logits)
        return log_probs, dec_hidden

    def forward(self, src, src_mask, tgt_in, pos=None):
        batch_size = src.size(0)
        enc_inputs = self.emb_drop(self.embedding(src))
        dec_inputs = self.emb_drop(self.embedding(tgt_in))
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(0).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        log_probs, _ = self.decode(dec_inputs, hn, cn, h_in, src_mask)
        return log_probs, edit_logits

    def get_log_prob(self, logits):
        logits_reshape = logits.view(-1, self.vocab_size)
        log_probs = F.log_softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return log_probs
        return log_probs.view(logits.size(0), logits.size(1), logits.size(2))

    def predict_greedy(self, src, src_mask, pos=None):
        """ Predict with greedy decoding. """
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(PAD_ID).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        dec_inputs = self.embedding(self.SOS_tensor)
        dec_inputs = dec_inputs.expand(batch_size, dec_inputs.size(0), dec_inputs.size(1))
        done = [(False) for _ in range(batch_size)]
        total_done = 0
        max_len = 0
        output_seqs = [[] for _ in range(batch_size)]
        while total_done < batch_size and max_len < self.max_dec_len:
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask)
            assert log_probs.size(1) == 1, 'Output must have 1-step of output.'
            _, preds = log_probs.squeeze(1).max(1, keepdim=True)
            dec_inputs = self.embedding(preds)
            max_len += 1
            for i in range(batch_size):
                if not done[i]:
                    token = preds.data[i][0].item()
                    if token == EOS_ID:
                        done[i] = True
                        total_done += 1
                    else:
                        output_seqs[i].append(token)
        return output_seqs, edit_logits

    def predict(self, src, src_mask, pos=None, beam_size=5):
        """ Predict with beam search. """
        if beam_size == 1:
            return self.predict_greedy(src, src_mask, pos=pos)
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(PAD_ID).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        with torch.no_grad():
            h_in = h_in.data.repeat(beam_size, 1, 1)
            src_mask = src_mask.repeat(beam_size, 1)
            hn = hn.data.repeat(beam_size, 1)
            cn = cn.data.repeat(beam_size, 1)
        beam = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]

        def update_state(states, idx, positions, beam_size):
            """ Select the states according to back pointers. """
            for e in states:
                br, d = e.size()
                s = e.contiguous().view(beam_size, br // beam_size, d)[:, idx]
                s.data.copy_(s.data.index_select(0, positions))
        for i in range(self.max_dec_len):
            dec_inputs = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            dec_inputs = self.embedding(dec_inputs)
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask)
            log_probs = log_probs.view(beam_size, batch_size, -1).transpose(0, 1).contiguous()
            done = []
            for b in range(batch_size):
                is_done = beam[b].advance(log_probs.data[b])
                if is_done:
                    done += [b]
                update_state((hn, cn), b, beam[b].get_current_origin(), beam_size)
            if len(done) == batch_size:
                break
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[0]]
            k = ks[0]
            hyp = beam[b].get_hyp(k)
            hyp = prune_hyp(hyp)
            hyp = [i.item() for i in hyp]
            all_hyp += [hyp]
        return all_hyp, edit_logits


def word_lens_to_idxs_fast(token_lens):
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([(i + offset) for i in range(token_len)] + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


class Base_Model(nn.Module):

    def __init__(self, config, task_name):
        super().__init__()
        self.config = config
        self.task_name = task_name
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.xlmr = XLMRobertaModel.from_pretrained(config.embedding_name, cache_dir=os.path.join(config._cache_dir, config.embedding_name), output_hidden_states=True)
        adapters.init(self.xlmr)
        self.xlmr_dropout = nn.Dropout(p=config.embedding_dropout)
        task_config = AdapterConfig.load('pfeiffer', reduction_factor=6 if config.embedding_name == 'xlm-roberta-base' else 4)
        self.xlmr.add_adapter(task_name, config=task_config)
        self.xlmr.train_adapter([task_name])
        self.xlmr.set_active_adapters([task_name])

    def encode(self, piece_idxs, attention_masks):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]
        wordpiece_reprs = xlmr_outputs[:, 1:-1, :]
        wordpiece_reprs = self.xlmr_dropout(wordpiece_reprs)
        return wordpiece_reprs

    def encode_words(self, piece_idxs, attention_masks, word_lens):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]
        cls_reprs = xlmr_outputs[:, 0, :].unsqueeze(1)
        idxs, masks, token_num, token_len = word_lens_to_idxs_fast(word_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.xlmr_dim) + 1
        masks = xlmr_outputs.new(masks).unsqueeze(-1)
        xlmr_outputs = torch.gather(xlmr_outputs, 1, idxs) * masks
        xlmr_outputs = xlmr_outputs.view(batch_size, token_num, token_len, self.xlmr_dim)
        xlmr_outputs = xlmr_outputs.sum(2)
        return xlmr_outputs, cls_reprs

    def forward(self, batch):
        raise NotImplementedError


class Multilingual_Embedding(Base_Model):

    def __init__(self, config, model_name='embedding'):
        super(Multilingual_Embedding, self).__init__(config, task_name=model_name)

    def get_tokenizer_inputs(self, batch):
        wordpiece_reprs = self.encode(piece_idxs=batch.piece_idxs, attention_masks=batch.attention_masks)
        return wordpiece_reprs

    def get_tagger_inputs(self, batch):
        word_reprs, cls_reprs = self.encode_words(piece_idxs=batch.piece_idxs, attention_masks=batch.attention_masks, word_lens=batch.word_lens)
        return word_reprs, cls_reprs


class Deep_Biaffine(nn.Module):
    """
    implemented based on the paper https://arxiv.org/abs/1611.01734
    """

    def __init__(self, in_dim1, in_dim2, hidden_dim, output_dim):
        super().__init__()
        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn1 = nn.Sequential(nn.Linear(in_dim1, hidden_dim), nn.ReLU(), nn.Dropout(0.5))
        self.ffn2 = nn.Sequential(nn.Linear(in_dim2, hidden_dim), nn.ReLU(), nn.Dropout(0.5))
        self.pairwise_weight = nn.Parameter(torch.Tensor(in_dim1 + 1, in_dim2 + 1, output_dim))
        self.pairwise_weight.data.zero_()

    def forward(self, x1, x2):
        h1 = self.ffn1(x1)
        h2 = self.ffn2(x2)
        g1 = torch.cat([h1, h1.new_ones(*h1.size()[:-1], 1)], len(h1.size()) - 1)
        g2 = torch.cat([h2, h2.new_ones(*h2.size()[:-1], 1)], len(h2.size()) - 1)
        g1_size = g1.size()
        g2_size = g2.size()
        g1_w = torch.mm(g1.view(-1, g1_size[-1]), self.pairwise_weight.view(-1, (self.in_dim2 + 1) * self.output_dim))
        g2 = g2.transpose(1, 2)
        g1_w_g2 = g1_w.view(g1_size[0], g1_size[1] * self.output_dim, g2_size[2]).bmm(g2)
        g1_w_g2 = g1_w_g2.view(g1_size[0], g1_size[1], self.output_dim, g2_size[1]).transpose(2, 3)
        return g1_w_g2


class Linears(nn.Module):

    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias) for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


def viterbi_decode(scores, transition_params):
    """
    Decode a tag sequence with viterbi algorithm.
    scores: seq_len x num_tags (numpy array)
    transition_params: num_tags x num_tags (numpy array)
    @return:
        viterbi: a list of tag ids with highest score
        viterbi_score: the highest score
    """
    trellis = np.zeros_like(scores)
    backpointers = np.zeros_like(scores, dtype=np.int32)
    trellis[0] = scores[0]
    for t in range(1, scores.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = scores[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)
    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


class NERClassifier(nn.Module):

    def __init__(self, config, language):
        super().__init__()
        self.config = config
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.entity_label_stoi = config.ner_vocabs[language]
        self.entity_label_itos = {i: s for s, i in self.entity_label_stoi.items()}
        self.entity_label_num = len(self.entity_label_stoi)
        self.entity_label_ffn = Linears([self.xlmr_dim, config.hidden_num, self.entity_label_num], dropout_prob=config.linear_dropout, bias=config.linear_bias, activation=config.linear_activation)
        self.crit = CRFLoss(self.entity_label_num)
        if not config.training:
            self.initialized_weights = self.state_dict()
            self.pretrained_ner_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language, '{}.ner.mdl'.format(language)), map_location=self.config.device)['adapters']
            for name, value in self.pretrained_ner_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value
            self.load_state_dict(self.initialized_weights)
            None

    def forward(self, batch, word_reprs):
        batch_size, _, _ = word_reprs.size()
        logits = self.entity_label_ffn(word_reprs)
        loss, trans = self.crit(logits, batch.word_mask, batch.entity_label_idxs)
        return loss

    def predict(self, batch, word_reprs):
        batch_size, _, _ = word_reprs.size()
        logits = self.entity_label_ffn(word_reprs)
        _, trans = self.crit(logits, batch.word_mask, batch.entity_label_idxs)
        trans = trans.data.cpu().numpy()
        scores = logits.data.cpu().numpy()
        bs = logits.size(0)
        tag_seqs = []
        for i in range(bs):
            tags, _ = viterbi_decode(scores[i, :batch.word_num[i]], trans)
            tags = [self.entity_label_itos[t] for t in tags]
            tag_seqs += [tags]
        return tag_seqs


DEPREL = 'deprel'


FEATS = 'feats'


UPOS = 'upos'


XPOS = 'xpos'


lang2treebank = {'afrikaans': 'UD_Afrikaans-AfriBooms', 'ancient-greek-perseus': 'UD_Ancient_Greek-Perseus', 'ancient-greek': 'UD_Ancient_Greek-PROIEL', 'arabic': 'UD_Arabic-PADT', 'armenian': 'UD_Armenian-ArmTDP', 'basque': 'UD_Basque-BDT', 'belarusian': 'UD_Belarusian-HSE', 'bulgarian': 'UD_Bulgarian-BTB', 'catalan': 'UD_Catalan-AnCora', 'chinese': 'UD_Simplified_Chinese-GSDSimp', 'traditional-chinese': 'UD_Chinese-GSD', 'classical-chinese': 'UD_Classical_Chinese-Kyoto', 'croatian': 'UD_Croatian-SET', 'czech-cac': 'UD_Czech-CAC', 'czech-cltt': 'UD_Czech-CLTT', 'czech-fictree': 'UD_Czech-FicTree', 'czech': 'UD_Czech-PDT', 'danish': 'UD_Danish-DDT', 'dutch': 'UD_Dutch-Alpino', 'dutch-lassysmall': 'UD_Dutch-LassySmall', 'english': 'UD_English-EWT', 'english-gum': 'UD_English-GUM', 'english-lines': 'UD_English-LinES', 'english-partut': 'UD_English-ParTUT', 'estonian': 'UD_Estonian-EDT', 'estonian-ewt': 'UD_Estonian-EWT', 'finnish-ftb': 'UD_Finnish-FTB', 'finnish': 'UD_Finnish-TDT', 'french': 'UD_French-GSD', 'french-partut': 'UD_French-ParTUT', 'french-sequoia': 'UD_French-Sequoia', 'french-spoken': 'UD_French-Spoken', 'galician': 'UD_Galician-CTG', 'galician-treegal': 'UD_Galician-TreeGal', 'german': 'UD_German-GSD', 'german-hdt': 'UD_German-HDT', 'greek': 'UD_Greek-GDT', 'hebrew': 'UD_Hebrew-HTB', 'hindi': 'UD_Hindi-HDTB', 'hungarian': 'UD_Hungarian-Szeged', 'indonesian': 'UD_Indonesian-GSD', 'irish': 'UD_Irish-IDT', 'italian': 'UD_Italian-ISDT', 'italian-partut': 'UD_Italian-ParTUT', 'italian-postwita': 'UD_Italian-PoSTWITA', 'italian-twittiro': 'UD_Italian-TWITTIRO', 'italian-vit': 'UD_Italian-VIT', 'japanese': 'UD_Japanese-GSD', 'kazakh': 'UD_Kazakh-KTB', 'korean': 'UD_Korean-GSD', 'korean-kaist': 'UD_Korean-Kaist', 'kurmanji': 'UD_Kurmanji-MG', 'latin': 'UD_Latin-ITTB', 'latin-perseus': 'UD_Latin-Perseus', 'latin-proiel': 'UD_Latin-PROIEL', 'latvian': 'UD_Latvian-LVTB', 'lithuanian': 'UD_Lithuanian-ALKSNIS', 'lithuanian-hse': 'UD_Lithuanian-HSE', 'marathi': 'UD_Marathi-UFAL', 'norwegian-nynorsk': 'UD_Norwegian_Nynorsk-Nynorsk', 'norwegian-nynorsklia': 'UD_Norwegian_Nynorsk-NynorskLIA', 'norwegian-bokmaal': 'UD_Norwegian-Bokmaal', 'old-french': 'UD_Old_French-SRCMF', 'old-russian': 'UD_Old_Russian-TOROT', 'persian': 'UD_Persian-Seraji', 'polish-lfg': 'UD_Polish-LFG', 'polish': 'UD_Polish-PDB', 'portuguese': 'UD_Portuguese-Bosque', 'portuguese-gsd': 'UD_Portuguese-GSD', 'romanian-nonstandard': 'UD_Romanian-Nonstandard', 'romanian': 'UD_Romanian-RRT', 'russian-gsd': 'UD_Russian-GSD', 'russian': 'UD_Russian-SynTagRus', 'russian-taiga': 'UD_Russian-Taiga', 'scottish-gaelic': 'UD_Scottish_Gaelic-ARCOSG', 'serbian': 'UD_Serbian-SET', 'slovak': 'UD_Slovak-SNK', 'slovenian': 'UD_Slovenian-SSJ', 'slovenian-sst': 'UD_Slovenian-SST', 'spanish': 'UD_Spanish-AnCora', 'spanish-gsd': 'UD_Spanish-GSD', 'swedish-lines': 'UD_Swedish-LinES', 'swedish': 'UD_Swedish-Talbanken', 'tamil': 'UD_Tamil-TTB', 'telugu': 'UD_Telugu-MTG', 'turkish': 'UD_Turkish-IMST', 'ukrainian': 'UD_Ukrainian-IU', 'urdu': 'UD_Urdu-UDTB', 'uyghur': 'UD_Uyghur-UDT', 'vietnamese': 'UD_Vietnamese-VLSP', 'vietnamese-vtb': 'UD_Vietnamese-VTB', 'customized': 'UD_Customized', 'customized-mwt': 'UD_Customized-MWT', 'customized-ner': 'UD_Customized-NER', 'customized-mwt-ner': 'UD_Customized-MWT-NER'}


treebank2lang = {v: k for k, v in lang2treebank.items()}


class PosDepClassifier(nn.Module):

    def __init__(self, config, treebank_name):
        super().__init__()
        self.config = config
        self.vocabs = config.vocabs[treebank_name]
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.upos_embedding = nn.Embedding(num_embeddings=len(self.vocabs[UPOS]), embedding_dim=50)
        self.upos_ffn = nn.Linear(self.xlmr_dim, len(self.vocabs[UPOS]))
        self.xpos_ffn = nn.Linear(self.xlmr_dim + 50, len(self.vocabs[XPOS]))
        self.feats_ffn = nn.Linear(self.xlmr_dim, len(self.vocabs[FEATS]))
        self.down_dim = self.xlmr_dim // 4
        self.down_project = nn.Linear(self.xlmr_dim, self.down_dim)
        self.unlabeled = Deep_Biaffine(self.down_dim, self.down_dim, self.down_dim, 1)
        self.deprel = Deep_Biaffine(self.down_dim, self.down_dim, self.down_dim, len(self.vocabs[DEPREL]))
        self.criteria = torch.nn.CrossEntropyLoss()
        if not config.training:
            self.initialized_weights = self.state_dict()
            language = treebank2lang[treebank_name]
            self.pretrained_tagger_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language, '{}.tagger.mdl'.format(language)), map_location=self.config.device)['adapters']
            for name, value in self.pretrained_tagger_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value
            self.load_state_dict(self.initialized_weights)
            None

    def forward(self, batch, word_reprs, cls_reprs):
        upos_scores = self.upos_ffn(word_reprs)
        upos_scores = upos_scores.view(-1, len(self.vocabs[UPOS]))
        xpos_reprs = torch.cat([word_reprs, self.upos_embedding(batch.upos_ids)], dim=2)
        xpos_scores = self.xpos_ffn(xpos_reprs)
        xpos_scores = xpos_scores.view(-1, len(self.vocabs[XPOS]))
        feats_scores = self.feats_ffn(word_reprs)
        feats_scores = feats_scores.view(-1, len(self.vocabs[FEATS]))
        loss = self.criteria(upos_scores, batch.upos_type_idxs) + self.criteria(xpos_scores, batch.xpos_type_idxs) + self.criteria(feats_scores, batch.feats_type_idxs)
        dep_reprs = torch.cat([cls_reprs, word_reprs], dim=1)
        dep_reprs = self.down_project(dep_reprs)
        unlabeled_scores = self.unlabeled(dep_reprs, dep_reprs).squeeze(3)
        diag = torch.eye(batch.head_idxs.size(-1) + 1, dtype=torch.bool).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))
        unlabeled_scores = unlabeled_scores[:, 1:, :]
        unlabeled_scores = unlabeled_scores.masked_fill(batch.word_mask.unsqueeze(1), -float('inf'))
        unlabeled_target = batch.head_idxs.masked_fill(batch.word_mask[:, 1:], -100)
        loss += self.criteria(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)), unlabeled_target.view(-1))
        deprel_scores = self.deprel(dep_reprs, dep_reprs)
        deprel_scores = deprel_scores[:, 1:]
        deprel_scores = torch.gather(deprel_scores, 2, batch.head_idxs.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, len(self.vocabs[DEPREL]))).view(-1, len(self.vocabs[DEPREL]))
        deprel_target = batch.deprel_idxs.masked_fill(batch.word_mask[:, 1:], -100)
        loss += self.criteria(deprel_scores.contiguous(), deprel_target.view(-1))
        return loss

    def predict(self, batch, word_reprs, cls_reprs):
        upos_scores = self.upos_ffn(word_reprs)
        predicted_upos = torch.argmax(upos_scores, dim=2)
        xpos_reprs = torch.cat([word_reprs, self.upos_embedding(predicted_upos)], dim=2)
        xpos_scores = self.xpos_ffn(xpos_reprs)
        predicted_xpos = torch.argmax(xpos_scores, dim=2)
        feats_scores = self.feats_ffn(word_reprs)
        predicted_feats = torch.argmax(feats_scores, dim=2)
        dep_reprs = torch.cat([cls_reprs, word_reprs], dim=1)
        dep_reprs = self.down_project(dep_reprs)
        unlabeled_scores = self.unlabeled(dep_reprs, dep_reprs).squeeze(3)
        diag = torch.eye(batch.head_idxs.size(-1) + 1, dtype=torch.bool).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))
        deprel_scores = self.deprel(dep_reprs, dep_reprs)
        dep_preds = []
        dep_preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
        dep_preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())
        return predicted_upos, predicted_xpos, predicted_feats, dep_preds


class TokenizerClassifier(nn.Module):

    def __init__(self, config, treebank_name):
        super().__init__()
        self.config = config
        self.xlmr_dim = 768 if config.embedding_name == 'xlm-roberta-base' else 1024
        self.tokenizer_ffn = nn.Linear(self.xlmr_dim, 5)
        self.criteria = torch.nn.CrossEntropyLoss()
        if not config.training:
            language = treebank2lang[treebank_name]
            self.pretrained_tokenizer_weights = torch.load(os.path.join(self.config._cache_dir, self.config.embedding_name, language, '{}.tokenizer.mdl'.format(language)), map_location=self.config.device)['adapters']
            self.initialized_weights = self.state_dict()
            for name, value in self.pretrained_tokenizer_weights.items():
                if name in self.initialized_weights:
                    self.initialized_weights[name] = value
            self.load_state_dict(self.initialized_weights)
            None

    def forward(self, wordpiece_reprs, batch):
        wordpiece_scores = self.tokenizer_ffn(wordpiece_reprs)
        wordpiece_scores = wordpiece_scores.view(-1, 5)
        token_type_idxs = batch.token_type_idxs.view(-1)
        loss = self.criteria(wordpiece_scores, token_type_idxs)
        return loss

    def predict(self, batch, wordpiece_reprs):
        wordpiece_scores = self.tokenizer_ffn(wordpiece_reprs)
        predicted_wordpiece_labels = torch.argmax(wordpiece_scores, dim=2)
        return predicted_wordpiece_labels, batch.wordpiece_ends, batch.paragraph_index


def SequenceLoss(vocab_size):
    weight = torch.ones(vocab_size)
    weight[PAD_ID] = 0
    crit = nn.NLLLoss(weight)
    return crit


class MixLoss(nn.Module):
    """
    A mixture of SequenceLoss and CrossEntropyLoss.
    Loss = SequenceLoss + alpha * CELoss
    """

    def __init__(self, vocab_size, alpha):
        super().__init__()
        self.seq_loss = SequenceLoss(vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()
        assert alpha >= 0
        self.alpha = alpha

    def forward(self, seq_inputs, seq_targets, class_inputs, class_targets):
        sl = self.seq_loss(seq_inputs, seq_targets)
        cel = self.ce_loss(class_inputs, class_targets)
        loss = sl + self.alpha * cel
        return loss


class MaxEntropySequenceLoss(nn.Module):
    """
    A max entropy loss that encourage the model to have large entropy,
    therefore giving more diverse outputs.

    Loss = NLLLoss + alpha * EntropyLoss
    """

    def __init__(self, vocab_size, alpha):
        super().__init__()
        weight = torch.ones(vocab_size)
        weight[PAD_ID] = 0
        self.nll = nn.NLLLoss(weight)
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        inputs: [N, C]
        targets: [N]
        """
        assert inputs.size(0) == targets.size(0)
        nll_loss = self.nll(inputs, targets)
        mask = targets.eq(PAD_ID).unsqueeze(1).expand_as(inputs)
        masked_inputs = inputs.clone().masked_fill_(mask, 0.0)
        p = torch.exp(masked_inputs)
        ent_loss = p.mul(masked_inputs).sum() / inputs.size(0)
        loss = nll_loss + self.alpha * ent_loss
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (BasicAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {})),
    (DeepAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {})),
    (Deep_Biaffine,
     lambda: ([], {'in_dim1': 4, 'in_dim2': 4, 'hidden_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (LinearAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {})),
    (Linears,
     lambda: ([], {'dimensions': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

