
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


import logging


from typing import Optional


from collections import defaultdict


from collections import deque


import numpy as np


import torch


import random


from torch import nn


from torch.nn import CrossEntropyLoss


import time


import torch.nn as nn


from numpy import linalg as LA


from scipy.stats import spearmanr


from scipy.stats import pearsonr


import copy


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


import re


import inspect


from torch import optim


from typing import Tuple


from typing import Union


from torch.nn import BCEWithLogitsLoss


from torch.nn import MSELoss


from sklearn.metrics import accuracy_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import f1_score


import math


from typing import List


from torch.nn import LayerNorm


from typing import Callable


from typing import Sequence


import warnings


from typing import Dict


from torch import Tensor


from torch.nn import Parameter


from itertools import chain


from torch.utils.data import DataLoader


from itertools import groupby


class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']
        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1, bidirectional=True, dropout=self.dpout_model)
        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort) if self.is_cuda() else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        idx_unsort = torch.from_numpy(idx_unsort) if self.is_cuda() else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)
        if self.pool_type == 'mean':
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == 'max':
            if not self.max_pad:
                sent_output[sent_output == 0] = -1000000000.0
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        return emb

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        word_dict = {}
        sentences = [(s.split() if not tokenize else self.tokenize(s)) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        None
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')
                if k > K and all([(w in word_vec) for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        None

    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        None

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        None

    def get_batch(self, batch):
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
        return torch.FloatTensor(embed)

    def tokenize(self, s):
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [([self.bos] + s.split() + [self.eos] if not tokenize else [self.bos] + self.tokenize(s) + [self.eos]) for s in sentences]
        n_w = np.sum([len(x) for x in sentences])
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors.                                Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f
        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            None
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]
        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(sentences, bsize, tokenize, verbose)
        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            if self.is_cuda():
                batch = batch
            with torch.no_grad():
                batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]
        if verbose:
            None
        return embeddings

    def visualize(self, sent, tokenize=True):
        sent = sent.split() if not tokenize else self.tokenize(sent)
        sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]
        if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
            import warnings
            warnings.warn('No words in "%s" have w2v vectors. Replacing                            by "%s %s"..' % (sent, self.bos, self.eos))
        batch = self.get_batch(sent)
        if self.is_cuda():
            batch = batch
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum(idxs == k) for k in range(len(sent[0]))]
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [(100.0 * n / np.sum(argmaxs)) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()
        return output, idxs


class COCOProjNet(nn.Module):

    def __init__(self, config):
        super(COCOProjNet, self).__init__()
        self.imgdim = config['imgdim']
        self.sentdim = config['sentdim']
        self.projdim = config['projdim']
        self.imgproj = nn.Sequential(nn.Linear(self.imgdim, self.projdim))
        self.sentproj = nn.Sequential(nn.Linear(self.sentdim, self.projdim))

    def forward(self, img, sent, imgc, sentc):
        img = img.unsqueeze(1).expand_as(imgc).contiguous()
        img = img.view(-1, self.imgdim)
        imgc = imgc.view(-1, self.imgdim)
        sent = sent.unsqueeze(1).expand_as(sentc).contiguous()
        sent = sent.view(-1, self.sentdim)
        sentc = sentc.view(-1, self.sentdim)
        imgproj = self.imgproj(img)
        imgproj = imgproj / torch.sqrt(torch.pow(imgproj, 2).sum(1, keepdim=True)).expand_as(imgproj)
        imgcproj = self.imgproj(imgc)
        imgcproj = imgcproj / torch.sqrt(torch.pow(imgcproj, 2).sum(1, keepdim=True)).expand_as(imgcproj)
        sentproj = self.sentproj(sent)
        sentproj = sentproj / torch.sqrt(torch.pow(sentproj, 2).sum(1, keepdim=True)).expand_as(sentproj)
        sentcproj = self.sentproj(sentc)
        sentcproj = sentcproj / torch.sqrt(torch.pow(sentcproj, 2).sum(1, keepdim=True)).expand_as(sentcproj)
        anchor1 = torch.sum(imgproj * sentproj, 1)
        anchor2 = torch.sum(sentproj * imgproj, 1)
        img_sentc = torch.sum(imgproj * sentcproj, 1)
        sent_imgc = torch.sum(sentproj * imgcproj, 1)
        return anchor1, anchor2, img_sentc, sent_imgc

    def proj_sentence(self, sent):
        output = self.sentproj(sent)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output

    def proj_image(self, img):
        output = self.imgproj(img)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss
    """

    def __init__(self, margin):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor1, anchor2, img_sentc, sent_imgc):
        cost_sent = torch.clamp(self.margin - anchor1 + img_sentc, min=0.0).sum()
        cost_img = torch.clamp(self.margin - anchor2 + sent_imgc, min=0.0).sum()
        loss = cost_sent + cost_img
        return loss


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        if self.temp == 0:
            x = x.squeeze(1)
            y = y.squeeze(0)
            return torch.matmul(x, y.t())
        else:
            return self.cos(x, y) / self.temp


class CSSL(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cos_sim_fct = Similarity(temp=self.config.cl_temp)

    def multiple2one_pooling(self, src, index, dim=1, pool_type='amax'):
        return torch.zeros_like(src).scatter_reduce(dim, index[:, :, None].expand_as(src), src, reduce=pool_type, include_self=False)

    def eop_level_matrix_cl_loss(self, anchor_eop_features, cl_segment_ids):
        """
        eg. anchor_eop_features is [e1t1p1, e1t1p2, e1t1p3, e1t2p1, e1t3p1, e1t3p2, ...] where e means example, t means topic, p means paragraph
        we can construct similarity matrix like this:
            a1,     a2, a3, b1, c1, c2
        a1  ignore,  +,  +,  -,  -,  -
        a2  +     ignore +,  -,  -,  -
        a3  +,     +, ignore,-,  -,  -
        b1  -,      -,  -,  ignore, -,
        c1
        c2
        """
        loss = None
        total_topic_cnt = cl_segment_ids[-1] + 1
        total_eop_cnt = len(cl_segment_ids)
        accumulate_eop_cnts = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        d_cnt = {i: cl_segment_ids.count(i) for i in range(total_topic_cnt)}
        d_left = {i: accumulate_eop_cnts[i] for i in range(total_topic_cnt)}
        d_right = {i: (total_eop_cnt - d_cnt[i] - d_left[i]) for i in range(total_topic_cnt)}
        sim_mask_for_numerator, sim_mask_for_denominator = [], []
        for i, id_ in enumerate(cl_segment_ids):
            sim_mask = [False] * d_left[id_] + [True] * d_cnt[id_] + [False] * d_right[id_]
            sim_mask[i] = False
            sim_mask_for_numerator.append(sim_mask)
            sim_mask = [True] * d_left[id_] + [False] * d_cnt[id_] + [True] * d_right[id_]
            sim_mask_for_denominator.append(sim_mask)
        sim_mask_for_numerator = torch.tensor(sim_mask_for_numerator)
        sim_mask_for_denominator = torch.tensor(sim_mask_for_denominator)
        st_cos_sim = self.cos_sim_fct(anchor_eop_features.unsqueeze(1), anchor_eop_features.unsqueeze(0))
        exp_st_cos_sim = torch.exp(st_cos_sim)
        numerator = torch.sum(sim_mask_for_numerator * exp_st_cos_sim, 0)
        denominator = numerator + torch.sum(sim_mask_for_denominator * exp_st_cos_sim, 0)
        cl_prob = numerator / denominator
        if torch.isnan(cl_prob).any():
            None
        elif min(cl_prob[cl_prob != 0].shape) == 0:
            None
        else:
            cl_loss = -1 * torch.log(cl_prob[cl_prob != 0])
            loss = cl_loss.mean()
        return loss

    def eot_level_matrix_cl_loss(self, anchor_eop_features, cl_segment_ids):
        pass

    def cl_loss_for_list(self, anchor_eop_features, anchor_features, positive_eop_indices, negative_eop_indices):
        loss = torch.tensor(0.0)
        m_positive_similarity = []
        for i in range(self.config.cl_positive_k):
            positive_eop_features = anchor_eop_features[positive_eop_indices[i]]
            positive_similarity = self.cos_sim_fct(anchor_features, positive_eop_features)
            m_positive_similarity.append(positive_similarity.unsqueeze(0))
        m_positive_similarity = torch.cat(m_positive_similarity)
        n_negative_similarity = []
        for i in range(self.config.cl_negative_k):
            negative_eop_features = anchor_eop_features[negative_eop_indices[i]]
            negative_similarity = self.cos_sim_fct(anchor_features, negative_eop_features)
            n_negative_similarity.append(negative_similarity.unsqueeze(0))
        n_negative_similarity = torch.cat(n_negative_similarity)
        similarity_matrix = torch.cat((m_positive_similarity, n_negative_similarity))
        exp_similarity_matrix = torch.exp(similarity_matrix)
        sim_mask_for_numerator = [([1] * anchor_features.shape[0]) for _ in range(self.config.cl_positive_k)] + [([0] * anchor_features.shape[0]) for _ in range(self.config.cl_negative_k)]
        sim_mask_for_numerator = torch.tensor(sim_mask_for_numerator)
        numerator = torch.sum(exp_similarity_matrix * sim_mask_for_numerator, 0)
        denominator = torch.sum(exp_similarity_matrix, 0)
        cl_loss = -1 * torch.log(numerator / denominator)
        loss = cl_loss.mean()
        return loss

    def eop_level_list_cl_loss(self, anchor_eop_features, cl_segment_ids):
        total_topic_cnt = cl_segment_ids[-1] + 1
        total_eop_cnt = len(cl_segment_ids)
        accumulate_eop_cnts = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        bot_indices = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        eot_indices = [(v - 1) for v in accumulate_eop_cnts[1:]] + [total_eop_cnt - 1]
        topic_id2bot_eot_indices = {}
        for id_, (start_id, end_id) in enumerate(zip(bot_indices, eot_indices)):
            topic_id2bot_eot_indices[id_] = start_id, end_id
        positive_eop_indices = [[] for _ in range(self.config.cl_positive_k)]
        negative_eop_indices = [[] for _ in range(self.config.cl_negative_k)]
        for eop_index, eop_topic_id in enumerate(cl_segment_ids):
            start_id, end_id = topic_id2bot_eot_indices[eop_topic_id]
            choice_ids = list(range(start_id, end_id))
            if len(choice_ids) == 0:
                choice_ids = [end_id]
            pos_id = eop_index
            for i in range(self.config.cl_positive_k):
                pos_id -= 1
                if pos_id < start_id:
                    pos_id = random.choice(choice_ids)
                positive_eop_indices[i].append(pos_id)
            choice_ids = list(range(end_id + 1, eot_indices[-1] + 1))
            if len(choice_ids) == 0:
                choice_ids = list(range(bot_indices[0], bot_indices[1]))
            pos_id = end_id
            for i in range(self.config.cl_negative_k):
                pos_id += 1
                if pos_id >= total_eop_cnt:
                    pos_id = random.choice(choice_ids)
                negative_eop_indices[i].append(pos_id)
        loss = self.cl_loss_for_list(anchor_eop_features, anchor_eop_features, positive_eop_indices, negative_eop_indices)
        return loss

    def eot_level_list_cl_loss(self, anchor_eop_features, cl_segment_ids):
        """
        get cl_positive_k and cl_negative_k eop features for anchor eot features,
        then compute similarity of each pair consists of eot and one eop.
        eg. a1 a2 a3 b1 c1 c2
        first anchor eot features are [a3, b1, c2]
        if cl_positive_k == 1, then positive eop features are [a2, b1, c1]
        if cl_negative_k == 1, then negative eop features are [b1, c1, a1].
        we can implement the idea by cl_segment_ids [0 0 0 1 2 2]
        """
        loss = None
        total_topic_cnt = cl_segment_ids[-1] + 1
        total_eop_cnt = len(cl_segment_ids)
        accumulate_eop_cnts = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        bot_indices = [cl_segment_ids.index(i) for i in range(total_topic_cnt)]
        eot_indices = [(v - 1) for v in accumulate_eop_cnts[1:]] + [total_eop_cnt - 1]
        eot_features = anchor_eop_features[eot_indices]
        positive_eop_indices = [[] for _ in range(self.config.cl_positive_k)]
        for start_id, end_id in zip(bot_indices, eot_indices):
            choice_ids = list(range(start_id, end_id))
            if len(choice_ids) == 0:
                choice_ids = [end_id]
            pos_id = end_id
            for i in range(self.config.cl_positive_k):
                pos_id -= 1
                if pos_id < start_id:
                    pos_id = random.choice(choice_ids)
                positive_eop_indices[i].append(pos_id)
        negative_eop_indices = [[] for _ in range(self.config.cl_negative_k)]
        for end_id in eot_indices:
            choice_ids = list(range(end_id + 1, eot_indices[-1] + 1))
            if len(choice_ids) == 0:
                choice_ids = list(range(bot_indices[0], bot_indices[1]))
            pos_id = end_id
            for i in range(self.config.cl_negative_k):
                pos_id += 1
                if pos_id >= total_eop_cnt:
                    pos_id = random.choice(choice_ids)
                negative_eop_indices[i].append(pos_id)
        loss = self.cl_loss_for_list(anchor_eop_features, eot_features, positive_eop_indices, negative_eop_indices)
        return loss

    def forward(self, sequence_output, labels, extract_eop_segment_ids, eop_index_for_aggregate_batch_eop_features):
        loss = torch.tensor(0.0)
        bs, seq_length, hidden_size = sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[2]
        eop_level_output = self.multiple2one_pooling(sequence_output, extract_eop_segment_ids, pool_type='amax')
        tmp_eop_index = eop_index_for_aggregate_batch_eop_features + torch.arange(bs).unsqueeze(1).expand_as(eop_index_for_aggregate_batch_eop_features) * seq_length
        tmp_eop_index = tmp_eop_index.reshape(-1)
        eop_index_for_aggregate_batch_eop_features = eop_index_for_aggregate_batch_eop_features.reshape(-1)
        eop_index = tmp_eop_index[eop_index_for_aggregate_batch_eop_features != 0]
        anchor_eop_features = eop_level_output.reshape(bs * seq_length, -1)[eop_index]
        eop_labels = [l[l != -100] for l in labels]
        cl_segment_ids = []
        seg_id = 0
        for example_eop_labels in eop_labels:
            if len(example_eop_labels) == 0:
                continue
            for l in example_eop_labels:
                cl_segment_ids.append(seg_id)
                if l == 0:
                    seg_id += 1
            if example_eop_labels[-1] == 1:
                seg_id += 1
        if len(cl_segment_ids) > 2 and cl_segment_ids[-1] > 0:
            if self.config.cl_anchor_level == 'eop_matrix':
                loss = self.eop_level_matrix_cl_loss(anchor_eop_features, cl_segment_ids)
            elif self.config.cl_anchor_level == 'eot_list':
                loss = self.eot_level_list_cl_loss(anchor_eop_features, cl_segment_ids)
            elif self.config.cl_anchor_level == 'eop_list':
                loss = self.eop_level_list_cl_loss(anchor_eop_features, cl_segment_ids)
            else:
                raise ValueError('not supported cl_anchor_level %s ' % self.config.cl_anchor_level)
        return loss


class EopPairCosineSimilarity(nn.Module):

    def __init__(self, temp):
        super().__init__()
        self.cos_sim_fct = Similarity(temp=temp)

    def forward(self, sequence_output, labels):
        batch_eop_pair_cos_sim, batch_eop_labels = [], []
        max_eop_sent_cnt = 0
        for example_sequence_output, example_labels in zip(sequence_output, labels):
            eop_mask = example_labels != -100
            batch_eop_labels.append(example_labels[eop_mask])
            example_eop_sent_out = example_sequence_output[eop_mask]
            eop_sent_cnt = example_eop_sent_out.shape[0]
            max_eop_sent_cnt = max(max_eop_sent_cnt, eop_sent_cnt)
            sent_index = torch.arange(0, eop_sent_cnt)
            next_sent_index = (sent_index + 1) % eop_sent_cnt
            next_sent_out = example_eop_sent_out[next_sent_index]
            cos_sim = self.cos_sim_fct(example_eop_sent_out, next_sent_out)
            batch_eop_pair_cos_sim.append(cos_sim)
        for i, (cos_sim, eop_labels) in enumerate(zip(batch_eop_pair_cos_sim, batch_eop_labels)):
            batch_eop_pair_cos_sim[i] = torch.cat((cos_sim, torch.ones(max_eop_sent_cnt - cos_sim.shape[0]) * -100)).unsqueeze(0)
            batch_eop_labels[i] = torch.cat((eop_labels, torch.ones(max_eop_sent_cnt - eop_labels.shape[0], dtype=eop_labels.dtype) * -100)).unsqueeze(0)
        batch_eop_pair_cos_sim = torch.cat(batch_eop_pair_cos_sim)
        batch_eop_labels = torch.cat(batch_eop_labels)
        return batch_eop_pair_cos_sim, batch_eop_labels


class SentenceFeaturesExtractor(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sequence_output, sent_token_mask):
        sent_features = sequence_output[sent_token_mask != -100]
        sent_labels = [l[l != -100] for l in sent_token_mask]
        topic_segment_ids = []
        seg_id = 0
        for example_sent_labels in sent_labels:
            if len(example_sent_labels) == 0:
                continue
            for l in example_sent_labels:
                topic_segment_ids.append(seg_id)
                if l == 0:
                    seg_id += 1
            if example_sent_labels[-1] == 1:
                seg_id += 1
        return sent_features, torch.tensor(topic_segment_ids)


class TSSP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_tssp_labels)

    def forward(self, sent_token_mask, da_seq_output, da_sent_pair_orders):
        tssp_loss = torch.tensor(0.0)
        if self.config.tssp_loss_weight == 0:
            return loss
        sent_extractor = SentenceFeaturesExtractor()
        sent_features, _ = sent_extractor(sequence_output=da_seq_output, sent_token_mask=sent_token_mask)
        logits = self.classifier(sent_features)
        tssp_labels = da_sent_pair_orders[da_sent_pair_orders != -100]
        loss_fct = torch.nn.CrossEntropyLoss()
        tssp_loss = loss_fct(logits.reshape(-1, self.config.num_tssp_labels), tssp_labels.reshape(-1))
        return self.config.tssp_loss_weight * tssp_loss


class FocalLoss(nn.CrossEntropyLoss):
    """ Focal loss for classification tasks on imbalanced datasets """

    def __init__(self, gamma, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction='none', label_smoothing=label_smoothing)
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        loss = None
        cross_entropy = super().forward(input_, target)
        if torch.isnan(cross_entropy).any():
            None
            loss = torch.tensor(0, dtype=torch.float, requires_grad=True)
            return loss
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def get_loss_fct(gamma, weight_label_zero, device):
    weight = None
    if weight_label_zero != 0.5:
        weight = torch.tensor([weight_label_zero, 1 - weight_label_zero], dtype=torch.float32)
    if gamma != 0:
        loss_fct = FocalLoss(gamma=gamma, weight=weight)
    else:
        loss_fct = CrossEntropyLoss(weight=weight)
    return loss_fct


class LossCalculator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eop_pair_cos_sim = EopPairCosineSimilarity(temp=self.config.ts_score_predictor_cos_temp)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cssl = CSSL(config=config)
        self.tssp = TSSP(config=config)

    def multiple2one_pooling(self, src, index, dim=1, pool_type='amax'):
        return torch.zeros_like(src).scatter_reduce(dim, index[:, :, None].expand_as(src), src, reduce=pool_type, include_self=False)

    def forward(self, sequence_output, labels, extract_eop_segment_ids=None, eop_index_for_aggregate_batch_eop_features=None, sent_token_mask=None, sent_pair_orders=None, da_example_flag=False):
        loss = torch.tensor(0, dtype=torch.float, requires_grad=True)
        batch_eop_pair_cos_sim, batch_eop_labels = self.eop_pair_cos_sim(sequence_output, labels)
        if self.config.ts_score_predictor == 'lt':
            logits = self.classifier(sequence_output)
            loss_fct = get_loss_fct(gamma=self.config.focal_loss_gamma, weight_label_zero=self.config.weight_label_zero, device=sequence_output.device)
            ts_loss = loss_fct(logits.reshape(-1, self.config.num_labels), labels.reshape(-1))
        elif self.config.ts_score_predictor == 'cos':
            loss_fct = torch.nn.BCEWithLogitsLoss()
            ts_loss = loss_fct(batch_eop_pair_cos_sim.reshape(-1), batch_eop_labels.reshape(-1).float())
            logits = torch.sigmoid(batch_eop_pair_cos_sim)
        else:
            raise ValueError('not supported ts_score_predictor %s' % ts_score_predictor)
        loss += self.config.ts_loss_weight * ts_loss
        if da_example_flag is False and self.config.cl_loss_weight != 0:
            assert extract_eop_segment_ids is not None and eop_index_for_aggregate_batch_eop_features is not None
            cl_loss = self.cssl(sequence_output=sequence_output, labels=labels, extract_eop_segment_ids=extract_eop_segment_ids, eop_index_for_aggregate_batch_eop_features=eop_index_for_aggregate_batch_eop_features)
            loss += self.config.cl_loss_weight * cl_loss
        if da_example_flag is True and self.config.tssp_loss_weight != 0:
            tssp_loss = self.tssp(sent_token_mask=sent_token_mask, da_seq_output=sequence_output, da_sent_pair_orders=sent_pair_orders)
            loss += self.config.tssp_loss_weight * tssp_loss
        return loss, logits, batch_eop_pair_cos_sim


class EopFeaturesExtractor(nn.Module):

    def __init__(self):
        super().__init__()

    def multiple2one_pooling(self, src, index, dim=1, pool_type='amax'):
        return torch.zeros_like(src).scatter_reduce(dim, index[:, :, None].expand_as(src), src, reduce=pool_type, include_self=False)

    def forward(self, sequence_output, labels, extract_eop_segment_ids, eop_index_for_aggregate_batch_eop_features):
        loss = None
        bs, seq_length, hidden_size = sequence_output.shape[0], sequence_output.shape[1], sequence_output.shape[2]
        eop_level_output = self.multiple2one_pooling(sequence_output, extract_eop_segment_ids, pool_type='amax')
        tmp_eop_index = eop_index_for_aggregate_batch_eop_features + torch.arange(bs).unsqueeze(1).expand_as(eop_index_for_aggregate_batch_eop_features) * seq_length
        tmp_eop_index = tmp_eop_index.reshape(-1)
        eop_index_for_aggregate_batch_eop_features = eop_index_for_aggregate_batch_eop_features.reshape(-1)
        eop_index = tmp_eop_index[eop_index_for_aggregate_batch_eop_features != 0]
        eop_features = eop_level_output.reshape(bs * seq_length, -1)[eop_index]
        eop_labels = [l[l != -100] for l in labels]
        topic_segment_ids = []
        seg_id = 0
        for example_eop_labels in eop_labels:
            if len(example_eop_labels) == 0:
                continue
            for l in example_eop_labels:
                topic_segment_ids.append(seg_id)
                if l == 0:
                    seg_id += 1
            if example_eop_labels[-1] == 1:
                seg_id += 1
        return eop_features, topic_segment_ids


class Fp32GroupNorm(nn.GroupNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(input.float(), self.num_groups, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(input.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class TransposeLast(nn.Module):

    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class ConvFeatureExtractionModel(nn.Module):

    def __init__(self, conv_layers: 'List[Tuple[int, int, int]]', dropout: 'float'=0.0, mode: 'str'='default', conv_bias: 'bool'=False, conv_type: 'str'='default'):
        super().__init__()
        assert mode in {'default', 'layer_norm'}

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):

            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            assert (is_layer_norm and is_group_norm) == False, 'layer norm and group norm are exclusive'
            if is_layer_norm:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.Sequential(TransposeLast(), Fp32LayerNorm(dim, elementwise_affine=True), TransposeLast()), nn.GELU())
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), Fp32GroupNorm(dim, dim, affine=True), nn.GELU())
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
        self.conv_type = conv_type
        if self.conv_type == 'default':
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, 'invalid conv definition: ' + str(cl)
                dim, k, stride = cl
                self.conv_layers.append(block(in_d, dim, k, stride, is_layer_norm=mode == 'layer_norm', is_group_norm=mode == 'default' and i == 0, conv_bias=conv_bias))
                in_d = dim
        elif self.conv_type == 'conv2d':
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                dim, k, stride = cl
                self.conv_layers.append(torch.nn.Conv2d(in_d, dim, k, stride))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == 'custom':
            in_d = 1
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                dim, k, stride = cl
                self.conv_layers.append(torch.nn.Conv2d(in_d, dim, k, stride, padding=1))
                self.conv_layers.append(torch.nn.LayerNorm([dim, idim]))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(torch.nn.MaxPool2d(2, stride=2, ceil_mode=True))
                    idim = int(math.ceil(idim / 2))
        else:
            pass

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        if self.conv_type == 'custom':
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == 'conv2d':
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x


class GradMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SamePad(nn.Module):

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class Swish(nn.Module):
    """Swish function
    """

    def __init__(self):
        """Construct an MultiHeadedAttention object."""
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class GLU_Linear(nn.Module):

    def __init__(self, input_dim, output_dim, glu_type='sigmoid', bias_in_glu=True):
        super(GLU_Linear, self).__init__()
        self.glu_type = glu_type
        self.output_dim = output_dim
        if glu_type == 'sigmoid':
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == 'swish':
            self.glu_act = Swish()
        elif glu_type == 'relu':
            self.glu_act = torch.nn.ReLU()
        elif glu_type == 'gelu':
            self.glu_act = torch.nn.GELU()
        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        x = self.linear(x)
        if self.glu_type == 'bilinear':
            x = x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2]
        else:
            x = x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2])
        return x


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """
    if p <= 0:
        return module
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4
    if not is_conv:
        assert module.weight.size(1) % block_size == 0, 'Input features must be a multiple of block sizes'
    elif module.kernel_size == (1, 1):
        assert module.in_channels % block_size == 0, 'Input channels must be a multiple of block sizes'
    else:
        k = module.kernel_size[0] * module.kernel_size[1]
        assert k % block_size == 0, 'Kernel size must be a multiple of block size'

    def _forward_pre_hook(mod, input):
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
            mask = mask
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8, has_relative_attention_bias=False, num_buckets=32, max_distance=128, gru_rel_pos=False, rescale_init=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        k_bias = True
        if rescale_init:
            k_bias = False
        k_embed_dim = embed_dim
        q_embed_dim = embed_dim
        self.k_proj = quant_noise(nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        relative_postion_if_large = max_exact + torch.log(relative_positions.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(self, query, key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, need_weights: 'bool'=True, static_kv: 'bool'=False, attn_mask: 'Optional[Tensor]'=None, before_softmax: 'bool'=False, need_head_weights: 'bool'=False, position_bias: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        is_tpu = query.device.type == 'xla'
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]
        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)
        if not is_tpu and incremental_state is None and not static_kv and not torch.jit.is_scripting() and self.q_head_dim == self.head_dim:
            assert key is not None and value is not None
            assert attn_mask is None
            attn_mask_rel_pos = None
            if position_bias is not None:
                attn_mask_rel_pos = position_bias
                if self.gru_rel_pos:
                    query_layer = query.transpose(0, 1)
                    new_x_shape = query_layer.size()[:-1] + (self.num_heads, -1)
                    query_layer = query_layer.view(*new_x_shape)
                    query_layer = query_layer.permute(0, 2, 1, 3)
                    _B, _H, _L, __ = query_layer.size()
                    gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                    attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias
                attn_mask_rel_pos = attn_mask_rel_pos.view((-1, tgt_len, tgt_len))
            k_proj_bias = self.k_proj.bias
            if k_proj_bias is None:
                k_proj_bias = torch.zeros_like(self.q_proj.bias)
            x, attn = F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]), torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)), self.bias_k, self.bias_v, self.add_zero_attn, self.dropout_module.p, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, need_weights, attn_mask_rel_pos, use_separate_proj_weight=True, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight)
            return x, attn, position_bias
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and 'prev_key' in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if 'prev_value' in saved_state:
                _prev_value = saved_state['prev_value']
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: 'Optional[Tensor]' = None
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask, batch_size=bsz, src_len=k.size(1), static_kv=static_kv)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v, position_bias
        if position_bias is not None:
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                position_bias = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias
            position_bias = position_bias.view(attn_weights.size())
            attn_weights = attn_weights + position_bias
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: 'Optional[Tensor]' = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: 'Optional[Tensor]', prev_key_padding_mask: 'Optional[Tensor]', batch_size: 'int', src_len: 'int', static_kv: 'bool') ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]') ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: 'Dict[str, Optional[Tensor]]' = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', buffer: 'Dict[str, Optional[Tensor]]'):
        return self.set_incremental_state(incremental_state, 'attn_state', buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: 'int', src_len: 'int', bsz: 'int'):
        return attn_weights


def gelu(x: 'torch.Tensor') ->torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, '_a'):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def get_activation_fn(activation: 'str'):
    """Returns the activation function corresponding to `activation`"""
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return gelu
    elif activation == 'gelu_fast':
        warnings.warn('--activation-fn=gelu_fast has been renamed to gelu_accurate')
        return gelu_accurate
    elif activation == 'gelu_accurate':
        return gelu_accurate
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'linear':
        return lambda x: x
    elif activation == 'glu':
        return lambda x: x
    else:
        raise RuntimeError('--activation-fn {} not supported'.format(activation))


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, embedding_dim: 'float'=768, ffn_embedding_dim: 'float'=3072, num_attention_heads: 'float'=8, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.1, activation_fn: 'str'='relu', layer_norm_first: 'bool'=False, has_relative_attention_bias: 'bool'=False, num_buckets: 'int'=0, max_distance: 'int'=0, rescale_init: 'bool'=False, gru_rel_pos: 'bool'=False) ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(self.embedding_dim, num_attention_heads, dropout=attention_dropout, self_attention=True, has_relative_attention_bias=has_relative_attention_bias, num_buckets=num_buckets, max_distance=max_distance, rescale_init=rescale_init, gru_rel_pos=gru_rel_pos)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        if self.activation_name == 'glu':
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, 'swish')
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(self, x: 'torch.Tensor', self_attn_mask: 'torch.Tensor'=None, self_attn_padding_mask: 'torch.Tensor'=None, need_weights: 'bool'=False, pos_bias=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=False, attn_mask=self_attn_mask, position_bias=pos_bias)
            x = self.dropout1(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == 'glu':
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=need_weights, attn_mask=self_attn_mask, position_bias=pos_bias)
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            residual = x
            if self.activation_name == 'glu':
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        return x, attn, pos_bias


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TransformerEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.pos_conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=args.conv_pos, padding=args.conv_pos // 2, groups=args.conv_pos_groups)
        dropout = 0
        std = math.sqrt(4 * (1.0 - dropout) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name='weight', dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())
        if hasattr(args, 'relative_position_embedding'):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0
        self.layers = nn.ModuleList([TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=self.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_fn=args.activation_fn, layer_norm_first=args.layer_norm_first, has_relative_attention_bias=self.relative_position_embedding and i == 0, num_buckets=self.num_buckets, max_distance=self.max_distance, gru_rel_pos=args.gru_rel_pos) for i in range(args.encoder_layers)])
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x[padding_mask] = 0
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or dropout_probability > self.layerdrop:
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, self_attn_mask=streaming_mask, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break
        if r is not None:
            x = r
        x = x.transpose(0, 1)
        return x, layer_results


def compute_mask_indices(shape: 'Tuple[int, int]', padding_mask: 'Optional[torch.Tensor]', mask_prob: 'float', mask_length: 'int', mask_type: 'str'='static', mask_other: 'float'=0.0, min_masks: 'int'=0, no_overlap: 'bool'=False, min_space: 'int'=0) ->np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask
        if mask_type == 'static':
            lengths = np.full(num_mask, mask_length)
        elif mask_type == 'uniform':
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == 'normal':
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == 'poisson':
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception('unknown mask selection ' + mask_type)
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)
        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts
            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter((e - s if e - s >= length + min_space else 0 for s, e in parts), np.int)
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
            mask_idc = np.asarray([(mask_idc[j] + offset) for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))
    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask


class WavLM(nn.Module):

    def __init__(self, cfg: 'WavLMConfig') ->None:
        super().__init__()
        logger.info(f'WavLM Config: {cfg.__dict__}')
        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(conv_layers=feature_enc_layers, dropout=0.0, mode=cfg.extractor_mode, conv_bias=cfg.conv_bias)
        self.post_extract_proj = nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = cfg.feature_grad_mult
        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices((B, T), padding_mask, self.mask_prob, self.mask_length, self.mask_selection, self.mask_other, min_masks=2, no_overlap=self.no_mask_overlap, min_space=self.mask_min_space)
            mask_indices = torch.from_numpy(mask_indices)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None
        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices((B, C), None, self.mask_channel_prob, self.mask_channel_length, self.mask_channel_selection, self.mask_channel_other, no_overlap=self.no_mask_channel_overlap, min_space=self.mask_channel_min_space)
            mask_channel_indices = torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0
        return x, mask_indices

    def forward_padding_mask(self, features: 'torch.Tensor', padding_mask: 'torch.Tensor') ->torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_features(self, source: 'torch.Tensor', padding_mask: 'Optional[torch.Tensor]'=None, mask: 'bool'=False, ret_conv: 'bool'=False, output_layer: 'Optional[int]'=None, ret_layer_results: 'bool'=False):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features
        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=None if output_layer is None else output_layer - 1)
        res = {'x': x, 'padding_mask': padding_mask, 'features': features, 'layer_results': layer_results}
        feature = res['features'] if ret_conv else res['x']
        if ret_layer_results:
            feature = feature, res['layer_results']
        return feature, res['padding_mask']


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (COCOProjNet,
     lambda: ([], {'config': SimpleNamespace(imgdim=4, sentdim=4, projdim=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Fp32GroupNorm,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fp32LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GLU_Linear,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 0, 4])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (PairwiseRankingLoss,
     lambda: ([], {'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SamePad,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SentenceFeaturesExtractor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Similarity,
     lambda: ([], {'temp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TransposeLast,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

