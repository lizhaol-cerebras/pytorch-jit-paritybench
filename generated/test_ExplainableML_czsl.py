
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


import numpy as np


import random


from torch.utils.data import Dataset


import torch


import torchvision.transforms as transforms


from itertools import product


import torch.nn as nn


import torch.nn.functional as F


import copy


from scipy.stats import hmean


import scipy.sparse as sp


import math


from torch.nn.init import xavier_uniform_


from torchvision import models


from torchvision.models.resnet import ResNet


from torchvision.models.resnet import BasicBlock


import torchvision.models as tmodels


import itertools


import collections


from torch.distributions.bernoulli import Bernoulli


from torch.autograd import Variable


from sklearn.svm import LinearSVC


from sklearn.model_selection import GridSearchCV


import scipy.io


from sklearn.calibration import CalibratedClassifierCV


import torchvision.models as models


from torch.utils.tensorboard import SummaryWriter


import torch.backends.cudnn as cudnn


import torch.optim as optim


class MLP(nn.Module):
    """
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    """

    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True, dropout=False, norm=False, layers=[]):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias=bias))
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace=True))
            if dropout:
                mod.append(nn.Dropout(p=0.5))
        mod.append(nn.Linear(incoming, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(inplace=True))
        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        return self.mod(x)


class Reshape(nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights, dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i, n in enumerate(names):
            for j, m in enumerate(names):
                dict_sim[n, m] = similarity[i, j].item()
        return dict_sim
    return pairing_names, similarity


DATA_FOLDER = 'ROOT_FOLDER'


def load_fasttext_embeddings(vocab):
    custom_map = {'Faux.Fur': 'fake fur', 'Faux.Leather': 'fake leather', 'Full.grain.leather': 'thick leather', 'Hair.Calf': 'hairy leather', 'Patent.Leather': 'shiny leather', 'Boots.Ankle': 'ankle boots', 'Boots.Knee.High': 'kneehigh boots', 'Boots.Mid-Calf': 'midcalf boots', 'Shoes.Boat.Shoes': 'boatshoes', 'Shoes.Clogs.and.Mules': 'clogs shoes', 'Shoes.Flats': 'flats shoes', 'Shoes.Heels': 'heels', 'Shoes.Loafers': 'loafers', 'Shoes.Oxfords': 'oxford shoes', 'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers', 'traffic_light': 'traficlight', 'trash_can': 'trashcan', 'dry-erase_board': 'dry_erase_board', 'black_and_white': 'black_white', 'eiffel_tower': 'tower'}
    vocab_lower = [v.lower() for v in vocab]
    vocab = []
    for current in vocab_lower:
        if current in custom_map:
            vocab.append(custom_map[current])
        else:
            vocab.append(current)
    ft = fasttext.load_model(DATA_FOLDER + '/fast/cc.en.300.bin')
    embeds = []
    for k in vocab:
        if '_' in k:
            ks = k.split('_')
            emb = np.stack([ft.get_word_vector(it) for it in ks]).mean(axis=0)
        else:
            emb = ft.get_word_vector(k)
        embeds.append(emb)
    embeds = torch.Tensor(np.stack(embeds))
    None
    return embeds


def load_glove_embeddings(vocab):
    """
    Inputs
        emb_file: Text file with word embedding pairs e.g. Glove, Processed in lower case.
        vocab: List of words
    Returns
        Embedding Matrix
    """
    vocab = [v.lower() for v in vocab]
    emb_file = DATA_FOLDER + '/glove/glove.6B.300d.txt'
    model = {}
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        model[line[0]] = wvec
    custom_map = {'faux.fur': 'fake_fur', 'faux.leather': 'fake_leather', 'full.grain.leather': 'thick_leather', 'hair.calf': 'hair_leather', 'patent.leather': 'shiny_leather', 'boots.ankle': 'ankle_boots', 'boots.knee.high': 'knee_high_boots', 'boots.mid-calf': 'midcalf_boots', 'shoes.boat.shoes': 'boat_shoes', 'shoes.clogs.and.mules': 'clogs_shoes', 'shoes.flats': 'flats_shoes', 'shoes.heels': 'heels', 'shoes.loafers': 'loafers', 'shoes.oxfords': 'oxford_shoes', 'shoes.sneakers.and.athletic.shoes': 'sneakers', 'traffic_light': 'traffic_light', 'trash_can': 'trashcan', 'dry-erase_board': 'dry_erase_board', 'black_and_white': 'black_white', 'eiffel_tower': 'tower', 'nubuck': 'grainy_leather'}
    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k:
            ks = k.split('_')
            emb = torch.stack([model[it] for it in ks]).mean(dim=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.stack(embeds)
    None
    return embeds


def load_word2vec_embeddings(vocab):
    model = models.KeyedVectors.load_word2vec_format(DATA_FOLDER + '/w2v/GoogleNews-vectors-negative300.bin', binary=True)
    custom_map = {'Faux.Fur': 'fake_fur', 'Faux.Leather': 'fake_leather', 'Full.grain.leather': 'thick_leather', 'Hair.Calf': 'hair_leather', 'Patent.Leather': 'shiny_leather', 'Boots.Ankle': 'ankle_boots', 'Boots.Knee.High': 'knee_high_boots', 'Boots.Mid-Calf': 'midcalf_boots', 'Shoes.Boat.Shoes': 'boat_shoes', 'Shoes.Clogs.and.Mules': 'clogs_shoes', 'Shoes.Flats': 'flats_shoes', 'Shoes.Heels': 'heels', 'Shoes.Loafers': 'loafers', 'Shoes.Oxfords': 'oxford_shoes', 'Shoes.Sneakers.and.Athletic.Shoes': 'sneakers', 'traffic_light': 'traffic_light', 'trash_can': 'trashcan', 'dry-erase_board': 'dry_erase_board', 'black_and_white': 'black_white', 'eiffel_tower': 'tower'}
    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k and k not in model:
            ks = k.split('_')
            emb = np.stack([model[it] for it in ks]).mean(axis=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.Tensor(np.stack(embeds))
    None
    return embeds


def load_word_embeddings(emb_type, vocab):
    if emb_type == 'glove':
        embeds = load_glove_embeddings(vocab)
    elif emb_type == 'fasttext':
        embeds = load_fasttext_embeddings(vocab)
    elif emb_type == 'word2vec':
        embeds = load_word2vec_embeddings(vocab)
    elif emb_type == 'ft+w2v':
        embeds1 = load_fasttext_embeddings(vocab)
        embeds2 = load_word2vec_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds2], dim=1)
        None
    elif emb_type == 'ft+gl':
        embeds1 = load_fasttext_embeddings(vocab)
        embeds2 = load_glove_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds2], dim=1)
        None
    elif emb_type == 'ft+ft':
        embeds1 = load_fasttext_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds1], dim=1)
        None
    elif emb_type == 'gl+w2v':
        embeds1 = load_glove_embeddings(vocab)
        embeds2 = load_word2vec_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds2], dim=1)
        None
    elif emb_type == 'ft+w2v+gl':
        embeds1 = load_fasttext_embeddings(vocab)
        embeds2 = load_word2vec_embeddings(vocab)
        embeds3 = load_glove_embeddings(vocab)
        embeds = torch.cat([embeds1, embeds2, embeds3], dim=1)
        None
    else:
        raise ValueError('Invalid embedding')
    return embeds


class CompCos(nn.Module):

    def __init__(self, dset, args):
        super(CompCos, self).__init__()
        self.args = args
        self.dset = dset

        def get_all_ids(relevant_pairs):
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs)
            objs = torch.LongTensor(objs)
            pairs = torch.LongTensor(pairs)
            return attrs, objs, pairs
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long(), torch.arange(len(self.dset.objs)).long()
        self.factor = 2
        self.scale = self.args.cosine_scale
        if dset.open_world:
            self.train_forward = self.train_forward_open
            self.known_pairs = dset.train_pairs
            seen_pair_set = set(self.known_pairs)
            mask = [(1 if pair in seen_pair_set else 0) for pair in dset.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.0
            self.activated = False
            self.attrs = dset.attrs
            self.objs = dset.objs
            self.possible_pairs = dset.pairs
            self.validation_pairs = dset.val_pairs
            self.feasibility_margin = (1 - self.seen_mask).float()
            self.epoch_max_margin = self.args.epoch_max_margin
            self.cosine_margin_factor = -args.margin
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for a, o in self.known_pairs:
                self.obj_by_attrs_train[a].append(o)
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for a, o in self.known_pairs:
                self.attrs_by_obj_train[o].append(a)
        else:
            self.train_forward = self.train_forward_closed
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
        try:
            self.args.fc_emb = self.args.fc_emb.split(',')
        except:
            self.args.fc_emb = [self.args.fc_emb]
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)
        self.image_embedder = MLP(dset.feat_dim, int(args.emb_dim), relu=args.relu, num_layers=args.nlayers, dropout=self.args.dropout, norm=self.args.norm, layers=layers)
        self.composition = args.composition
        input_dim = args.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False
        self.projection = nn.Linear(input_dim * 2, args.emb_dim)

    def freeze_representations(self):
        None
        for param in self.image_embedder.parameters():
            param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], 1)
        output = self.projection(inputs)
        output = F.normalize(output, dim=1)
        return output

    def compute_feasibility(self):
        obj_embeddings = self.obj_embedder(torch.arange(len(self.objs)).long())
        obj_embedding_sim = compute_cosine_similarity(self.objs, obj_embeddings, return_dict=True)
        attr_embeddings = self.attr_embedder(torch.arange(len(self.attrs)).long())
        attr_embedding_sim = compute_cosine_similarity(self.attrs, attr_embeddings, return_dict=True)
        feasibility_scores = self.seen_mask.clone().float()
        for a in self.attrs:
            for o in self.objs:
                if (a, o) not in self.known_pairs:
                    idx = self.dset.all_pair2idx[a, o]
                    score_obj = self.get_pair_scores_objs(a, o, obj_embedding_sim)
                    score_attr = self.get_pair_scores_attrs(a, o, attr_embedding_sim)
                    score = (score_obj + score_attr) / 2
                    feasibility_scores[idx] = score
        self.feasibility_scores = feasibility_scores
        return feasibility_scores * (1 - self.seen_mask.float())

    def get_pair_scores_objs(self, attr, obj, obj_embedding_sim):
        score = -1.0
        for o in self.objs:
            if o != obj and attr in self.attrs_by_obj_train[o]:
                temp_score = obj_embedding_sim[obj, o]
                if temp_score > score:
                    score = temp_score
        return score

    def get_pair_scores_attrs(self, attr, obj, attr_embedding_sim):
        score = -1.0
        for a in self.attrs:
            if a != attr and obj in self.obj_by_attrs_train[a]:
                temp_score = attr_embedding_sim[attr, a]
                if temp_score > score:
                    score = temp_score
        return score

    def update_feasibility(self, epoch):
        self.activated = True
        feasibility_scores = self.compute_feasibility()
        self.feasibility_margin = min(1.0, epoch / self.epoch_max_margin) * (self.cosine_margin_factor * feasibility_scores.float())

    def val_forward(self, x):
        img = x[0]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)
        score = torch.matmul(img_feats_normed, pair_embeds)
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]
        return None, scores

    def val_forward_with_threshold(self, x, th=0.0):
        img = x[0]
        img_feats = self.image_embedder(img)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)
        score = torch.matmul(img_feats_normed, pair_embeds)
        mask = (self.feasibility_scores >= th).float()
        score = score * mask + (1.0 - mask) * -1.0
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]
        return None, scores

    def train_forward_open(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img_feats = self.image_embedder(img)
        pair_embed = self.compose(self.train_attrs, self.train_objs).permute(1, 0)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_pred = torch.matmul(img_feats_normed, pair_embed)
        if self.activated:
            pair_pred += (1 - self.seen_mask) * self.feasibility_margin
            loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)
        else:
            pair_pred = pair_pred * self.seen_mask + (1 - self.seen_mask) * -10
            loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)
        return loss_cos.mean(), None

    def train_forward_closed(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        img_feats = self.image_embedder(img)
        pair_embed = self.compose(self.train_attrs, self.train_objs).permute(1, 0)
        img_feats_normed = F.normalize(img_feats, dim=1)
        pair_pred = torch.matmul(img_feats_normed, pair_embed)
        loss_cos = F.cross_entropy(self.scale * pair_pred, pairs)
        return loss_cos.mean(), None

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        self.layer = nn.Linear(in_channels, out_channels)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        outputs = torch.mm(adj, torch.mm(inputs, self.layer.weight.T)) + self.layer.bias
        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs


def normt_spm(mx, method='in'):
    if method == 'in':
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    if method == 'sym':
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GCN(nn.Module):

    def __init__(self, adj, in_channels, out_channels, hidden_layers):
        super().__init__()
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj
        self.train_adj = self.adj
        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False
        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)
            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)
            last_c = c
        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)
        self.layers = layers

    def forward(self, x):
        if self.training:
            for conv in self.layers:
                x = conv(x, self.train_adj)
        else:
            for conv in self.layers:
                x = conv(x, self.adj)
        return F.normalize(x)


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, dropout=False, relu=True, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        self.out_features = out_features
        self.residual = residual
        self.layer = nn.Linear(self.in_features, self.out_features, bias=False)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        if self.dropout is not None:
            input = self.dropout(input)
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        mm_term = torch.mm(support, self.layer.weight.T)
        output = theta * mm_term + (1 - theta) * r
        if self.residual:
            output = output + input
        if self.relu is not None:
            output = self.relu(output)
        return output


class GCNII(nn.Module):

    def __init__(self, adj, in_channels, out_channels, hidden_dim, hidden_layers, lamda, alpha, variant, dropout=True):
        super(GCNII, self).__init__()
        self.alpha = alpha
        self.lamda = lamda
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj
        i = 0
        layers = nn.ModuleList()
        self.fc_dim = nn.Linear(in_channels, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        for i, c in enumerate(range(hidden_layers)):
            conv = GraphConvolution(hidden_dim, hidden_dim, variant=variant, dropout=dropout)
            layers.append(conv)
        self.layers = layers
        self.fc_out = nn.Linear(hidden_dim, out_channels)

    def forward(self, x):
        _layers = []
        layer_inner = self.relu(self.fc_dim(self.dropout(x)))
        _layers.append(layer_inner)
        for i, con in enumerate(self.layers):
            layer_inner = con(layer_inner, self.adj, _layers[0], self.lamda, self.alpha, i + 1)
        layer_inner = self.fc_out(self.dropout(layer_inner))
        return layer_inner


class GraphFull(nn.Module):

    def __init__(self, dset, args):
        super(GraphFull, self).__init__()
        self.args = args
        self.dset = dset
        self.val_forward = self.val_forward_dotpr
        self.train_forward = self.train_forward_normal
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        self.pairs = dset.pairs
        if self.args.train_only:
            train_idx = []
            for current in dset.train_pairs:
                train_idx.append(dset.all_pair2idx[current] + self.num_attrs + self.num_objs)
            self.train_idx = torch.LongTensor(train_idx)
        self.args.fc_emb = self.args.fc_emb.split(',')
        layers = []
        for a in self.args.fc_emb:
            a = int(a)
            layers.append(a)
        if args.nlayers:
            self.image_embedder = MLP(dset.feat_dim, args.emb_dim, num_layers=args.nlayers, dropout=self.args.dropout, norm=self.args.norm, layers=layers, relu=True)
        all_words = list(self.dset.attrs) + list(self.dset.objs)
        self.displacement = len(all_words)
        self.obj_to_idx = {word: idx for idx, word in enumerate(self.dset.objs)}
        self.attr_to_idx = {word: idx for idx, word in enumerate(self.dset.attrs)}
        if args.graph_init is not None:
            path = args.graph_init
            graph = torch.load(path)
            embeddings = graph['embeddings']
            adj = graph['adj']
            self.embeddings = embeddings
        else:
            embeddings = self.init_embeddings(all_words)
            adj = self.adj_from_pairs()
            self.embeddings = embeddings
        hidden_layers = self.args.gr_emb
        if args.gcn_type == 'gcn':
            self.gcn = GCN(adj, self.embeddings.shape[1], args.emb_dim, hidden_layers)
        else:
            self.gcn = GCNII(adj, self.embeddings.shape[1], args.emb_dim, args.hidden_dim, args.gcn_nlayers, lamda=0.5, alpha=0.1, variant=False)

    def init_embeddings(self, all_words):

        def get_compositional_embeddings(embeddings, pairs):
            composition_embeds = []
            for attr, obj in pairs:
                attr_embed = embeddings[self.attr_to_idx[attr]]
                obj_embed = embeddings[self.obj_to_idx[obj] + self.num_attrs]
                composed_embed = (attr_embed + obj_embed) / 2
                composition_embeds.append(composed_embed)
            composition_embeds = torch.stack(composition_embeds)
            None
            return composition_embeds
        embeddings = load_word_embeddings(self.args.emb_init, all_words)
        composition_embeds = get_compositional_embeddings(embeddings, self.pairs)
        full_embeddings = torch.cat([embeddings, composition_embeds], dim=0)
        return full_embeddings

    def update_dict(self, wdict, row, col, data):
        wdict['row'].append(row)
        wdict['col'].append(col)
        wdict['data'].append(data)

    def adj_from_pairs(self):

        def edges_from_pairs(pairs):
            weight_dict = {'data': [], 'row': [], 'col': []}
            for i in range(self.displacement):
                self.update_dict(weight_dict, i, i, 1.0)
            for idx, (attr, obj) in enumerate(pairs):
                attr_idx, obj_idx = self.attr_to_idx[attr], self.obj_to_idx[obj] + self.num_attrs
                self.update_dict(weight_dict, attr_idx, obj_idx, 1.0)
                self.update_dict(weight_dict, obj_idx, attr_idx, 1.0)
                node_id = idx + self.displacement
                self.update_dict(weight_dict, node_id, node_id, 1.0)
                self.update_dict(weight_dict, node_id, attr_idx, 1.0)
                self.update_dict(weight_dict, node_id, obj_idx, 1.0)
                self.update_dict(weight_dict, attr_idx, node_id, 1.0)
                self.update_dict(weight_dict, obj_idx, node_id, 1.0)
            return weight_dict
        edges = edges_from_pairs(self.pairs)
        adj = sp.csr_matrix((edges['data'], (edges['row'], edges['col'])), shape=(len(self.pairs) + self.displacement, len(self.pairs) + self.displacement))
        return adj

    def train_forward_normal(self, x):
        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]
        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = img
        current_embeddings = self.gcn(self.embeddings)
        if self.args.train_only:
            pair_embed = current_embeddings[self.train_idx]
        else:
            pair_embed = current_embeddings[self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :]
        pair_embed = pair_embed.permute(1, 0)
        pair_pred = torch.matmul(img_feats, pair_embed)
        loss = F.cross_entropy(pair_pred, pairs)
        return loss, None

    def val_forward_dotpr(self, x):
        img = x[0]
        if self.args.nlayers:
            img_feats = self.image_embedder(img)
        else:
            img_feats = img
        current_embedddings = self.gcn(self.embeddings)
        pair_embeds = current_embedddings[self.num_attrs + self.num_objs:self.num_attrs + self.num_objs + self.num_pairs, :].permute(1, 0)
        score = torch.matmul(img_feats, pair_embeds)
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]
        return None, scores

    def val_forward_distance_fast(self, x):
        img = x[0]
        img_feats = self.image_embedder(img)
        current_embeddings = self.gcn(self.embeddings)
        pair_embeds = current_embeddings[self.num_attrs + self.num_objs:, :]
        batch_size, pairs, features = img_feats.shape[0], pair_embeds.shape[0], pair_embeds.shape[1]
        img_feats = img_feats[:, None, :].expand(-1, pairs, -1)
        pair_embeds = pair_embeds[None, :, :].expand(batch_size, -1, -1)
        diff = (img_feats - pair_embeds) ** 2
        score = diff.sum(2) * -1
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred


class ManifoldModel(nn.Module):

    def __init__(self, dset, args):
        super(ManifoldModel, self).__init__()
        self.args = args
        self.dset = dset

        def get_all_ids(relevant_pairs):
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs)
            objs = torch.LongTensor(objs)
            pairs = torch.LongTensor(pairs)
            return attrs, objs, pairs
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long(), torch.arange(len(self.dset.objs)).long()
        self.factor = 2
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs = self.val_subs, self.val_attrs, self.val_objs
        if args.lambda_aux > 0 or args.lambda_cls_attr > 0 or args.lambda_cls_obj > 0:
            None
            self.obj_clf = nn.Linear(args.emb_dim, len(dset.objs))
            self.attr_clf = nn.Linear(args.emb_dim, len(dset.attrs))

    def train_forward_bce(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4][:, 0], x[5][:, 0]
        img_feat = self.image_embedder(img)
        labels = np.random.binomial(1, 0.25, attrs.shape[0])
        labels = torch.from_numpy(labels).bool()
        sampled_attrs, sampled_objs = neg_attrs.clone(), neg_objs.clone()
        sampled_attrs[labels] = attrs[labels]
        sampled_objs[labels] = objs[labels]
        labels = labels.float()
        composed_clf = self.compose(attrs, objs)
        p = torch.sigmoid((img_feat * composed_clf).sum(1))
        loss = F.binary_cross_entropy(p, labels)
        return loss, None

    def train_forward_triplet(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4][:, 0], x[5][:, 0]
        img_feats = self.image_embedder(img)
        positive = self.compose(attrs, objs)
        negative = self.compose(neg_attrs, neg_objs)
        loss = F.triplet_margin_loss(img_feats, positive, negative, margin=self.args.margin)
        if self.args.lambda_aux > 0:
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            loss += self.args.lambda_aux * loss_aux
        return loss, None

    def val_forward_distance(self, x):
        img = x[0]
        batch_size = img.shape[0]
        img_feats = self.image_embedder(img)
        scores = {}
        pair_embeds = self.compose(self.val_attrs, self.val_objs)
        for itr, pair in enumerate(self.dset.pairs):
            pair_embed = pair_embeds[itr, None].expand(batch_size, pair_embeds.size(1))
            score = self.compare_metric(img_feats, pair_embed)
            scores[pair] = score
        return None, scores

    def val_forward_distance_fast(self, x):
        img = x[0]
        batch_size = img.shape[0]
        img_feats = self.image_embedder(img)
        pair_embeds = self.compose(self.val_attrs, self.val_objs)
        batch_size, pairs, features = img_feats.shape[0], pair_embeds.shape[0], pair_embeds.shape[1]
        img_feats = img_feats[:, None, :].expand(-1, pairs, -1)
        pair_embeds = pair_embeds[None, :, :].expand(batch_size, -1, -1)
        diff = (img_feats - pair_embeds) ** 2
        score = diff.sum(2) * -1
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]
        return None, scores

    def val_forward_direct(self, x):
        img = x[0]
        batch_size = img.shape[0]
        img_feats = self.image_embedder(img)
        pair_embeds = self.compose(self.val_attrs, self.val_objs).permute(1, 0)
        score = torch.matmul(img_feats, pair_embeds)
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:, self.dset.all_pair2idx[pair]]
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred


class RedWine(ManifoldModel):

    def __init__(self, dset, args):
        super(RedWine, self).__init__(dset, args)
        self.image_embedder = lambda img: img
        self.compare_metric = lambda img_feats, pair_embed: torch.sigmoid((img_feats * pair_embed).sum(1))
        self.train_forward = self.train_forward_bce
        self.val_forward = self.val_forward_distance_fast
        in_dim = dset.feat_dim if args.clf_init else args.emb_dim
        self.T = nn.Sequential(nn.Linear(2 * in_dim, 3 * in_dim), nn.LeakyReLU(0.1, True), nn.Linear(3 * in_dim, 3 * in_dim // 2), nn.LeakyReLU(0.1, True), nn.Linear(3 * in_dim // 2, dset.feat_dim))
        self.attr_embedder = nn.Embedding(len(dset.attrs), in_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), in_dim)
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)
        elif args.clf_init:
            for idx, attr in enumerate(dset.attrs):
                at_id = self.dset.attr2idx[attr]
                weight = torch.load('%s/svm/attr_%d' % (args.data_dir, at_id)).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            for idx, obj in enumerate(dset.objs):
                obj_id = self.dset.obj2idx[obj]
                weight = torch.load('%s/svm/obj_%d' % (args.data_dir, obj_id)).coef_.squeeze()
                self.obj_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
        else:
            None
            return
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def compose(self, attrs, objs):
        attr_wt = self.attr_embedder(attrs)
        obj_wt = self.obj_embedder(objs)
        inp_wts = torch.cat([attr_wt, obj_wt], 1)
        composed_clf = self.T(inp_wts)
        return composed_clf


class LabelEmbedPlus(ManifoldModel):

    def __init__(self, dset, args):
        super(LabelEmbedPlus, self).__init__(dset, args)
        if 'conv' in args.image_extractor:
            self.image_embedder = torch.nn.Sequential(torch.nn.Conv2d(dset.feat_dim, args.emb_dim, 7), torch.nn.ReLU(True), Reshape(-1, args.emb_dim))
        else:
            self.image_embedder = MLP(dset.feat_dim, args.emb_dim)
        self.compare_metric = lambda img_feats, pair_embed: -F.pairwise_distance(img_feats, pair_embed)
        self.train_forward = self.train_forward_triplet
        self.val_forward = self.val_forward_distance_fast
        input_dim = dset.feat_dim if args.clf_init else args.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)
        self.T = MLP(2 * input_dim, args.emb_dim, num_layers=args.nlayers)
        if args.emb_init:
            pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)
        elif args.clf_init:
            for idx, attr in enumerate(dset.attrs):
                at_id = dset.attrs.index(attr)
                weight = torch.load('%s/svm/attr_%d' % (args.data_dir, at_id)).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            for idx, obj in enumerate(dset.objs):
                obj_id = dset.objs.index(obj)
                weight = torch.load('%s/svm/obj_%d' % (args.data_dir, obj_id)).coef_.squeeze()
                self.obj_emb.weight[idx].data.copy_(torch.from_numpy(weight))
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def compose(self, attrs, objs):
        inputs = [self.attr_embedder(attrs), self.obj_embedder(objs)]
        inputs = torch.cat(inputs, 1)
        output = self.T(inputs)
        return output


class AttributeOperator(ManifoldModel):

    def __init__(self, dset, args):
        super(AttributeOperator, self).__init__(dset, args)
        self.image_embedder = MLP(dset.feat_dim, args.emb_dim)
        self.compare_metric = lambda img_feats, pair_embed: -F.pairwise_distance(img_feats, pair_embed)
        self.val_forward = self.val_forward_distance_fast
        self.attr_ops = nn.ParameterList([nn.Parameter(torch.eye(args.emb_dim)) for _ in range(len(self.dset.attrs))])
        self.obj_embedder = nn.Embedding(len(dset.objs), args.emb_dim)
        if args.emb_init:
            pretrained_weight = load_word_embeddings('glove', dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)
        self.inverse_cache = {}
        if args.lambda_ant > 0 and args.dataset == 'mitstates':
            antonym_list = open(DATA_FOLDER + '/data/antonyms.txt').read().strip().split('\n')
            antonym_list = [l.split() for l in antonym_list]
            antonym_list = [[self.dset.attrs.index(a1), self.dset.attrs.index(a2)] for a1, a2 in antonym_list]
            antonyms = {}
            antonyms.update({a1: a2 for a1, a2 in antonym_list})
            antonyms.update({a2: a1 for a1, a2 in antonym_list})
            self.antonyms, self.antonym_list = antonyms, antonym_list
        if args.static_inp:
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def apply_ops(self, ops, rep):
        out = torch.bmm(ops, rep.unsqueeze(2)).squeeze(2)
        out = F.relu(out)
        return out

    def compose(self, attrs, objs):
        obj_rep = self.obj_embedder(objs)
        attr_ops = torch.stack([self.attr_ops[attr.item()] for attr in attrs])
        embedded_reps = self.apply_ops(attr_ops, obj_rep)
        return embedded_reps

    def apply_inverse(self, img_rep, attrs):
        inverse_ops = []
        for i in range(img_rep.size(0)):
            attr = attrs[i]
            if attr not in self.inverse_cache:
                self.inverse_cache[attr] = self.attr_ops[attr].inverse()
            inverse_ops.append(self.inverse_cache[attr])
        inverse_ops = torch.stack(inverse_ops)
        obj_rep = self.apply_ops(inverse_ops, img_rep)
        return obj_rep

    def train_forward(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs, inv_attrs, comm_attrs = x[4][:, 0], x[5][:, 0], x[6], x[7]
        batch_size = img.size(0)
        loss = []
        anchor = self.image_embedder(img)
        obj_emb = self.obj_embedder(objs)
        pos_ops = torch.stack([self.attr_ops[attr.item()] for attr in attrs])
        positive = self.apply_ops(pos_ops, obj_emb)
        neg_obj_emb = self.obj_embedder(neg_objs)
        neg_ops = torch.stack([self.attr_ops[attr.item()] for attr in neg_attrs])
        negative = self.apply_ops(neg_ops, neg_obj_emb)
        loss_triplet = F.triplet_margin_loss(anchor, positive, negative, margin=self.args.margin)
        loss.append(loss_triplet)
        if self.args.lambda_aux > 0:
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            loss.append(self.args.lambda_aux * loss_aux)
        if self.args.lambda_inv > 0:
            obj_rep = self.apply_inverse(anchor, attrs)
            new_ops = torch.stack([self.attr_ops[attr.item()] for attr in inv_attrs])
            new_rep = self.apply_ops(new_ops, obj_rep)
            new_positive = self.apply_ops(new_ops, obj_emb)
            loss_inv = F.triplet_margin_loss(new_rep, new_positive, positive, margin=self.args.margin)
            loss.append(self.args.lambda_inv * loss_inv)
        if self.args.lambda_comm > 0:
            B = torch.stack([self.attr_ops[attr.item()] for attr in comm_attrs])
            BA = self.apply_ops(B, positive)
            AB = self.apply_ops(pos_ops, self.apply_ops(B, obj_emb))
            loss_comm = ((AB - BA) ** 2).sum(1).mean()
            loss.append(self.args.lambda_comm * loss_comm)
        if self.args.lambda_ant > 0:
            select_idx = [i for i in range(batch_size) if attrs[i].item() in self.antonyms]
            if len(select_idx) > 0:
                select_idx = torch.LongTensor(select_idx)
                attr_subset = attrs[select_idx]
                antonym_ops = torch.stack([self.attr_ops[self.antonyms[attr.item()]] for attr in attr_subset])
                Ao = anchor[select_idx]
                if self.args.lambda_inv > 0:
                    o = obj_rep[select_idx]
                else:
                    o = self.apply_inverse(Ao, attr_subset)
                BAo = self.apply_ops(antonym_ops, Ao)
                loss_cycle = ((BAo - o) ** 2).sum(1).mean()
                loss.append(self.args.lambda_ant * loss_cycle)
        loss = sum(loss)
        return loss, None

    def forward(self, x):
        loss, pred = super(AttributeOperator, self).forward(x)
        self.inverse_cache = {}
        return loss, pred


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class GatingSampler(nn.Module):
    """Docstring for GatingSampler. """

    def __init__(self, gater, stoch_sample=True, temperature=1.0):
        """TODO: to be defined1.

        """
        nn.Module.__init__(self)
        self.gater = gater
        self._stoch_sample = stoch_sample
        self._temperature = temperature

    def disable_stochastic_sampling(self):
        self._stoch_sample = False

    def enable_stochastic_sampling(self):
        self._stoch_sample = True

    def forward(self, tdesc=None, return_additional=False, gating_wt=None):
        if self.gater is None and return_additional:
            return None, None
        elif self.gater is None:
            return None
        if gating_wt is not None:
            return_wts = gating_wt
            gating_g = self.gater(tdesc, gating_wt=gating_wt)
        else:
            gating_g = self.gater(tdesc)
            return_wts = None
            if isinstance(gating_g, tuple):
                return_wts = gating_g[1]
                gating_g = gating_g[0]
        if not self._stoch_sample:
            sampled_g = gating_g
        else:
            raise NotImplementedError
        if return_additional:
            return sampled_g, return_wts
        return sampled_g


class GatedModularNet(nn.Module):
    """
        An interface for creating modular nets
    """

    def __init__(self, module_list, start_modules=None, end_modules=None, single_head=False, chain=False):
        """TODO: to be defined1.

        :module_list: TODO
        :g: TODO

        """
        nn.Module.__init__(self)
        self._module_list = nn.ModuleList([nn.ModuleList(m) for m in module_list])
        self.num_layers = len(self._module_list)
        if start_modules is not None:
            self._start_modules = nn.ModuleList(start_modules)
        else:
            self._start_modules = None
        if end_modules is not None:
            self._end_modules = nn.ModuleList(end_modules)
            self.num_layers += 1
        else:
            self._end_modules = None
        self.sampled_g = None
        self.single_head = single_head
        self._chain = chain

    def forward(self, x, sampled_g=None, t=None, return_feat=False):
        """TODO: Docstring for forward.

        :x: Input data
        :g: Gating tensor (#Task x )#num_layer x #num_mods x #num_mods
        :t: task ID
        :returns: TODO

        """
        if t is None:
            t_not_set = True
            t = torch.tensor([0] * x.shape[0], dtype=x.dtype).long()
        else:
            t_not_set = False
            t = t.squeeze()
        if self._start_modules is not None:
            prev_out = [mod(x) for mod in self._start_modules]
        else:
            prev_out = [x]
        if sampled_g is None:
            prev_out = sum(prev_out) / float(len(prev_out))
            for li in range(len(self._module_list)):
                prev_out = sum([mod(prev_out) for mod in self._module_list[li]]) / float(len(self._module_list[li]))
            features = prev_out
            if self._end_modules is not None:
                if t_not_set or self.single_head:
                    prev_out = self._end_modules[0](prev_out)
                else:
                    prev_out = torch.cat([self._end_modules[tid](prev_out[bi:bi + 1]) for bi, tid in enumerate(t)], 0)
            if return_feat:
                return prev_out, features
            return prev_out
        else:
            for li in range(len(self._module_list)):
                curr_out = []
                for j in range(len(self._module_list[li])):
                    gind = j if not self._chain else 0
                    module_in_wt = sampled_g[li + 1][gind]
                    module_in_wt = module_in_wt.transpose(0, 1)
                    add_dims = prev_out[0].dim() + 1 - module_in_wt.dim()
                    module_in_wt = module_in_wt.view(*module_in_wt.shape, *([1] * add_dims))
                    module_in_wt = module_in_wt.expand(len(prev_out), *prev_out[0].shape)
                    module_in = sum([(module_in_wt[i] * prev_out[i]) for i in range(len(prev_out))])
                    mod = self._module_list[li][j]
                    curr_out.append(mod(module_in))
                prev_out = curr_out
            if self._end_modules is not None:
                li = self.num_layers - 1
                if t_not_set or self.single_head:
                    module_in_wt = sampled_g[li + 1][0]
                    module_in_wt = module_in_wt.transpose(0, 1)
                    add_dims = prev_out[0].dim() + 1 - module_in_wt.dim()
                    module_in_wt = module_in_wt.view(*module_in_wt.shape, *([1] * add_dims))
                    module_in_wt = module_in_wt.expand(len(prev_out), *prev_out[0].shape)
                    module_in = sum([(module_in_wt[i] * prev_out[i]) for i in range(len(prev_out))])
                    features = module_in
                    prev_out = self._end_modules[0](module_in)
                else:
                    curr_out = []
                    for bi, tid in enumerate(t):
                        gind = tid if not self._chain else 0
                        module_in_wt = sampled_g[li + 1][gind]
                        module_in_wt = module_in_wt.transpose(0, 1)
                        add_dims = prev_out[0].dim() + 1 - module_in_wt.dim()
                        module_in_wt = module_in_wt.view(*module_in_wt.shape, *([1] * add_dims))
                        module_in_wt = module_in_wt.expand(len(prev_out), *prev_out[0].shape)
                        module_in = sum([(module_in_wt[i] * prev_out[i]) for i in range(len(prev_out))])
                        features = module_in
                        mod = self._end_modules[tid]
                        curr_out.append(mod(module_in[bi:bi + 1]))
                    prev_out = curr_out
                    prev_out = torch.cat(prev_out, 0)
            if return_feat:
                return prev_out, features
            return prev_out


class CompositionalModel(nn.Module):

    def __init__(self, dset, args):
        super(CompositionalModel, self).__init__()
        self.args = args
        self.dset = dset
        attrs, objs = zip(*self.dset.pairs)
        attrs = [dset.attr2idx[attr] for attr in attrs]
        objs = [dset.obj2idx[obj] for obj in objs]
        self.val_attrs = torch.LongTensor(attrs)
        self.val_objs = torch.LongTensor(objs)

    def train_forward_softmax(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4], x[5]
        inv_attrs, comm_attrs = x[6], x[7]
        sampled_attrs = torch.cat((attrs.unsqueeze(1), neg_attrs), 1)
        sampled_objs = torch.cat((objs.unsqueeze(1), neg_objs), 1)
        img_ind = torch.arange(sampled_objs.shape[0]).unsqueeze(1).repeat(1, sampled_attrs.shape[1])
        flat_sampled_attrs = sampled_attrs.view(-1)
        flat_sampled_objs = sampled_objs.view(-1)
        flat_img_ind = img_ind.view(-1)
        labels = torch.zeros_like(sampled_attrs[:, 0]).long()
        self.composed_g = self.compose(flat_sampled_attrs, flat_sampled_objs)
        cls_scores, feat = self.comp_network(img[flat_img_ind], self.composed_g, return_feat=True)
        pair_scores = cls_scores[:, :1]
        pair_scores = pair_scores.view(*sampled_attrs.shape)
        loss = 0
        loss_cls = F.cross_entropy(pair_scores, labels)
        loss += loss_cls
        loss_obj = torch.FloatTensor([0])
        loss_attr = torch.FloatTensor([0])
        loss_sparse = torch.FloatTensor([0])
        loss_unif = torch.FloatTensor([0])
        loss_aux = torch.FloatTensor([0])
        acc = (pair_scores.argmax(1) == labels).sum().float() / float(len(labels))
        all_losses = {}
        all_losses['total_loss'] = loss
        all_losses['main_loss'] = loss_cls
        all_losses['aux_loss'] = loss_aux
        all_losses['obj_loss'] = loss_obj
        all_losses['attr_loss'] = loss_attr
        all_losses['sparse_loss'] = loss_sparse
        all_losses['unif_loss'] = loss_unif
        return loss, all_losses, acc, (pair_scores, feat)

    def val_forward(self, x):
        img = x[0]
        batch_size = img.shape[0]
        pair_scores = torch.zeros(batch_size, len(self.val_attrs))
        pair_feats = torch.zeros(batch_size, len(self.val_attrs), self.args.emb_dim)
        pair_bs = len(self.val_attrs)
        for pi in range(math.ceil(len(self.val_attrs) / pair_bs)):
            self.compose_g = self.compose(self.val_attrs[pi * pair_bs:(pi + 1) * pair_bs], self.val_objs[pi * pair_bs:(pi + 1) * pair_bs])
            compose_g = self.compose_g
            expanded_im = img.unsqueeze(1).repeat(1, compose_g[0][0].shape[0], *tuple([1] * (img.dim() - 1))).view(-1, *img.shape[1:])
            expanded_compose_g = [[g.unsqueeze(0).repeat(batch_size, *tuple([1] * g.dim())).view(-1, *g.shape[1:]) for g in layer_g] for layer_g in compose_g]
            this_pair_scores, this_feat = self.comp_network(expanded_im, expanded_compose_g, return_feat=True)
            featnorm = torch.norm(this_feat, p=2, dim=-1)
            this_feat = this_feat.div(featnorm.unsqueeze(-1).expand_as(this_feat))
            this_pair_scores = this_pair_scores[:, :1].view(batch_size, -1)
            this_feat = this_feat.view(batch_size, -1, self.args.emb_dim)
            pair_scores[:, pi * pair_bs:pi * pair_bs + this_pair_scores.shape[1]] = this_pair_scores[:, :]
            pair_feats[:, pi * pair_bs:pi * pair_bs + this_pair_scores.shape[1], :] = this_feat[:]
        scores = {}
        feats = {}
        for i, (attr, obj) in enumerate(self.dset.pairs):
            scores[attr, obj] = pair_scores[:, i]
            feats[attr, obj] = pair_feats[:, i]
        return None, scores

    def forward(self, x, with_grad=False):
        if self.training:
            loss, loss_aux, acc, pred = self.train_forward(x)
        else:
            loss_aux = torch.Tensor([0])
            loss = torch.Tensor([0])
            if not with_grad:
                with torch.no_grad():
                    acc, pred = self.val_forward(x)
            else:
                acc, pred = self.val_forward(x)
        return loss, pred


class GeneralNormalizedNN(nn.Module):
    """Docstring for GatedCompositionalModel. """

    def __init__(self, num_layers, num_modules_per_layer, in_dim, inter_dim):
        """TODO: to be defined1. """
        nn.Module.__init__(self)
        self.start_modules = [nn.Sequential()]
        self.layer1 = [[nn.Sequential(nn.Linear(in_dim, inter_dim), nn.BatchNorm1d(inter_dim), nn.ReLU()) for _ in range(num_modules_per_layer)]]
        if num_layers > 1:
            self.layer2 = [[nn.Sequential(nn.BatchNorm1d(inter_dim), nn.Linear(inter_dim, inter_dim), nn.BatchNorm1d(inter_dim), nn.ReLU()) for _m in range(num_modules_per_layer)] for _l in range(num_layers - 1)]
        self.avgpool = nn.Sequential()
        self.fc = [nn.Sequential(nn.BatchNorm1d(inter_dim), nn.Linear(inter_dim, 1))]


class GeneralGatingNN(nn.Module):

    def __init__(self, num_mods, tdim, inter_tdim, randinit=False):
        """TODO: to be defined1.

        :num_mods: TODO
        :tdim: TODO

        """
        nn.Module.__init__(self)
        self._num_mods = num_mods
        self._tdim = tdim
        self._inter_tdim = inter_tdim
        task_outdim = self._inter_tdim
        self.task_linear1 = nn.Linear(self._tdim, task_outdim, bias=False)
        self.task_bn1 = nn.BatchNorm1d(task_outdim)
        self.task_linear2 = nn.Linear(task_outdim, task_outdim, bias=False)
        self.task_bn2 = nn.BatchNorm1d(task_outdim)
        self.joint_linear1 = nn.Linear(task_outdim, task_outdim, bias=False)
        self.joint_bn1 = nn.BatchNorm1d(task_outdim)
        num_out = [[1]] + [[self._num_mods[i - 1] for _ in range(self._num_mods[i])] for i in range(1, len(self._num_mods))]
        count = 0
        out_ind = []
        for i in range(len(num_out)):
            this_out_ind = []
            for j in range(len(num_out[i])):
                this_out_ind.append([count, count + num_out[i][j]])
                count += num_out[i][j]
            out_ind.append(this_out_ind)
        self.out_ind = out_ind
        self.out_count = count
        self.joint_linear2 = nn.Linear(task_outdim, count, bias=False)

        def apply_init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, nn.ModuleList):
                for subm in m:
                    if isinstance(subm, nn.ModuleList):
                        for subsubm in subm:
                            apply_init(subsubm)
                    else:
                        apply_init(subm)
            else:
                apply_init(m)
        if not randinit:
            self.joint_linear2.weight.data.zero_()

    def forward(self, tdesc=None):
        """TODO: Docstring for function.

        :arg1: TODO
        :returns: TODO

        """
        if tdesc is None:
            return None
        x = tdesc
        task_embeds1 = F.relu(self.task_bn1(self.task_linear1(x)))
        joint_embed = task_embeds1
        joint_embed = self.joint_linear2(joint_embed)
        joint_embed = [[joint_embed[:, self.out_ind[i][j][0]:self.out_ind[i][j][1]] for j in range(len(self.out_ind[i]))] for i in range(len(self.out_ind))]
        gating_wt = joint_embed
        prob_g = [[F.softmax(wt, -1) for wt in gating_wt[i]] for i in range(len(gating_wt))]
        return prob_g, gating_wt


def modularize_network(model, stoch_sample=False, use_full_model=False, tdim=200, inter_tdim=200, gater_type='general', single_head=True, num_classes=[2], num_lookup_gating=10):
    start_modules = model.start_modules
    end_modules = [nn.Sequential(model.avgpool, Flatten(), fci) for fci in model.fc]
    module_list = []
    li = 1
    while True:
        if hasattr(model, 'layer{}'.format(li)):
            module_list.extend(getattr(model, 'layer{}'.format(li)))
            li += 1
        else:
            break
    num_module_list = [len(start_modules)] + [len(layer) for layer in module_list] + [len(end_modules)]
    gated_model_func = GatedModularNet
    gated_net = gated_model_func(module_list, start_modules=start_modules, end_modules=end_modules, single_head=single_head)
    gater_func = GeneralGatingNN
    gater = gater_func(num_mods=num_module_list, tdim=tdim, inter_tdim=inter_tdim)
    fan_in = num_module_list
    if use_full_model:
        gater = None
    gating_sampler = GatingSampler(gater=gater, stoch_sample=stoch_sample)
    return gated_net, gating_sampler, num_module_list, fan_in


def modular_general(num_layers, num_modules_per_layer, feat_dim, inter_dim, stoch_sample=False, use_full_model=False, num_classes=[2], single_head=True, gater_type='general', tdim=300, inter_tdim=300, num_lookup_gating=10):
    model = GeneralNormalizedNN(num_layers, num_modules_per_layer, feat_dim, inter_dim)
    gated_net, gating_sampler, num_module_list, fan_in = modularize_network(model, stoch_sample, use_full_model=use_full_model, gater_type=gater_type, tdim=tdim, inter_tdim=inter_tdim, single_head=single_head, num_classes=num_classes, num_lookup_gating=num_lookup_gating)
    return gated_net, gating_sampler, num_module_list, fan_in


class GatedGeneralNN(CompositionalModel):
    """Docstring for GatedCompositionalModel. """

    def __init__(self, dset, args, num_layers=2, num_modules_per_layer=3, stoch_sample=False, use_full_model=False, num_classes=[2], gater_type='general'):
        """TODO: to be defined1.

        :dset: TODO
        :args: TODO

        """
        CompositionalModel.__init__(self, dset, args)
        self.train_forward = self.train_forward_softmax
        self.compose_type = 'nn'
        gating_in_dim = 128
        if args.emb_init:
            gating_in_dim = 300
        elif args.clf_init:
            gating_in_dim = 512
        if self.compose_type == 'nn':
            tdim = gating_in_dim * 2
            inter_tdim = self.args.embed_rank
            self.attr_embedder = nn.Embedding(len(dset.attrs) + 1, gating_in_dim, padding_idx=len(dset.attrs))
            self.obj_embedder = nn.Embedding(len(dset.objs) + 1, gating_in_dim, padding_idx=len(dset.objs))
            if args.emb_init:
                pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
                self.attr_embedder.weight[:-1, :].data.copy_(pretrained_weight)
                pretrained_weight = load_word_embeddings(args.emb_init, dset.objs)
                self.obj_embedder.weight.data[:-1, :].copy_(pretrained_weight)
            elif args.clf_init:
                for idx, attr in enumerate(dset.attrs):
                    at_id = self.dset.attr2idx[attr]
                    weight = torch.load('%s/svm/attr_%d' % (args.data_dir, at_id)).coef_.squeeze()
                    self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
                for idx, obj in enumerate(dset.objs):
                    obj_id = self.dset.obj2idx[obj]
                    weight = torch.load('%s/svm/obj_%d' % (args.data_dir, obj_id)).coef_.squeeze()
                    self.obj_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            else:
                n_attr = len(dset.attrs)
                gating_in_dim = 300
                tdim = gating_in_dim * 2 + n_attr
                self.attr_embedder = nn.Embedding(n_attr, n_attr)
                self.attr_embedder.weight.data.copy_(torch.from_numpy(np.eye(n_attr)))
                self.obj_embedder = nn.Embedding(len(dset.objs) + 1, gating_in_dim, padding_idx=len(dset.objs))
                pretrained_weight = load_word_embeddings('/home/ubuntu/workspace/czsl/data/glove/glove.6B.300d.txt', dset.objs)
                self.obj_embedder.weight.data[:-1, :].copy_(pretrained_weight)
        else:
            raise NotImplementedError
        self.comp_network, self.gating_network, self.nummods, _ = modular_general(num_layers=num_layers, num_modules_per_layer=num_modules_per_layer, feat_dim=dset.feat_dim, inter_dim=args.emb_dim, stoch_sample=stoch_sample, use_full_model=use_full_model, tdim=tdim, inter_tdim=inter_tdim, gater_type=gater_type)
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def compose(self, attrs, objs):
        obj_wt = self.obj_embedder(objs)
        if self.compose_type == 'nn':
            attr_wt = self.attr_embedder(attrs)
            inp_wts = torch.cat([attr_wt, obj_wt], 1)
        else:
            raise NotImplementedError
        composed_g, composed_g_wt = self.gating_network(inp_wts, return_additional=True)
        return composed_g


class Symnet(nn.Module):

    def __init__(self, dset, args):
        super(Symnet, self).__init__()
        self.dset = dset
        self.args = args
        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.image_embedder = MLP(dset.feat_dim, args.emb_dim, relu=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.nll_loss = nn.NLLLoss()
        self.softmax = nn.Softmax(dim=1)
        self.attr_embedder = nn.Embedding(len(dset.attrs), args.emb_dim)
        self.obj_classifier = MLP(args.emb_dim, len(dset.objs), num_layers=2, relu=False, dropout=True, layers=[512])
        self.attr_classifier = MLP(args.emb_dim, len(dset.attrs), num_layers=2, relu=False, dropout=True, layers=[512])
        self.CoN_fc_attention = MLP(args.emb_dim, args.emb_dim, num_layers=2, relu=False, dropout=True, layers=[512])
        self.CoN_emb = MLP(args.emb_dim + args.emb_dim, args.emb_dim, num_layers=2, relu=False, dropout=True, layers=[768])
        self.DecoN_fc_attention = MLP(args.emb_dim, args.emb_dim, num_layers=2, relu=False, dropout=True, layers=[512])
        self.DecoN_emb = MLP(args.emb_dim + args.emb_dim, args.emb_dim, num_layers=2, relu=False, dropout=True, layers=[768])
        pretrained_weight = load_word_embeddings(args.emb_init, dset.attrs)
        self.attr_embedder.weight.data.copy_(pretrained_weight)
        for param in self.attr_embedder.parameters():
            param.requires_grad = False

    def CoN(self, img_embedding, attr_embedding):
        attention = torch.sigmoid(self.CoN_fc_attention(attr_embedding))
        img_embedding = attention * img_embedding + img_embedding
        hidden = torch.cat([img_embedding, attr_embedding], dim=1)
        output = self.CoN_emb(hidden)
        return output

    def DeCoN(self, img_embedding, attr_embedding):
        attention = torch.sigmoid(self.DecoN_fc_attention(attr_embedding))
        img_embedding = attention * img_embedding + img_embedding
        hidden = torch.cat([img_embedding, attr_embedding], dim=1)
        output = self.DecoN_emb(hidden)
        return output

    def distance_metric(self, a, b):
        return torch.norm(a - b, dim=-1)

    def RMD_prob(self, feat_plus, feat_minus, repeat_img_feat):
        """return attribute classification probability with our RMD"""
        d_plus = self.distance_metric(feat_plus, repeat_img_feat).reshape(-1, self.num_attrs)
        d_minus = self.distance_metric(feat_minus, repeat_img_feat).reshape(-1, self.num_attrs)
        return d_minus - d_plus

    def train_forward(self, x):
        pos_image_feat, pos_attr_id, pos_obj_id = x[0], x[1], x[2]
        neg_attr_id = x[4][:, 0]
        batch_size = pos_image_feat.size(0)
        loss = []
        pos_attr_emb = self.attr_embedder(pos_attr_id)
        neg_attr_emb = self.attr_embedder(neg_attr_id)
        pos_img = self.image_embedder(pos_image_feat)
        pos_rA = self.DeCoN(pos_img, pos_attr_emb)
        pos_aA = self.CoN(pos_img, pos_attr_emb)
        pos_rB = self.DeCoN(pos_img, neg_attr_emb)
        pos_aB = self.CoN(pos_img, neg_attr_emb)
        attr_emb = torch.LongTensor(np.arange(self.num_attrs))
        attr_emb = self.attr_embedder(attr_emb)
        tile_attr_emb = attr_emb.repeat(batch_size, 1)
        if self.args.lambda_cls_attr > 0:
            score_pos_A = self.attr_classifier(pos_img)
            loss_cls_pos_a = self.ce_loss(score_pos_A, pos_attr_id)
            score_pos_rA_A = self.attr_classifier(pos_rA)
            total = sum(score_pos_rA_A)
            loss_cls_pos_rA_a = self.ce_loss(total - score_pos_rA_A, pos_attr_id)
            repeat_img_feat = torch.repeat_interleave(pos_img, self.num_attrs, 0)
            feat_plus = self.CoN(repeat_img_feat, tile_attr_emb)
            feat_minus = self.DeCoN(repeat_img_feat, tile_attr_emb)
            score_cls_rmd = self.RMD_prob(feat_plus, feat_minus, repeat_img_feat)
            loss_cls_rmd = self.ce_loss(score_cls_rmd, pos_attr_id)
            loss_cls_attr = self.args.lambda_cls_attr * sum([loss_cls_pos_a, loss_cls_pos_rA_a, loss_cls_rmd])
            loss.append(loss_cls_attr)
        if self.args.lambda_cls_obj > 0:
            score_pos_O = self.obj_classifier(pos_img)
            loss_cls_pos_o = self.ce_loss(score_pos_O, pos_obj_id)
            score_pos_rA_O = self.obj_classifier(pos_rA)
            loss_cls_pos_rA_o = self.ce_loss(score_pos_rA_O, pos_obj_id)
            score_pos_aB_O = self.obj_classifier(pos_aB)
            loss_cls_pos_aB_o = self.ce_loss(score_pos_aB_O, pos_obj_id)
            loss_cls_obj = self.args.lambda_cls_obj * sum([loss_cls_pos_o, loss_cls_pos_rA_o, loss_cls_pos_aB_o])
            loss.append(loss_cls_obj)
        if self.args.lambda_sym > 0:
            loss_sys_pos = self.mse_loss(pos_aA, pos_img)
            loss_sys_neg = self.mse_loss(pos_rB, pos_img)
            loss_sym = self.args.lambda_sym * (loss_sys_pos + loss_sys_neg)
            loss.append(loss_sym)
        if self.args.lambda_axiom > 0:
            loss_clo = loss_inv = loss_com = 0
            pos_aA_rA = self.DeCoN(pos_aA, pos_attr_emb)
            pos_rB_aB = self.CoN(pos_rB, neg_attr_emb)
            loss_clo = self.mse_loss(pos_aA_rA, pos_rA) + self.mse_loss(pos_rB_aB, pos_aB)
            pos_rA_aA = self.CoN(pos_rA, pos_attr_emb)
            pos_aB_rB = self.DeCoN(pos_aB, neg_attr_emb)
            loss_inv = self.mse_loss(pos_rA_aA, pos_img) + self.mse_loss(pos_aB_rB, pos_img)
            pos_aA_rB = self.DeCoN(pos_aA, neg_attr_emb)
            pos_rB_aA = self.DeCoN(pos_rB, pos_attr_emb)
            loss_com = self.mse_loss(pos_aA_rB, pos_rB_aA)
            loss_axiom = self.args.lambda_axiom * (loss_clo + loss_inv + loss_com)
            loss.append(loss_axiom)
        if self.args.lambda_trip > 0:
            pos_triplet = F.triplet_margin_loss(pos_img, pos_aA, pos_rA)
            neg_triplet = F.triplet_margin_loss(pos_img, pos_rB, pos_aB)
            loss_triplet = self.args.lambda_trip * (pos_triplet + neg_triplet)
            loss.append(loss_triplet)
        loss = sum(loss)
        return loss, None

    def val_forward(self, x):
        pos_image_feat, pos_attr_id, pos_obj_id = x[0], x[1], x[2]
        batch_size = pos_image_feat.shape[0]
        pos_img = self.image_embedder(pos_image_feat)
        repeat_img_feat = torch.repeat_interleave(pos_img, self.num_attrs, 0)
        attr_emb = torch.LongTensor(np.arange(self.num_attrs))
        attr_emb = self.attr_embedder(attr_emb)
        tile_attr_emb = attr_emb.repeat(batch_size, 1)
        feat_plus = self.CoN(repeat_img_feat, tile_attr_emb)
        feat_minus = self.DeCoN(repeat_img_feat, tile_attr_emb)
        score_cls_rmd = self.RMD_prob(feat_plus, feat_minus, repeat_img_feat)
        prob_A_rmd = F.softmax(score_cls_rmd, dim=1)
        score_obj = self.obj_classifier(pos_img)
        prob_O = F.softmax(score_obj, dim=1)
        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            score = prob_A_rmd[:, attr_id] * prob_O[:, obj_id]
            scores[attr, obj] = score
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred


class VisualProductNN(nn.Module):

    def __init__(self, dset, args):
        super(VisualProductNN, self).__init__()
        self.attr_clf = MLP(dset.feat_dim, len(dset.attrs), 2, relu=False)
        self.obj_clf = MLP(dset.feat_dim, len(dset.objs), 2, relu=False)
        self.dset = dset

    def train_forward(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        attr_pred = self.attr_clf(img)
        obj_pred = self.obj_clf(img)
        attr_loss = F.cross_entropy(attr_pred, attrs)
        obj_loss = F.cross_entropy(obj_pred, objs)
        loss = attr_loss + obj_loss
        return loss, None

    def val_forward(self, x):
        img = x[0]
        attr_pred = F.softmax(self.attr_clf(img), dim=1)
        obj_pred = F.softmax(self.obj_clf(img), dim=1)
        scores = {}
        for itr, (attr, obj) in enumerate(self.dset.pairs):
            attr_id, obj_id = self.dset.attr2idx[attr], self.dset.obj2idx[obj]
            score = attr_pred[:, attr_id] * obj_pred[:, obj_id]
            scores[attr, obj] = score
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GeneralGatingNN,
     lambda: ([], {'num_mods': [4, 4], 'tdim': 4, 'inter_tdim': 4}),
     lambda: ([], {})),
    (GraphConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (MLP,
     lambda: ([], {'inp_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

