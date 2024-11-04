import sys
_module = sys.modules[__name__]
del sys
Standard_Training = _module
auxiliaries = _module
auxiliaries_nofaiss = _module
datasets = _module
evaluate = _module
googlenet = _module
losses = _module
netlib = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import warnings


import numpy as np


import time


import random


import matplotlib


import matplotlib.pyplot as plt


import pandas as pd


import torch


import torch.nn as nn


import torch.multiprocessing


from torch.utils.data import Dataset


from torchvision import transforms


from sklearn import metrics


from sklearn import cluster


from sklearn.cluster import KMeans


from scipy.spatial.distance import squareform


from scipy.spatial.distance import pdist


from scipy.spatial.distance import cdist


import copy


from scipy.spatial import distance


from sklearn.preprocessing import normalize


from collections import namedtuple


import torch.nn.functional as F


from torch.utils import model_zoo


import itertools as it


from sklearn.decomposition import PCA


import torchvision.models as models


model_urls = {'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth'}


def googlenet(pretrained=False, **kwargs):
    """GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = GoogLeNet(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['googlenet']))
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model
    return GoogLeNet(**kwargs)


def rename_attr(model, attr, name):
    """
    Rename attribute in a class. Simply helper function.

    Args:
        model:  General Class for which attributes should be renamed.
        attr:   str, Name of target attribute.
        name:   str, New attribute name.
    """
    setattr(model, name, getattr(model, attr))
    delattr(model, attr)


class GoogLeNet(nn.Module):
    """
    Container for GoogLeNet s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, opt):
        """
        Args:
            opt: argparse.Namespace, contains all training-specific parameters.
        Returns:
            Nothing!
        """
        super(GoogLeNet, self).__init__()
        self.pars = opt
        self.model = googlenet.googlenet(num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else False)
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None
        rename_attr(self.model, 'fc', 'last_linear')
        self.layer_blocks = nn.ModuleList([self.model.inception3a, self.model.inception3b, self.model.maxpool3, self.model.inception4a, self.model.inception4b, self.model.inception4c, self.model.inception4d, self.model.inception4e, self.model.maxpool4, self.model.inception5a, self.model.inception5b, self.model.avgpool])
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

    def forward(self, x):
        x = self.model.conv3(self.model.conv2(self.model.maxpool1(self.model.conv1(x))))
        x = self.model.maxpool2(x)
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = x.view(x.size(0), -1)
        x = self.model.dropout(x)
        mod_x = self.model.last_linear(x)
        return mod_x if self.pars.loss == 'npair' else torch.nn.functional.normalize(mod_x, dim=-1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(BasicConv2d(in_channels, ch3x3red, kernel_size=1), BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(BasicConv2d(in_channels, ch5x5red, kernel_size=1), BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True), BasicConv2d(in_channels, pool_proj, kernel_size=1))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)
        return x


class TupleSampler:
    """
    Container for all sampling methods that can be used in conjunction with the respective loss functions.
    Based on batch-wise sampling, i.e. given a batch of training data, sample useful data tuples that are
    used to train the network more efficiently.
    """

    def __init__(self, method='random'):
        """
        Args:
            method: str, name of sampling method to use.
        Returns:
            Nothing!
        """
        self.method = method
        if method == 'semihard':
            self.give = self.semihardsampling
        if method == 'softhard':
            self.give = self.softhardsampling
        elif method == 'distance':
            self.give = self.distanceweightedsampling
        elif method == 'npair':
            self.give = self.npairsampling
        elif method == 'random':
            self.give = self.randomsampling

    def randomsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and randomly
        selects <len(batch)> triplets.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().numpy()
        unique_classes = np.unique(labels)
        indices = np.arange(len(batch))
        class_dict = {i: indices[labels == i] for i in unique_classes}
        sampled_triplets = [list(it.product([x], [x], [y for y in unique_classes if x != y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]
        sampled_triplets = [[x for x in list(it.product(*[class_dict[j] for j in i])) if x[0] != x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]
        sampled_triplets = random.sample(sampled_triplets, batch.shape[0])
        return sampled_triplets

    def semihardsampling(self, batch, labels, margin=0.2):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().numpy()
        bs = batch.size(0)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()
        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            neg = labels != l
            pos = labels == l
            anchors.append(i)
            pos[i] = False
            p = np.random.choice(np.where(pos)[0])
            positives.append(p)
            neg_mask = np.logical_and(neg, d > d[p])
            neg_mask = np.logical_and(neg_mask, d < margin + d[p])
            if neg_mask.sum() > 0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))
        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    def softhardsampling(self, batch, labels):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on semihard sampling introduced in 'https://arxiv.org/pdf/1503.03832.pdf'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().numpy()
        bs = batch.size(0)
        distances = self.pdist(batch.detach()).detach().cpu().numpy()
        positives, negatives = [], []
        anchors = []
        for i in range(bs):
            l, d = labels[i], distances[i]
            anchors.append(i)
            neg = labels != l
            pos = labels == l
            pos[i] = False
            neg_mask = np.logical_and(neg, d < d[np.where(pos)[0]].max())
            pos_mask = np.logical_and(pos, d > d[np.where(neg)[0]].min())
            if pos_mask.sum() > 0:
                positives.append(np.random.choice(np.where(pos_mask)[0]))
            else:
                positives.append(np.random.choice(np.where(pos)[0]))
            if neg_mask.sum() > 0:
                negatives.append(np.random.choice(np.where(neg_mask)[0]))
            else:
                negatives.append(np.random.choice(np.where(neg)[0]))
        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        return sampled_triplets

    def distanceweightedsampling(self, batch, labels, lower_cutoff=0.5, upper_cutoff=1.4):
        """
        This methods finds all available triplets in a batch given by the classes provided in labels, and select
        triplets based on distance sampling introduced in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
            lower_cutoff: float, lower cutoff value for negatives that are too close to anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
            upper_cutoff: float, upper cutoff value for positives that are too far away from the anchor embeddings. Set to literature value. They will be assigned a zero-sample probability.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        bs = batch.shape[0]
        distances = self.pdist(batch.detach()).clamp(min=lower_cutoff)
        positives, negatives = [], []
        labels_visited = []
        anchors = []
        for i in range(bs):
            neg = labels != labels[i]
            pos = labels == labels[i]
            q_d_inv = self.inverse_sphere_distances(batch, distances[i], labels, labels[i])
            pos[i] = 0
            positives.append(np.random.choice(np.where(pos)[0]))
            negatives.append(np.random.choice(bs, p=q_d_inv))
        sampled_triplets = [[a, p, n] for a, p, n in zip(list(range(bs)), positives, negatives)]
        return sampled_triplets

    def npairsampling(self, batch, labels):
        """
        This methods finds N-Pairs in a batch given by the classes provided in labels in the
        creation fashion proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'.

        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        label_set, count = np.unique(labels, return_counts=True)
        label_set = label_set[count >= 2]
        pos_pairs = np.array([np.random.choice(np.where(labels == x)[0], 2, replace=False) for x in label_set])
        neg_tuples = []
        for idx in range(len(pos_pairs)):
            neg_tuples.append(pos_pairs[np.delete(np.arange(len(pos_pairs)), idx), 1])
        neg_tuples = np.array(neg_tuples)
        sampled_npairs = [[a, p, *list(neg)] for (a, p), neg in zip(pos_pairs, neg_tuples)]
        return sampled_npairs

    def pdist(self, A):
        """
        Efficient function to compute the distance matrix for a matrix A.

        Args:
            A:   Matrix/Tensor for which the distance matrix is to be computed.
            eps: float, minimal distance/clampling value to ensure no zero values.
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        prod = torch.mm(A, A.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min=0)
        return res.clamp(min=0).sqrt()

    def inverse_sphere_distances(self, batch, dist, labels, anchor_label):
        """
        Function to utilise the distances of batch samples to compute their
        probability of occurence, and using the inverse to sample actual negatives to the resp. anchor.

        Args:
            batch:        torch.Tensor(), batch for which the sampling probabilities w.r.t to the anchor are computed. Used only to extract the shape.
            dist:         torch.Tensor(), computed distances between anchor to all batch samples.
            labels:       np.ndarray, labels for each sample for which distances were computed in dist.
            anchor_label: float, anchor label
        Returns:
            distance_matrix, clamped to ensure no zero values are passed.
        """
        bs, dim = len(dist), batch.shape[-1]
        log_q_d_inv = (2.0 - float(dim)) * torch.log(dist) - float(dim - 3) / 2 * torch.log(1.0 - 0.25 * dist.pow(2))
        log_q_d_inv[np.where(labels == anchor_label)[0]] = 0
        q_d_inv = torch.exp(log_q_d_inv - torch.max(log_q_d_inv))
        q_d_inv[np.where(labels == anchor_label)[0]] = 0
        q_d_inv = q_d_inv / q_d_inv.sum()
        return q_d_inv.detach().cpu().numpy()


class TripletLoss(torch.nn.Module):

    def __init__(self, margin=1, sampling_method='random'):
        """
        Basic Triplet Loss as proposed in 'FaceNet: A Unified Embedding for Face Recognition and Clustering'
        Args:
            margin:             float, Triplet Margin - Ensures that positives aren't placed arbitrarily close to the anchor.
                                Similarl, negatives should not be placed arbitrarily far away.
            sampling_method:    Method to use for sampling training triplets. Used for the TupleSampler-class.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.sampler = TupleSampler(method=sampling_method)

    def triplet_distance(self, anchor, positive, negative):
        """
        Compute triplet loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            triplet loss (torch.Tensor())
        """
        return torch.nn.functional.relu((anchor - positive).pow(2).sum() - (anchor - negative).pow(2).sum() + self.margin)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            triplet loss (torch.Tensor(), batch-averaged)
        """
        sampled_triplets = self.sampler.give(batch, labels)
        loss = torch.stack([self.triplet_distance(batch[triplet[0], :], batch[triplet[1], :], batch[triplet[2], :]) for triplet in sampled_triplets])
        return torch.mean(loss)


class NPairLoss(torch.nn.Module):

    def __init__(self, l2=0.02):
        """
        Basic N-Pair Loss as proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'

        Args:
            l2: float, weighting parameter for weight penality due to embeddings not being normalized.
        Returns:
            Nothing!
        """
        super(NPairLoss, self).__init__()
        self.sampler = TupleSampler(method='npair')
        self.l2 = l2

    def npair_distance(self, anchor, positive, negatives):
        """
        Compute basic N-Pair loss.

        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            n-pair loss (torch.Tensor())
        """
        return torch.log(1 + torch.sum(torch.exp(anchor.mm((negatives - positive).transpose(0, 1)))))

    def weightsum(self, anchor, positive):
        """
        Compute weight penalty.
        NOTE: Only need to penalize anchor and positive since the negatives are created based on these.

        Args:
            anchor, positive: torch.Tensor(), resp. embeddings for anchor and positive samples.
        Returns:
            torch.Tensor(), Weight penalty
        """
        return torch.sum(anchor ** 2 + positive ** 2)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            n-pair loss (torch.Tensor(), batch-averaged)
        """
        sampled_npairs = self.sampler.give(batch, labels)
        loss = torch.stack([self.npair_distance(batch[npair[0]:npair[0] + 1, :], batch[npair[1]:npair[1] + 1, :], batch[npair[2:], :]) for npair in sampled_npairs])
        loss = loss + self.l2 * torch.mean(torch.stack([self.weightsum(batch[npair[0], :], batch[npair[1], :]) for npair in sampled_npairs]))
        return torch.mean(loss)


class MarginLoss(torch.nn.Module):

    def __init__(self, margin=0.2, nu=0, beta=1.2, n_classes=100, beta_constant=False, sampling_method='distance'):
        """
        Basic Margin Loss as proposed in 'Sampling Matters in Deep Embedding Learning'.

        Args:
            margin:          float, fixed triplet margin (see also TripletLoss).
            nu:              float, regularisation weight for beta. Zero by default (in literature as well).
            beta:            float, initial value for trainable class margins. Set to default literature value.
            n_classes:       int, number of target class. Required because it dictates the number of trainable class margins.
            beta_constant:   bool, set to True if betas should not be trained.
            sampling_method: str, sampling method to use to generate training triplets.
        Returns:
            Nothing!
        """
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.n_classes = n_classes
        self.beta_constant = beta_constant
        self.beta_val = beta
        self.beta = beta if beta_constant else torch.nn.Parameter(torch.ones(n_classes) * beta)
        self.nu = nu
        self.sampling_method = sampling_method
        self.sampler = TupleSampler(method=sampling_method)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            margin loss (torch.Tensor(), batch-averaged)
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        sampled_triplets = self.sampler.give(batch, labels)
        d_ap, d_an = [], []
        for triplet in sampled_triplets:
            train_triplet = {'Anchor': batch[triplet[0], :], 'Positive': batch[triplet[1], :], 'Negative': batch[triplet[2]]}
            pos_dist = ((train_triplet['Anchor'] - train_triplet['Positive']).pow(2).sum() + 1e-08).pow(1 / 2)
            neg_dist = ((train_triplet['Anchor'] - train_triplet['Negative']).pow(2).sum() + 1e-08).pow(1 / 2)
            d_ap.append(pos_dist)
            d_an.append(neg_dist)
        d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)
        if self.beta_constant:
            beta = self.beta
        else:
            beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).type(torch.FloatTensor)
        pos_loss = torch.nn.functional.relu(d_ap - beta + self.margin)
        neg_loss = torch.nn.functional.relu(beta - d_an + self.margin)
        pair_count = torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).type(torch.FloatTensor)
        loss = torch.sum(pos_loss + neg_loss) if pair_count == 0.0 else torch.sum(pos_loss + neg_loss) / pair_count
        if self.nu:
            loss = loss + beta_regularisation_loss.type(torch.FloatTensor)
        return loss


class ProxyNCALoss(torch.nn.Module):

    def __init__(self, num_proxies, embedding_dim):
        """
        Basic ProxyNCA Loss as proposed in 'No Fuss Distance Metric Learning using Proxies'.

        Args:
            num_proxies:     int, number of proxies to use to estimate data groups. Usually set to number of classes.
            embedding_dim:   int, Required to generate initial proxies which are the same size as the actual data embeddings.
        Returns:
            Nothing!
        """
        super(ProxyNCALoss, self).__init__()
        self.num_proxies = num_proxies
        self.embedding_dim = embedding_dim
        self.PROXIES = torch.nn.Parameter(torch.randn(num_proxies, self.embedding_dim) / 8)
        self.all_classes = torch.arange(num_proxies)

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            proxynca loss (torch.Tensor(), batch-averaged)
        """
        batch = 3 * torch.nn.functional.normalize(batch, dim=1)
        PROXIES = 3 * torch.nn.functional.normalize(self.PROXIES, dim=1)
        pos_proxies = torch.stack([PROXIES[pos_label:pos_label + 1, :] for pos_label in labels])
        neg_proxies = torch.stack([torch.cat([self.all_classes[:class_label], self.all_classes[class_label + 1:]]) for class_label in labels])
        neg_proxies = torch.stack([PROXIES[neg_labels, :] for neg_labels in neg_proxies])
        dist_to_neg_proxies = torch.sum((batch[:, None, :] - neg_proxies).pow(2), dim=-1)
        dist_to_pos_proxies = torch.sum((batch[:, None, :] - pos_proxies).pow(2), dim=-1)
        negative_log_proxy_nca_loss = torch.mean(dist_to_pos_proxies[:, 0] + torch.logsumexp(-dist_to_neg_proxies, dim=1))
        return negative_log_proxy_nca_loss


class CEClassLoss(torch.nn.Module):

    def __init__(self, inp_dim, n_classes):
        """
        Basic Cross Entropy Loss for reference. Can be useful.
        Contains its own mapping network, so the actual network can remain untouched.

        Args:
            inp_dim:   int, embedding dimension of network.
            n_classes: int, number of target classes.
        Returns:
            Nothing!
        """
        super(CEClassLoss, self).__init__()
        self.mapper = torch.nn.Sequential(torch.nn.Linear(inp_dim, n_classes))
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
            labels:  np.ndarray [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        Returns:
            cross-entropy loss (torch.Tensor(), batch-averaged by default)
        """
        return self.ce_loss(self.mapper(batch), labels.type(torch.LongTensor))


class ResNet50(nn.Module):
    """
    Container for ResNet50 s.t. it can be used for metric learning.
    The Network has been broken down to allow for higher modularity, if one wishes
    to target specific layers/blocks directly.
    """

    def __init__(self, opt, list_style=False, no_norm=False):
        super(ResNet50, self).__init__()
        self.pars = opt
        if not opt.not_pretrained:
            None
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            None
        else:
            None
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        mod_x = self.model.last_linear(x)
        return mod_x if self.pars.loss == 'npair' else torch.nn.functional.normalize(mod_x, dim=-1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception,
     lambda: ([], {'in_channels': 4, 'ch1x1': 4, 'ch3x3red': 4, 'ch3x3': 4, 'ch5x5red': 4, 'ch5x5': 4, 'pool_proj': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionAux,
     lambda: ([], {'in_channels': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Confusezius_Deep_Metric_Learning_Baselines(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

