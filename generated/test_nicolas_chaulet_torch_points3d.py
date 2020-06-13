import sys
_module = sys.modules[__name__]
del sys
conf = _module
eval = _module
find_neighbour_dist = _module
forward_scripts = _module
forward = _module
find_env = _module
find_runs = _module
descriptor_matcher = _module
fpfh = _module
misc = _module
save_feature = _module
test = _module
mock_models = _module
mockdatasets = _module
test_api = _module
test_basedataset = _module
test_basemodel = _module
test_batch = _module
test_bn_scheduler = _module
test_boxstuff = _module
test_confusionMatrix = _module
test_dataset_factory = _module
test_filter = _module
test_fps = _module
test_grid_sampling = _module
test_ind_tracker = _module
test_interpolateop = _module
test_kpconv = _module
test_losses = _module
test_lr_scheduler = _module
test_make_pair = _module
test_model_checkpoint = _module
test_models = _module
test_modules = _module
test_msdata = _module
test_msdatapair = _module
test_normal = _module
test_random_sphere = _module
test_registration_metrics = _module
test_registration_tracker = _module
test_resolver = _module
test_sampler = _module
test_samplers = _module
test_sampling_strategy = _module
test_segmentationtracker = _module
test_shapenetforward = _module
test_shapenetparttracker = _module
test_sphere_sampling = _module
test_to_sparse = _module
test_transform = _module
test_unwrapped_unet_base = _module
test_visualization = _module
utils = _module
torch_points3d = _module
applications = _module
kpconv = _module
modelfactory = _module
models = _module
pointnet2 = _module
rsconv = _module
core = _module
base_conv = _module
dense = _module
message_passing = _module
partial_dense = _module
common_modules = _module
base_modules = _module
dense_modules = _module
spatial_transform = _module
data_transform = _module
feature_augment = _module
features = _module
filters = _module
grid_transform = _module
inference_transforms = _module
sparse_transforms = _module
transforms = _module
initializer = _module
initializer = _module
losses = _module
dirichlet_loss = _module
huber_loss = _module
losses = _module
metric_losses = _module
regularizer = _module
regularizers = _module
schedulers = _module
bn_schedulers = _module
lr_schedulers = _module
spatial_ops = _module
interpolate = _module
neighbour_finder = _module
sampling = _module
datasets = _module
base_dataset = _module
batch = _module
classification = _module
modelnet = _module
dataset_factory = _module
multiscale_data = _module
object_detection = _module
box_data = _module
scannet = _module
base3dmatch = _module
base_siamese_dataset = _module
basetest = _module
detector = _module
fusion = _module
general3dmatch = _module
pair = _module
test3dmatch = _module
samplers = _module
segmentation = _module
shapenet = _module
s3dis = _module
metrics = _module
base_tracker = _module
box_detection = _module
ap = _module
classification_tracker = _module
colored_tqdm = _module
confusion_matrix = _module
meters = _module
model_checkpoint = _module
object_detection_tracker = _module
registration_metrics = _module
registration_tracker = _module
s3dis_tracker = _module
scannet_segmentation_tracker = _module
segmentation_helpers = _module
segmentation_tracker = _module
shapenet_part_tracker = _module
base_architectures = _module
backbone = _module
unet = _module
base_model = _module
model_factory = _module
model_interface = _module
votenet = _module
base = _module
kpconv = _module
minkowski = _module
pointnet = _module
pointnet2 = _module
base = _module
kpconv = _module
minkowski = _module
pointcnn = _module
pointnet = _module
pointnet2 = _module
randlanet = _module
rsconv = _module
KPConv = _module
blocks = _module
convolution_ops = _module
kernel_utils = _module
kernels = _module
plyutils = _module
MinkowskiEngine = _module
common = _module
modules = _module
networks = _module
res16unet = _module
resunet = _module
PointCNN = _module
modules = _module
PointNet = _module
modules = _module
RSConv = _module
dense = _module
message_passing = _module
RandLANet = _module
modules = _module
VoteNet = _module
loss_helper = _module
proposal_module = _module
votenet_results = _module
voting_module = _module
dense = _module
box_utils = _module
colors = _module
config = _module
debugging_vars = _module
enums = _module
geometry = _module
mock = _module
activation_resolver = _module
model_definition_resolver = _module
resolver_utils = _module
running_stats = _module
timer = _module
transform_utils = _module
visualization = _module
experiment_manager = _module
visualizer = _module
train = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch.nn import Sequential


from torch.nn import Linear as Lin


from torch.nn import ReLU


from torch.nn import LeakyReLU


from torch.nn import BatchNorm1d as BN


from torch.nn import Dropout


import torch


import logging


import numpy as np


from abc import abstractmethod


from typing import *


from torch import nn


from torch.nn.parameter import Parameter


import torch.nn as nn


from torch.nn import Linear


from typing import List


from typing import Optional


import itertools


import math


import re


import random


from torch.nn import functional as F


from functools import partial


import numpy


import scipy


import torch.nn.functional as F


from torch.nn import init


from typing import Any


from collections import OrderedDict


from torch.autograd import Variable


from typing import Dict


from torch.optim.optimizer import Optimizer


from torch.optim.lr_scheduler import _LRScheduler


from collections import defaultdict


from torch.nn import Sequential as Seq


from torch.nn import Identity


import collections


from enum import Enum


from math import ceil


from torch.nn import Sequential as S


from torch.nn import Linear as L


from torch.nn import ELU


from torch.nn import Conv1d


import torch.nn


class MockModel(torch.nn.Module):
    """ Mock mdoel that does literaly nothing but holds a state
    """

    def __init__(self):
        super().__init__()
        self.state = torch.nn.parameter.Parameter(torch.tensor([1.0]))
        self.optimizer = torch.nn.Module()
        self.schedulers = {}


class ConvMockDown(torch.nn.Module):

    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.test_precompute = test_precompute

    def forward(self, data, *args, **kwargs):
        data.append(self.kwargs['down_conv_nn'])
        if self.test_precompute:
            assert kwargs['precomputed'] is not None
        return data


class InnerMock(torch.nn.Module):

    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, data, *args, **kwargs):
        data.append('inner')
        return data


class ConvMockUp(torch.nn.Module):

    def __init__(self, test_precompute=False, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.test_precompute = test_precompute

    def forward(self, data, *args, **kwargs):
        data = data[0].copy()
        data.append(self.kwargs['up_conv_nn'])
        if self.test_precompute:
            assert kwargs['precomputed'] is not None
        return data


class Conv2D(Seq):

    def __init__(self, in_channels, out_channels, bias=True, bn=True,
        activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        self.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),
            stride=(1, 1), bias=bias))
        if bn:
            self.append(nn.BatchNorm2d(out_channels))
        if activation:
            self.append(activation)


class MLP2D(Seq):

    def __init__(self, channels, bias=False, bn=True, activation=nn.
        LeakyReLU(negative_slope=0.01)):
        super().__init__()
        for i in range(len(channels) - 1):
            self.append(Conv2D(channels[i], channels[i + 1], bn=bn, bias=
                bias, activation=activation))


class GlobalDenseBaseModule(torch.nn.Module):

    def __init__(self, nn, aggr='max', bn=True, activation=torch.nn.
        LeakyReLU(negative_slope=0.01), **kwargs):
        super(GlobalDenseBaseModule, self).__init__()
        self.nn = MLP2D(nn, bn=bn, activation=activation, bias=False)
        if aggr.lower() not in ['mean', 'max']:
            raise Exception('The aggregation provided is unrecognized {}'.
                format(aggr))
        self._aggr = aggr.lower()

    @property
    def nb_params(self):
        """[This property is used to return the number of trainable parameters for a given layer]
        It is useful for debugging and reproducibility.
        Returns:
            [type] -- [description]
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params

    def forward(self, data, **kwargs):
        x, pos = data.x, data.pos
        pos_flipped = pos.transpose(1, 2).contiguous()
        x = self.nn(torch.cat([x, pos_flipped], dim=1).unsqueeze(-1))
        if self._aggr == 'max':
            x = x.squeeze(-1).max(-1)[0]
        elif self._aggr == 'mean':
            x = x.squeeze(-1).mean(-1)
        else:
            raise NotImplementedError(
                'The following aggregation {} is not recognized'.format(
                self._aggr))
        pos = None
        x = x.unsqueeze(-1)
        return Data(x=x, pos=pos)

    def __repr__(self):
        return '{}: {} (aggr={}, {})'.format(self.__class__.__name__, self.
            nb_params, self._aggr, self.nn)


def copy_from_to(data, batch):
    for key in data.keys:
        if key not in batch.keys:
            setattr(batch, key, getattr(data, key, None))


class BaseResnetBlock(torch.nn.Module):

    def __init__(self, indim, outdim, convdim):
        """
            indim: size of x at the input
            outdim: desired size of x at the output
            convdim: size of x following convolution
        """
        torch.nn.Module.__init__(self)
        self.indim = indim
        self.outdim = outdim
        self.convdim = convdim
        self.features_downsample_nn = MLP([self.indim, self.outdim // 4])
        self.features_upsample_nn = MLP([self.convdim, self.outdim])
        self.shortcut_feature_resize_nn = MLP([self.indim, self.outdim])
        self.activation = ReLU()

    @property
    @abstractmethod
    def convs(self):
        pass

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x = data.x
        shortcut = x
        x = self.features_downsample_nn(x)
        data = self.convs(data)
        x = data.x
        idx = data.idx
        x = self.features_upsample_nn(x)
        if idx is not None:
            shortcut = shortcut[idx]
        shortcut = self.shortcut_feature_resize_nn(shortcut)
        x = shortcut + x
        batch_obj.x = x
        batch_obj.pos = data.pos
        batch_obj.batch = data.batch
        copy_from_to(data, batch_obj)
        return batch_obj


class GlobalPartialDenseBaseModule(torch.nn.Module):

    def __init__(self, nn, aggr='max', *args, **kwargs):
        super(GlobalPartialDenseBaseModule, self).__init__()
        self.nn = MLP(nn)
        self.pool = global_max_pool if aggr == 'max' else global_mean_pool

    def forward(self, data, **kwargs):
        batch_obj = Batch()
        x, pos, batch = data.x, data.pos, data.batch
        x = self.nn(torch.cat([x, pos], dim=1))
        x = self.pool(x, batch)
        batch_obj.x = x
        batch_obj.pos = pos.new_zeros((x.size(0), 3))
        batch_obj.batch = torch.arange(x.size(0), device=x.device)
        copy_from_to(data, batch_obj)
        return batch_obj


class BaseModule(nn.Module):
    """ Base module class with some basic additions to the pytorch Module class
    """

    @property
    def nb_params(self):
        """This property is used to return the number of trainable parameters for a given layer
        It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params


class Seq(nn.Sequential):

    def __init__(self):
        super().__init__()
        self._num_modules = 0

    def append(self, module):
        self.add_module(str(self._num_modules), module)
        self._num_modules += 1


class BaseLinearTransformSTNkD(torch.nn.Module):
    """STN which learns a k-dimensional linear transformation

    Arguments:
        nn (torch.nn.Module) -- module which takes feat_x as input and regresses it to a global feature used to calculate the transform
        nn_feat_size -- the size of the global feature
        k -- the size of trans_x
        batch_size -- the number of examples per batch
    """

    def __init__(self, nn, nn_feat_size, k=3, batch_size=1):
        super().__init__()
        self.nn = nn
        self.k = k
        self.batch_size = batch_size
        self.fc_layer = Linear(nn_feat_size, k * k)
        torch.nn.init.constant_(self.fc_layer.weight, 0)
        torch.nn.init.constant_(self.fc_layer.bias, 0)
        self.identity = torch.eye(k).view(1, k * k).repeat(batch_size, 1)

    def forward(self, feat_x, trans_x, batch):
        """
            Learns and applies a linear transformation to trans_x based on feat_x.
            feat_x and trans_x may be the same or different.
        """
        global_feature = self.nn(feat_x, batch)
        trans = self.fc_layer(global_feature)
        trans = trans + self.identity.to(feat_x.device)
        trans = trans.view(-1, self.k, self.k)
        self.trans = trans
        batch_x = trans_x.view(trans_x.shape[0], 1, trans_x.shape[1])
        x_transformed = torch.bmm(batch_x, trans[batch])
        return x_transformed.view(len(trans_x), trans_x.shape[1])

    def get_orthogonal_regularization_loss(self):
        loss = torch.mean(torch.norm(torch.bmm(self.trans, self.trans.
            transpose(2, 1)) - self.identity.to(self.trans.device).view(-1,
            self.k, self.k), dim=(1, 2)))
        return loss


def _variance_estimator_sparse(r, pos, f, batch_idx):
    with torch.no_grad():
        assign_index = radius(pos, pos, r, batch_x=batch_idx, batch_y=batch_idx
            )
        y_idx, x_idx = assign_index
        grad_f = (f[x_idx] - f[y_idx]) ** 2
    y = scatter_add(grad_f, y_idx, dim=0, dim_size=pos.size(0))
    return y


def _dirichlet_sparse(r, pos, f, batch_idx, aggr):
    variances = _variance_estimator_sparse(r, pos, f, batch_idx)
    return 1 / 2.0 * aggr(variances)


_MAX_NEIGHBOURS = 32


def _variance_estimator_dense(r, pos, f):
    nei_idx = tp.ball_query(r, _MAX_NEIGHBOURS, pos, pos, sort=True)[0
        ].reshape(pos.shape[0], -1).long()
    f_neighboors = f.gather(1, nei_idx).reshape(f.shape[0], f.shape[1], -1)
    gradient = (f.unsqueeze(-1).repeat(1, 1, f_neighboors.shape[-1]) -
        f_neighboors) ** 2
    return gradient.sum(-1)


def _dirichlet_dense(r, pos, f, aggr):
    variances = _variance_estimator_dense(r, pos, f)
    return 1 / 2.0 * aggr(variances)


def dirichlet_loss(r, pos, f, batch_idx=None, aggr=torch.mean):
    """ Computes the Dirichlet loss (or L2 norm of the gradient) of f
    Arguments:
        r -- Radius for the beighbour search
        pos -- [N,3] (or [B,N,3] for dense format)  location of each point
        f -- [N] (or [B,N] for dense format)  Value of a function at each points
        batch_idx -- [N] Batch id of each point (Only for sparse format)
        aggr -- aggregation function for the final loss value
    """
    if batch_idx is None:
        assert f.dim() == 2 and pos.dim() == 3
        return _dirichlet_dense(r, pos, f, aggr)
    else:
        assert f.dim() == 1 and pos.dim() == 2
        return _dirichlet_sparse(r, pos, f, batch_idx, aggr)


class DirichletLoss(torch.nn.Module):
    """ L2 norm of the gradient estimated as the average change of a field value f
    accross neighbouring points within a radius r
    """

    def __init__(self, r, aggr=torch.mean):
        super().__init__()
        self._r = r
        self._aggr = aggr

    def forward(self, pos, f, batch_idx=None):
        """ Computes the Dirichlet loss (or L2 norm of the gradient) of f
        Arguments:
            pos -- [N,3] (or [B,N,3] for dense format)  location of each point
            f -- [N] (or [B,N] for dense format)  Value of a function at each points
            batch_idx -- [N] Batch id of each point (Only for sparse format)
        """
        return dirichlet_loss(self._r, pos, f, batch_idx=batch_idx, aggr=
            self._aggr)


def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss


class HuberLoss(torch.nn.Module):

    def __init__(self, delta=0.1):
        super().__init__()
        self._delta = delta

    def forward(self, error):
        return huber_loss(error, self._delta)


class LossAnnealer(torch.nn.modules.loss._Loss):
    """
    This class will be used to perform annealing between two losses
    """

    def __init__(self, args):
        super(LossAnnealer, self).__init__()
        self._coeff = 0.5
        self.normalized_loss = True

    def forward(self, loss_1, loss_2, **kwargs):
        annealing_alpha = kwargs.get('annealing_alpha', None)
        if annealing_alpha is None:
            return self._coeff * loss_1 + (1 - self._coeff) * loss_2
        else:
            return (1 - annealing_alpha) * loss_1 + annealing_alpha * loss_2


class FocalLoss(torch.nn.modules.loss._Loss):

    def __init__(self, gamma: float=2, alphas: Any=None, size_average: bool
        =True, normalized: bool=True):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._alphas = alphas
        self.size_average = size_average
        self.normalized = normalized

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = torch.gather(logpt, -1, target.unsqueeze(-1))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self._alphas is not None:
            at = self._alphas.gather(0, target)
            logpt = logpt * Variable(at)
        if self.normalized:
            sum_ = 1 / torch.sum((1 - pt) ** self._gamma)
        else:
            sum_ = 1
        loss = -1 * sum_ * (1 - pt) ** self._gamma * logpt
        return loss.sum()


class WrapperKLDivLoss(torch.nn.modules.loss._Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WrapperKLDivLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, label_vec=None, segm_size=None):
        label_vec = Variable(label_vec).float() / segm_size.unsqueeze(-1
            ).float()
        input = F.log_softmax(input, dim=-1)
        loss = torch.nn.modules.loss.KLDivLoss()(input, label_vec)
        return loss


def _hash(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)
    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, (d)] * M ** d
        else:
            hash_vec += arr[d] * M ** d
    return hash_vec


def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-07)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError('Not implemented')


class ContrastiveHardestNegativeLoss(nn.Module):
    """
    Compute contrastive loss between positive pairs and mine negative pairs which are not in the intersection of the two point clouds (taken from https://github.com/chrischoy/FCGF)
    Let :math:`(f_i, f^{+}_i)_{i=1 \\dots N}` set of positive_pairs and :math:`(f_i, f^{-}_i)_{i=1 \\dots M}` a set of negative pairs
    The loss is computed as:
    .. math::
        L = \\frac{1}{N^2} \\sum_{i=1}^N \\sum_{j=1}^N [d^{+}_{ij} - \\lambda_+]_+ + \\frac{1}{M} \\sum_{i=1}^M [\\lambda_{-} - d^{-}_i]_+

    where:
    .. math::
        d^{+}_{ij} = ||f_{i} - f^{+}_{j}||

    and
    .. math::
        d^{-}_{i} = \\min_{j}(||f_{i} - f^{-}_{j}||)

    In this loss, we only mine the negatives
    Parameters
    ----------

    pos_thresh:
        positive threshold of the positive distance
    neg_thresh:
        negative threshold of the negative distance
    num_pos:
        number of positive pairs
    num_hn_samples:
        number of negative point we mine.
    """

    def __init__(self, pos_thresh, neg_thresh, num_pos=5192, num_hn_samples
        =2048):
        nn.Module.__init__(self)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.num_pos = num_pos
        self.num_hn_samples = num_hn_samples

    def contrastive_hardest_negative_loss(self, F0, F1, positive_pairs,
        thresh=None):
        """
        Generate negative pairs
        """
        N0, N1 = len(F0), len(F1)
        N_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, self.num_hn_samples), replace=False
            )
        sel1 = np.random.choice(N1, min(N1, self.num_hn_samples), replace=False
            )
        if N_pos_pairs > self.num_pos:
            pos_sel = np.random.choice(N_pos_pairs, self.num_pos, replace=False
                )
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs
        subF0, subF1 = F0[sel0], F1[sel1]
        pos_ind0 = sample_pos_pairs[:, (0)].long()
        pos_ind1 = sample_pos_pairs[:, (1)].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
        D01 = pdist(posF0, subF1, dist_type='L2')
        D10 = pdist(posF1, subF0, dist_type='L2')
        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)
        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)
        pos_keys = _hash(positive_pairs, hash_seed)
        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)
        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys,
            assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys,
            assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)
        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def forward(self, F0, F1, matches, xyz0=None, xyz1=None):
        pos_loss, neg_loss = self.contrastive_hardest_negative_loss(F0, F1,
            matches.detach().cpu())
        return pos_loss + neg_loss


class BatchHardContrastiveLoss(nn.Module):
    """
        apply contrastive loss but mine the negative sample in the batch.
    apply a mask if the distance between negative pair is too close.
    Parameters
    ----------
    pos_thresh:
        positive threshold of the positive distance
    neg_thresh:
        negative threshold of the negative distance
    min_dist:
        minimum distance to be in the negative sample
    """

    def __init__(self, pos_thresh, neg_thresh, min_dist=0.15):
        nn.Module.__init__(self)
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.min_dist = min_dist

    def forward(self, F0, F1, positive_pairs, xyz0=None, xyz1=None):
        posF0 = F0[positive_pairs[:, (0)]]
        posF1 = F1[positive_pairs[:, (1)]]
        subxyz0 = xyz0[positive_pairs[:, (0)]]
        false_negative = pdist(subxyz0, subxyz0, dist_type='L2'
            ) > self.min_dist
        furthest_pos, _ = (posF0 - posF1).pow(2).max(1)
        neg_loss = F.relu(self.neg_thresh - (posF0[0] - posF1[
            false_negative[0]]).pow(2).sum(1).min()).pow(2) / len(posF0)
        for i in range(1, len(posF0)):
            neg_loss += F.relu(self.neg_thresh - (posF0[i] - posF1[
                false_negative[i]]).pow(2).sum(1).min()).pow(2) / len(posF0)
        pos_loss = F.relu(furthest_pos - self.pos_thresh).pow(2)
        return pos_loss.mean() + neg_loss.mean()


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    """

    def get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    def __init__(self, args_up=None, args_down=None, args_innermost=None,
        modules_lib=None, submodule=None, outermost=False, innermost=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            args_up -- arguments for up convs
            args_down -- arguments for down convs
            args_innermost -- arguments for innermost
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if innermost:
            assert outermost == False
            module_name = self.get_from_kwargs(args_innermost, 'module_name')
            inner_module_cls = getattr(modules_lib, module_name)
            self.inner = inner_module_cls(**args_innermost)
            upconv_cls = self.get_from_kwargs(args_up, 'up_conv_cls')
            self.up = upconv_cls(**args_up)
        else:
            downconv_cls = self.get_from_kwargs(args_down, 'down_conv_cls')
            upconv_cls = self.get_from_kwargs(args_up, 'up_conv_cls')
            downconv = downconv_cls(**args_down)
            upconv = upconv_cls(**args_up)
            self.down = downconv
            self.submodule = submodule
            self.up = upconv

    def forward(self, data, **kwargs):
        if self.innermost:
            data_out = self.inner(data, **kwargs)
            data = data_out, data
            return self.up(data, **kwargs)
        else:
            data_out = self.down(data, **kwargs)
            data_out2 = self.submodule(data_out, **kwargs)
            data = data_out2, data
            return self.up(data, **kwargs)


_torch_metric_learning_miners = sys.modules['pytorch_metric_learning.miners']


_custom_losses = sys.modules['torch_points3d.core.losses.losses']


_torch_metric_learning_losses = sys.modules['pytorch_metric_learning.losses']


def instantiate_loss_or_miner(option, mode='loss'):
    """
    create a loss from an OmegaConf dict such as
    TripletMarginLoss.
    params:
        margin=0.1
    It can also instantiate a miner to better learn a loss
    """
    class_ = getattr(option, 'class', None)
    try:
        params = option.params
    except KeyError:
        params = None
    try:
        lparams = option.lparams
    except KeyError:
        lparams = None
    if 'loss' in mode:
        cls = getattr(_custom_losses, class_, None)
        if not cls:
            cls = getattr(_torch_metric_learning_losses, class_, None)
            if not cls:
                raise ValueError('loss %s is nowhere to be found' % class_)
    elif mode == 'miner':
        cls = getattr(_torch_metric_learning_miners, class_, None)
        if not cls:
            raise ValueError('miner %s is nowhere to be found' % class_)
    else:
        raise NotImplementedError('Cannot instantiate this mode {}'.format(
            mode))
    if params and lparams:
        return cls(*lparams, **params)
    if params:
        return cls(**params)
    if lparams:
        return cls(*params)
    return cls()


log = logging.getLogger(__name__)


_custom_lr_scheduler = sys.modules[__name__]


class LRScheduler:

    def __init__(self, scheduler, scheduler_params, update_scheduler_on):
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params
        self._update_scheduler_on = update_scheduler_on

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def scheduler_opt(self):
        return self._scheduler._scheduler_opt

    def __repr__(self):
        return '{}({}, update_scheduler_on={})'.format(self._scheduler.
            __class__.__name__, self._scheduler_params, self.
            _update_scheduler_on)

    def step(self, *args, **kwargs):
        self._scheduler.step(*args, **kwargs)

    def state_dict(self):
        return self._scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self._scheduler.load_state_dict(state_dict)


class BaseInternalLossModule(torch.nn.Module):
    """ABC for modules which have internal loss(es)
    """

    @abstractmethod
    def get_internal_losses(self) ->Dict[str, Any]:
        pass


def gather(x, idx, method=2):
    """
    https://github.com/pytorch/pytorch/issues/15245
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """
    idx[idx == -1] = x.shape[0] - 1
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-09):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig ** 2 + eps))


def KPConv_ops(query_points, support_points, neighbors_indices, features,
    K_points, K_values, KP_extent, KP_influence, aggregation_mode):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter
    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n0_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest
    :return:                    [n_points, out_fdim]
    """
    int(K_points.shape[0])
    shadow_point = torch.ones_like(support_points[:1, :]) * 1000000.0
    support_points = torch.cat([support_points, shadow_point], dim=0)
    neighbors = gather(support_points, neighbors_indices)
    neighbors = neighbors - query_points.unsqueeze(1)
    neighbors.unsqueeze_(2)
    differences = neighbors - K_points
    sq_distances = torch.sum(differences ** 2, dim=3)
    if KP_influence == 'constant':
        all_weights = torch.ones_like(sq_distances)
        all_weights = all_weights.transpose(2, 1)
    elif KP_influence == 'linear':
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / KP_extent,
            min=0.0)
        all_weights = all_weights.transpose(2, 1)
    elif KP_influence == 'gaussian':
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = all_weights.transpose(2, 1)
    else:
        raise ValueError(
            'Unknown influence function type (config.KP_influence)')
    if aggregation_mode == 'closest':
        neighbors_1nn = torch.argmin(sq_distances, dim=-1)
        all_weights *= torch.transpose(torch.nn.functional.one_hot(
            neighbors_1nn, K_points.shape[0]), 1, 2)
    elif aggregation_mode != 'sum':
        raise ValueError(
            "Unknown convolution mode. Should be 'closest' or 'sum'")
    features = torch.cat([features, torch.zeros_like(features[:1, :])], dim=0)
    neighborhood_features = gather(features, neighbors_indices)
    weighted_features = torch.matmul(all_weights, neighborhood_features)
    weighted_features = weighted_features.permute(1, 0, 2)
    kernel_outputs = torch.matmul(weighted_features, K_values)
    output_features = torch.sum(kernel_outputs, dim=0)
    return output_features


def add_ones(query_points, x, add_one):
    if add_one:
        ones = torch.ones(query_points.shape[0], dtype=torch.float).unsqueeze(
            -1).to(query_points.device)
        if x is not None:
            x = torch.cat([ones.to(x.dtype), x], dim=-1)
        else:
            x = ones
    return x


def makedirs(path):
    """
    taken from https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/makedirs.py
    """
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def header_properties(field_list, field_names):
    lines = []
    lines.append('element vertex %d' % field_list[0].shape[0])
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1
    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.
    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.
    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.
    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])
    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)
    """
    field_list = list(field_list) if type(field_list) == list or type(
        field_list) == tuple else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            log.info('fields have more than 2 dimensions')
            return False
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        log.info('wrong field dimensions')
        return False
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        log.info('wrong number of field names')
        return False
    if not filename.endswith('.ply'):
        filename += '.ply'
    with open(filename, 'w') as plyfile:
        header = ['ply']
        header.append('format binary_' + sys.byteorder + '_endian 1.0')
        header.extend(header_properties(field_list, field_names))
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0])
                )
            header.append('property list uchar int vertex_indices')
        header.append('end_header')
        for line in header:
            plyfile.write('%s\n' % line)
    with open(filename, 'ab') as plyfile:
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1
        data.tofile(plyfile)
        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in
                range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8
                )
            data['0'] = triangular_faces[:, (0)]
            data['1'] = triangular_faces[:, (1)]
            data['2'] = triangular_faces[:, (2)]
            data.tofile(plyfile)
    return True


def kernel_point_optimization_debug(radius, num_points, num_kernels=1,
    dimension=3, fixed='center', ratio=1.0, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """
    radius0 = 1
    diameter0 = 2
    moving_factor = 0.01
    continuous_moving_decay = 0.9995
    thresh = 1e-05
    clip = 0.05 * radius0
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension
        ) * diameter0 - radius0
    while kernel_points.shape[0] < num_kernels * num_points:
        new_points = np.random.rand(num_kernels * num_points - 1, dimension
            ) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[(d2 < 0.5 * radius0 * radius0), :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((
        num_kernels, num_points, -1))
    if fixed == 'center':
        kernel_points[:, (0), :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, (1), (-1)] += 2 * radius0 / 3
        kernel_points[:, (2), (-1)] -= 2 * radius0 / 3
    if verbose > 1:
        fig = plt.figure()
    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3 / 
            2) + 1e-06)
        inter_grads = np.sum(inter_grads, axis=1)
        circle_grads = 10 * kernel_points
        gradients = inter_grads + circle_grads
        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[(iter), :] = np.max(gradients_norms, axis=1)
        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] -
            gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3
            :] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)
        if fixed == 'center':
            moving_dists[:, (0)] = 0
        if fixed == 'verticals':
            moving_dists[:, (0)] = 0
        kernel_points -= np.expand_dims(moving_dists, -1
            ) * gradients / np.expand_dims(gradients_norms + 1e-06, -1)
        if verbose:
            log.info('iter {:5d} / max grad = {:f}'.format(iter, np.max(
                gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[(0), :, (0)], kernel_points[(0), :, (1)],
                '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_ylim((-radius * 1.1, radius * 1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            log.info(moving_factor)
        moving_factor *= continuous_moving_decay
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])
    return kernel_points * radius, saved_gradient_norms


ply_dtypes = dict([(b'int8', 'i1'), (b'char', 'i1'), (b'uint8', 'u1'), (
    b'uchar', 'u1'), (b'int16', 'i2'), (b'short', 'i2'), (b'uint16', 'u2'),
    (b'ushort', 'u2'), (b'int32', 'i4'), (b'int', 'i4'), (b'uint32', 'u4'),
    (b'uint', 'u4'), (b'float32', 'f4'), (b'float', 'f4'), (b'float64',
    'f8'), (b'double', 'f8')])


def parse_header(plyfile, ext):
    line = []
    properties = []
    num_points = None
    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        if b'element' in line:
            line = line.split()
            num_points = int(line[2])
        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
    return num_points, properties


valid_formats = {'ascii': '', 'binary_big_endian': '>',
    'binary_little_endian': '<'}


def parse_mesh_header(plyfile, ext):
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None
    while b'end_header' not in line and line != b'':
        line = plyfile.readline()
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])
        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])
        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext +
                    ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)
    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """
    with open(filename, 'rb') as plyfile:
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')
        fmt = plyfile.readline().split()[1].decode()
        if fmt == 'ascii':
            raise ValueError('The file is not binary')
        ext = valid_formats[fmt]
        if triangular_mesh:
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)
            vertex_data = np.fromfile(plyfile, dtype=properties, count=
                num_points)
            face_properties = [('k', ext + 'u1'), ('v1', ext + 'i4'), ('v2',
                ext + 'i4'), ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=
                num_faces)
            faces = np.vstack((faces_data['v1'], faces_data['v2'],
                faces_data['v3'])).T
            data = [vertex_data, faces]
        else:
            num_points, properties = parse_header(plyfile, ext)
            data = np.fromfile(plyfile, dtype=properties, count=num_points)
    return data


class BasicBlock(nn.Module):
    """This module implements a basic residual convolution block using MinkowskiEngine

    Parameters
    ----------
    inplanes: int
        Input dimension
    planes: int
        Output dimension
    dilation: int
        Dilation value
    downsample: nn.Module
        If provided, downsample will be applied on input before doing residual addition
    bn_momentum: float
        Input dimension
    """
    EXPANSION = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, bn_momentum=0.1, dimension=-1):
        super(BasicBlock, self).__init__()
        assert dimension > 0
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=
            3, stride=stride, dilation=dilation, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3,
            stride=1, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    EXPANSION = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, bn_momentum=0.1, dimension=-1):
        super(Bottleneck, self).__init__()
        assert dimension > 0
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=
            1, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3,
            stride=stride, dilation=dilation, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(planes, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(planes, planes * self.
            EXPANSION, kernel_size=1, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(planes * self.EXPANSION,
            momentum=bn_momentum)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16, D=-1):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(ME.MinkowskiLinear(channel, channel //
            reduction), ME.MinkowskiReLU(inplace=True), ME.MinkowskiLinear(
            channel // reduction, channel), ME.MinkowskiSigmoid())
        self.pooling = ME.MinkowskiGlobalPooling(dimension=D)
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication(dimension=D)

    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = 64, 128, 256, 512

    def __init__(self, in_channels, out_channels, D=3, **kwargs):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None, 'BLOCK is not defined'
        assert self.PLANES is not None, 'PLANES is not defined'
        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):
        self.inplanes = self.INIT_DIM
        self.conv1 = ME.MinkowskiConvolution(in_channels, self.inplanes,
            kernel_size=5, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=D
            )
        self.layer1 = self._make_layer(self.BLOCK, self.PLANES[0], self.
            LAYERS[0], stride=2)
        self.layer2 = self._make_layer(self.BLOCK, self.PLANES[1], self.
            LAYERS[1], stride=2)
        self.layer3 = self._make_layer(self.BLOCK, self.PLANES[2], self.
            LAYERS[2], stride=2)
        self.layer4 = self._make_layer(self.BLOCK, self.PLANES[3], self.
            LAYERS[3], stride=2)
        self.conv5 = ME.MinkowskiConvolution(self.inplanes, self.inplanes,
            kernel_size=3, stride=3, dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)
        self.glob_avg = ME.MinkowskiGlobalMaxPooling(dimension=D)
        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out',
                    nonlinearity='relu')
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.EXPANSION:
            downsample = nn.Sequential(ME.MinkowskiConvolution(self.
                inplanes, planes * block.EXPANSION, kernel_size=1, stride=
                stride, dimension=self.D), ME.MinkowskiBatchNorm(planes *
                block.EXPANSION))
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=
            dilation, downsample=downsample, dimension=self.D))
        self.inplanes = planes * block.EXPANSION
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=
                dilation, dimension=self.D))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.glob_avg(x)
        return self.final(x)


class ConvType(Enum):
    """
  Define the kernel region type
  """
    HYPERCUBE = 0, 'HYPERCUBE'
    SPATIAL_HYPERCUBE = 1, 'SPATIAL_HYPERCUBE'
    SPATIO_TEMPORAL_HYPERCUBE = 2, 'SPATIO_TEMPORAL_HYPERCUBE'
    HYPERCROSS = 3, 'HYPERCROSS'
    SPATIAL_HYPERCROSS = 4, 'SPATIAL_HYPERCROSS'
    SPATIO_TEMPORAL_HYPERCROSS = 5, 'SPATIO_TEMPORAL_HYPERCROSS'
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = (6,
        'SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS ')

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(ME.MinkowskiInstanceNorm(n_channels), ME.
            MinkowskiBatchNorm(n_channels, momentum=bn_momentum))
    else:
        raise ValueError(f'Norm type: {norm_type} not supported')


class BottleneckBase(nn.Module):
    expansion = 4
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, conv_type=ConvType.HYPERCUBE, bn_momentum=0.1, D=3):
        super(BottleneckBase, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=
            bn_momentum)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
            dilation=dilation, conv_type=conv_type, D=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=
            bn_momentum)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
        self.norm3 = get_norm(self.NORM_TYPE, planes * self.expansion, D,
            bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class XConv(torch.nn.Module):
    """The convolutional operator on :math:`\\mathcal{X}`-transformed points
    from the `"PointCNN: Convolution On X-Transformed Points"
    <https://arxiv.org/abs/1801.07791>`_ paper
    .. math::
        \\mathbf{x}^{\\prime}_i = \\mathrm{Conv}\\left(\\mathbf{K},
        \\gamma_{\\mathbf{\\Theta}}(\\mathbf{P}_i - \\mathbf{p}_i) \\times
        \\left( h_\\mathbf{\\Theta}(\\mathbf{P}_i - \\mathbf{p}_i) \\, \\Vert \\,
        \\mathbf{x}_i \\right) \\right),
    where :math:`\\mathbf{K}` and :math:`\\mathbf{P}_i` denote the trainable
    filter and neighboring point positions of :math:`\\mathbf{x}_i`,
    respectively.
    :math:`\\gamma_{\\mathbf{\\Theta}}` and :math:`h_{\\mathbf{\\Theta}}` describe
    neural networks, *i.e.* MLPs, where :math:`h_{\\mathbf{\\Theta}}`
    individually lifts each point into a higher-dimensional space, and
    :math:`\\gamma_{\\mathbf{\\Theta}}` computes the :math:`\\mathcal{X}`-
    transformation matrix based on *all* points in a neighborhood.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Point cloud dimensionality.
        kernel_size (int): Size of the convolving kernel, *i.e.* number of
            neighbors including self-loops.
        hidden_channels (int, optional): Output size of
            :math:`h_{\\mathbf{\\Theta}}`, *i.e.* dimensionality of lifted
            points. If set to :obj:`None`, will be automatically set to
            :obj:`in_channels / 4`. (default: :obj:`None`)
        dilation (int, optional): The factor by which the neighborhood is
            extended, from which :obj:`kernel_size` neighbors are then
            uniformly sampled. Can be interpreted as the dilation rate of
            classical convolutional operators. (default: :obj:`1`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_cluster.knn_graph`.
    """

    def __init__(self, in_channels, out_channels, dim, kernel_size,
        hidden_channels=None, dilation=1, bias=True, **kwargs):
        super(XConv, self).__init__()
        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        assert hidden_channels > 0
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.kwargs = kwargs
        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size
        self.mlp1 = S(L(dim, C_delta), ELU(), BN(C_delta), L(C_delta,
            C_delta), ELU(), BN(C_delta), Reshape(-1, K, C_delta))
        self.mlp2 = S(L(D * K, K ** 2), ELU(), BN(K ** 2), Reshape(-1, K, K
            ), Conv1d(K, K ** 2, K, groups=K), ELU(), BN(K ** 2), Reshape(-
            1, K, K), Conv1d(K, K ** 2, K, groups=K), BN(K ** 2), Reshape(-
            1, K, K))
        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = S(Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in),
            Reshape(-1, C_in * depth_multiplier), L(C_in * depth_multiplier,
            C_out, bias=bias))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)
        reset(self.conv)

    def forward(self, x, pos, edge_index):
        posFrom, posTo = pos
        (N, D), K = posTo.size(), self.kernel_size
        idxFrom, idxTo = edge_index
        relPos = posTo[idxTo] - posFrom[idxFrom]
        x_star = self.mlp1(relPos)
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            x = x[idxFrom].view(N, K, self.in_channels)
            x_star = torch.cat([x_star, x], dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()
        x_star = x_star.view(N, self.in_channels + self.hidden_channels, K, 1)
        transform_matrix = self.mlp2(relPos.view(N, K * D))
        transform_matrix = transform_matrix.view(N, 1, K, K)
        x_transformed = torch.matmul(transform_matrix, x_star)
        x_transformed = x_transformed.view(N, -1, K)
        out = self.conv(x_transformed)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.
            in_channels, self.out_channels)


class MiniPointNet(torch.nn.Module):

    def __init__(self, local_nn, global_nn, aggr='max', return_local_out=False
        ):
        super().__init__()
        self.local_nn = MLP(local_nn)
        self.global_nn = MLP(global_nn) if global_nn else None
        self.g_pool = global_max_pool if aggr == 'max' else global_mean_pool
        self.return_local_out = return_local_out

    def forward(self, x, batch):
        y = x = self.local_nn(x)
        x = self.g_pool(x, batch)
        if self.global_nn:
            x = self.global_nn(x)
        if self.return_local_out:
            return x, y
        return x

    def forward_embedding(self, pos, batch):
        global_feat, local_feat = self.forward(pos, batch)
        indices = batch.unsqueeze(-1).repeat((1, global_feat.shape[-1]))
        gathered_global_feat = torch.gather(global_feat, 0, indices)
        x = torch.cat([local_feat, gathered_global_feat], -1)
        return x


class PointNetSTNkD(BaseLinearTransformSTNkD, BaseInternalLossModule):

    def __init__(self, k=64, local_nn=[64, 64, 128, 1024], global_nn=[1024,
        512, 256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1],
            k, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)

    def get_internal_losses(self):
        return {'orthogonal_regularization_loss': self.
            get_orthogonal_regularization_loss()}


class PointNetSTN3D(BaseLinearTransformSTNkD):

    def __init__(self, local_nn=[3, 64, 128, 1024], global_nn=[1024, 512, 
        256], batch_size=1):
        super().__init__(MiniPointNet(local_nn, global_nn), global_nn[-1], 
            3, batch_size)

    def forward(self, x, batch):
        return super().forward(x, x, batch)


class PointNetSeg(torch.nn.Module):

    def __init__(self, input_stn_local_nn=[3, 64, 128, 1024],
        input_stn_global_nn=[1024, 512, 256], local_nn_1=[3, 64, 64],
        feat_stn_k=64, feat_stn_local_nn=[64, 64, 128, 1024],
        feat_stn_global_nn=[1024, 512, 256], local_nn_2=[64, 64, 128, 1024],
        seg_nn=[1088, 512, 256, 128, 4], batch_size=1, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.input_stn = PointNetSTN3D(input_stn_local_nn,
            input_stn_global_nn, batch_size)
        self.local_nn_1 = MLP(local_nn_1)
        self.feat_stn = PointNetSTNkD(feat_stn_k, feat_stn_local_nn,
            feat_stn_global_nn, batch_size)
        self.local_nn_2 = MLP(local_nn_2)
        self.seg_nn = MLP(seg_nn)
        self._use_scatter_pooling = True

    def set_scatter_pooling(self, use_scatter_pooling):
        self._use_scatter_pooling = use_scatter_pooling

    def func_global_max_pooling(self, x3, batch):
        if self._use_scatter_pooling:
            return global_max_pool(x3, batch)
        else:
            global_feature = x3.max(1)
            return global_feature[0]

    def forward(self, x, batch):
        x = self.input_stn(x, batch)
        x = self.local_nn_1(x)
        x_feat_trans = self.feat_stn(x, batch)
        x3 = self.local_nn_2(x_feat_trans)
        global_feature = self.func_global_max_pooling(x3, batch)
        feat_concat = torch.cat([x_feat_trans, global_feature[batch]], dim=1)
        out = self.seg_nn(feat_concat)
        return out


class RSConvMapper(nn.Module):
    """[This class handles the special mechanism between the msg
        and the features of RSConv]
    """

    def __init__(self, down_conv_nn, use_xyz, bn=True, activation=nn.
        LeakyReLU(negative_slope=0.01), *args, **kwargs):
        super(RSConvMapper, self).__init__()
        self._down_conv_nn = down_conv_nn
        self._use_xyz = use_xyz
        self.nn = nn.ModuleDict()
        if len(self._down_conv_nn) == 2:
            self._first_layer = True
            f_in, f_intermediate, f_out = self._down_conv_nn[0]
            self.nn['features_nn'] = MLP2D(self._down_conv_nn[1], bn=bn,
                bias=False)
        else:
            self._first_layer = False
            f_in, f_intermediate, f_out = self._down_conv_nn
        self.nn['mlp_msg'] = MLP2D([f_in, f_intermediate, f_out], bn=bn,
            bias=False)
        self.nn['norm'] = Sequential(*[nn.BatchNorm2d(f_out), activation])
        self._f_out = f_out

    @property
    def f_out(self):
        return self._f_out

    def forward(self, features, msg):
        """
        features  -- [B, C, num_points, nsamples]
        msg  -- [B, 10, num_points, nsamples]

        The 10 features comes from [distance: 1,
                                    coord_origin:3,
                                    coord_target:3,
                                    delta_origin_target:3]
        """
        msg = self.nn['mlp_msg'](msg)
        if self._first_layer:
            features = self.nn['features_nn'](features)
        return self.nn['norm'](torch.mul(features, msg))


class SharedRSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(self, mapper: RSConvMapper, radius):
        super(SharedRSConv, self).__init__()
        self._mapper = mapper
        self._radius = radius

    def forward(self, aggr_features, centroids):
        """
        aggr_features  -- [B, 3 + 3 + C, num_points, nsamples]
        centroids  -- [B, 3, num_points, 1]
        """
        abs_coord = aggr_features[:, :3]
        delta_x = aggr_features[:, 3:6]
        features = aggr_features[:, 3:]
        nsample = abs_coord.shape[-1]
        coord_xi = centroids.repeat(1, 1, 1, nsample)
        distance = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)
        h_xi_xj = torch.cat((distance, coord_xi, abs_coord, delta_x), dim=1)
        return self._mapper(features, h_xi_xj)

    def __repr__(self):
        return '{}(radius={})'.format(self.__class__.__name__, self._radius)


class OriginalRSConv(nn.Module):
    """
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    """

    def __init__(self, mapping=None, first_layer=False, radius=None,
        activation=nn.ReLU(inplace=True)):
        super(OriginalRSConv, self).__init__()
        self.nn = nn.ModuleList()
        self._radius = radius
        self.mapping_func1 = mapping[0]
        self.mapping_func2 = mapping[1]
        self.cr_mapping = mapping[2]
        self.first_layer = first_layer
        if first_layer:
            self.xyz_raising = mapping[3]
            self.bn_xyz_raising = nn.BatchNorm2d(self.xyz_raising.out_channels)
            self.nn.append(self.bn_xyz_raising)
        self.bn_mapping = nn.BatchNorm2d(self.mapping_func1.out_channels)
        self.bn_rsconv = nn.BatchNorm2d(self.cr_mapping.in_channels)
        self.bn_channel_raising = nn.BatchNorm1d(self.cr_mapping.out_channels)
        self.nn.append(self.bn_mapping)
        self.nn.append(self.bn_rsconv)
        self.nn.append(self.bn_channel_raising)
        self.activation = activation

    def forward(self, input):
        x = input[:, 3:, :, :]
        nsample = x.size()[3]
        abs_coord = input[:, 0:3, :, :]
        delta_x = input[:, 3:6, :, :]
        coord_xi = abs_coord[:, :, :, 0:1].repeat(1, 1, 1, nsample)
        h_xi_xj = torch.norm(delta_x, p=2, dim=1).unsqueeze(1)
        h_xi_xj = torch.cat((h_xi_xj, coord_xi, abs_coord, delta_x), dim=1)
        h_xi_xj = self.mapping_func2(self.activation(self.bn_mapping(self.
            mapping_func1(h_xi_xj))))
        if self.first_layer:
            x = self.activation(self.bn_xyz_raising(self.xyz_raising(x)))
        x = F.max_pool2d(self.activation(self.bn_rsconv(torch.mul(h_xi_xj,
            x))), kernel_size=(1, nsample)).squeeze(3)
        x = self.activation(self.bn_channel_raising(self.cr_mapping(x)))
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.nn.__repr__())


class VotingModule(nn.Module):

    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super().__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3 + self.out_dim) * self
            .vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

    def forward(self, data):
        """ Votes for centres using a PN++ like architecture
        Returns
        -------
        data:
            - pos: position of the vote (centre of the box)
            - x: feature of the vote (original feature + processed feature)
            - seed_pos: position of the original point
        """
        if data.pos.dim() != 3:
            raise ValueError(
                'This method only supports dense convolutions for now')
        batch_size = data.pos.shape[0]
        num_points = data.pos.shape[1]
        num_votes = num_points * self.vote_factor
        x = F.relu(self.bn1(self.conv1(data.x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = x.transpose(2, 1).view(batch_size, num_points, self.vote_factor,
            3 + self.out_dim)
        offset = x[:, :, :, 0:3]
        vote_pos = data.pos.unsqueeze(2) + offset
        vote_pos = vote_pos.contiguous().view(batch_size, num_votes, 3)
        res_x = x[:, :, :, 3:]
        vote_x = data.x.transpose(2, 1).unsqueeze(2) + res_x
        vote_x = vote_x.contiguous().view(batch_size, num_votes, self.out_dim)
        vote_x = vote_x.transpose(2, 1).contiguous()
        return Data(pos=vote_pos, x=vote_x, seed_pos=data.pos)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_nicolas_chaulet_torch_points3d(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ContrastiveHardestNegativeLoss(*[], **{'pos_thresh': 4, 'neg_thresh': 4}), [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(HuberLoss(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(LossAnnealer(*[], **{'args': _mock_config()}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Seq(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
