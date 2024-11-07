
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


import numpy as np


import torch.nn.functional as F


import math


import scipy.special


from torch import nn


import torch.nn as nn


from torch.nn import Parameter


import warnings


from torch.utils.data.sampler import Sampler


import itertools


from collections import defaultdict


import copy


from sklearn.metrics import adjusted_mutual_info_score


from sklearn.metrics import normalized_mutual_info_score


import collections


import logging


import re


import scipy.stats


from torch.autograd import Variable


from numpy.testing import assert_almost_equal


from torch import Tensor


import scipy


from torch.nn.modules.module import Module


import torch.nn


import torch.nn.functional


from torch.nn import init


from torch.nn.parameter import Parameter


from collections import Counter


from torchvision import datasets


from torchvision import models


from torchvision import transforms


from sklearn.preprocessing import StandardScaler


import torch.distributed as dist


import torch.multiprocessing as mp


import torch.optim as optim


from torch.nn.parallel import DistributedDataParallel as DDP


import uuid


import torchvision


class BatchedDistance(torch.nn.Module):

    def __init__(self, distance, iter_fn=None, batch_size=32):
        super().__init__()
        self.distance = distance
        self.iter_fn = iter_fn
        self.batch_size = batch_size

    def forward(self, query_emb, ref_emb=None):
        ref_emb = ref_emb if ref_emb is not None else query_emb
        n = query_emb.shape[0]
        for s in range(0, n, self.batch_size):
            e = s + self.batch_size
            L = query_emb[s:e]
            mat = self.distance(L, ref_emb)
            self.iter_fn(mat, s, e)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.distance, name)


class BaseLossWrapper(torch.nn.Module):

    def __init__(self, loss, **kwargs):
        super().__init__(**kwargs)
        loss_name = type(loss).__name__
        self.check_loss_support(loss_name)

    @staticmethod
    def supported_losses():
        raise NotImplementedError

    @classmethod
    def check_loss_support(self, loss_name):
        raise NotImplementedError


class ModuleWithRecords(torch.nn.Module):

    def __init__(self, collect_stats=None):
        super().__init__()
        self.collect_stats = c_f.COLLECT_STATS if collect_stats is None else collect_stats

    def add_to_recordable_attributes(self, name=None, list_of_names=None, is_stat=False):
        if is_stat and not self.collect_stats:
            pass
        else:
            c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names, is_stat=is_stat)

    def reset_stats(self):
        c_f.reset_stats(self)


class CrossBatchMemory(BaseLossWrapper, ModuleWithRecords):

    def __init__(self, loss, embedding_size, memory_size=1024, miner=None, **kwargs):
        super().__init__(loss=loss, **kwargs)
        self.loss = loss
        self.miner = miner
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.reset_queue()
        self.add_to_recordable_attributes(list_of_names=['embedding_size', 'memory_size', 'queue_idx'], is_stat=False)

    @staticmethod
    def supported_losses():
        return ['AngularLoss', 'CircleLoss', 'ContrastiveLoss', 'GeneralizedLiftedStructureLoss', 'IntraPairVarianceLoss', 'LiftedStructureLoss', 'MarginLoss', 'MultiSimilarityLoss', 'NCALoss', 'NTXentLoss', 'SignalToNoiseRatioContrastiveLoss', 'SupConLoss', 'TripletMarginLoss', 'TupletMarginLoss']

    @classmethod
    def check_loss_support(cls, loss_name):
        if loss_name not in cls.supported_losses():
            raise Exception(f'CrossBatchMemory not supported for {loss_name}')

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_mask=None):
        if indices_tuple is not None and enqueue_mask is not None:
            raise ValueError('indices_tuple and enqueue_mask are mutually exclusive')
        if enqueue_mask is not None:
            assert len(enqueue_mask) == len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        self.reset_stats()
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(self.embedding_memory, device=device, dtype=embeddings.dtype)
        self.label_memory = c_f.to_device(self.label_memory, device=device, dtype=labels.dtype)
        if enqueue_mask is not None:
            emb_for_queue = embeddings[enqueue_mask]
            labels_for_queue = labels[enqueue_mask]
            embeddings = embeddings[~enqueue_mask]
            labels = labels[~enqueue_mask]
            do_remove_self_comparisons = False
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            do_remove_self_comparisons = True
        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)
        if not self.has_been_filled:
            E_mem = self.embedding_memory[:self.queue_idx]
            L_mem = self.label_memory[:self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory
        indices_tuple = self.create_indices_tuple(embeddings, labels, E_mem, L_mem, indices_tuple, do_remove_self_comparisons)
        loss = self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)
        return loss

    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = torch.arange(self.queue_idx, self.queue_idx + batch_size, device=labels.device) % self.memory_size
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if not self.has_been_filled and self.queue_idx <= prev_queue_idx:
            self.has_been_filled = True

    def create_indices_tuple(self, embeddings, labels, E_mem, L_mem, input_indices_tuple, do_remove_self_comparisons):
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)
        if do_remove_self_comparisons:
            indices_tuple = lmu.remove_self_comparisons(indices_tuple, self.curr_batch_idx, self.memory_size)
        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = lmu.convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = lmu.convert_to_triplets(input_indices_tuple, labels)
            indices_tuple = c_f.concatenate_indices_tuples(indices_tuple, input_indices_tuple)
        return indices_tuple

    def reset_queue(self):
        self.register_buffer('embedding_memory', torch.zeros(self.memory_size, self.embedding_size))
        self.register_buffer('label_memory', torch.zeros(self.memory_size).long())
        self.has_been_filled = False
        self.queue_idx = 0


class MultipleLosses(torch.nn.Module):

    def __init__(self, losses, miners=None, weights=None):
        super().__init__()
        self.is_dict = isinstance(losses, dict)
        self.losses = torch.nn.ModuleDict(losses) if self.is_dict else torch.nn.ModuleList(losses)
        if miners is not None:
            self.assertions_if_not_none(miners, match_all_keys=False)
            self.miners = torch.nn.ModuleDict(miners) if self.is_dict else torch.nn.ModuleList(miners)
        else:
            self.miners = None
        if weights is not None:
            self.assertions_if_not_none(weights, match_all_keys=True)
            self.weights = weights
        else:
            self.weights = {k: (1) for k in self.losses.keys()} if self.is_dict else [1] * len(losses)

    def forward(self, embeddings, labels, indices_tuple=None):
        if self.miners:
            assert indices_tuple is None
        total_loss = 0
        iterable = self.losses.items() if self.is_dict else enumerate(self.losses)
        for i, loss_func in iterable:
            curr_indices_tuple = self.get_indices_tuple(i, embeddings, labels, indices_tuple)
            total_loss += loss_func(embeddings, labels, curr_indices_tuple) * self.weights[i]
        return total_loss

    def get_indices_tuple(self, i, embeddings, labels, indices_tuple):
        if self.miners:
            if self.is_dict and i in self.miners or not self.is_dict and self.miners[i] is not None:
                indices_tuple = self.miners[i](embeddings, labels)
        return indices_tuple

    def assertions_if_not_none(self, x, match_all_keys):
        if x is not None:
            if self.is_dict:
                assert isinstance(x, dict)
                if match_all_keys:
                    assert sorted(list(x.keys())) == sorted(list(self.losses.keys()))
                else:
                    assert all(k in self.losses.keys() for k in x.keys())
            else:
                assert c_f.is_list_or_tuple(x)
                assert len(x) == len(self.losses)


class SelfSupervisedLoss(BaseLossWrapper):
    """
    Issue #411:

    A common use case is to have embeddings and ref_emb be augmented versions of each other.
    For most losses right now you have to create labels to indicate
    which embeddings correspond with which ref_emb.
    A wrapper that does this for the user would be nice.

        loss_fn = SelfSupervisedLoss(TripletMarginLoss())
        loss = loss_fn(embeddings, ref_emb1, ref_emb2, ...)

    where ref_embk = kth augmentation of embeddings.
    """

    def __init__(self, loss, symmetric=True, **kwargs):
        super().__init__(loss=loss, **kwargs)
        self.loss = loss
        self.symmetric = symmetric

    @staticmethod
    def supported_losses():
        return ['AngularLoss', 'CircleLoss', 'ContrastiveLoss', 'GeneralizedLiftedStructureLoss', 'IntraPairVarianceLoss', 'LiftedStructureLoss', 'MultiSimilarityLoss', 'NTXentLoss', 'SignalToNoiseRatioContrastiveLoss', 'SupConLoss', 'TripletMarginLoss', 'NCALoss', 'TupletMarginLoss']

    @classmethod
    def check_loss_support(cls, loss_name):
        if loss_name not in cls.supported_losses():
            raise Exception(f'SelfSupervisedLoss not supported for {loss_name}')

    def forward(self, embeddings, ref_emb):
        """
        embeddings: representations of the original set of inputs
        ref_emb:    representations of an augmentation of the inputs.
        *args:      variable length argument list, where each argument
                    is an additional representation of an augmented version of the input.
                    i.e. ref_emb2, ref_emb3, ...
        """
        labels = torch.arange(embeddings.shape[0])
        if self.symmetric:
            embeddings = torch.cat([embeddings, ref_emb], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            return self.loss(embeddings, labels)
        return self.loss(embeddings=embeddings, labels=labels, ref_emb=ref_emb, ref_labels=labels.clone())


class EmbeddingRegularizerMixin:

    def __init__(self, embedding_regularizer=None, embedding_reg_weight=1, **kwargs):
        self.embedding_regularizer = embedding_regularizer is not None
        super().__init__(**kwargs)
        self.embedding_regularizer = embedding_regularizer
        self.embedding_reg_weight = embedding_reg_weight
        if self.embedding_regularizer is not None:
            self.add_to_recordable_attributes(list_of_names=['embedding_reg_weight'], is_stat=False)

    def embedding_regularization_loss(self, embeddings):
        if self.embedding_regularizer is None:
            loss = 0
        else:
            loss = self.embedding_regularizer(embeddings) * self.embedding_reg_weight
        return {'losses': loss, 'indices': None, 'reduction_type': 'already_reduced'}

    def add_embedding_regularization_to_loss_dict(self, loss_dict, embeddings):
        if self.embedding_regularizer is not None:
            loss_dict['embedding_reg_loss'] = self.embedding_regularization_loss(embeddings)

    def regularization_loss_names(self):
        return ['embedding_reg_loss']


class BaseDistance(ModuleWithRecords):

    def __init__(self, normalize_embeddings=True, p=2, power=1, is_inverted=False, **kwargs):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.add_to_recordable_attributes(list_of_names=['p', 'power'], is_stat=False)
        self.add_to_recordable_attributes(list_of_names=['initial_avg_query_norm', 'initial_avg_ref_norm', 'final_avg_query_norm', 'final_avg_ref_norm'], is_stat=True)

    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        self.check_shapes(query_emb, ref_emb)
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(query_emb, ref_emb, query_emb_normalized, ref_emb_normalized)
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat ** self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        raise NotImplementedError

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.max(*args, **kwargs)
        return torch.min(*args, **kwargs)

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            return torch.min(*args, **kwargs)
        return torch.max(*args, **kwargs)

    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, dim=1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)

    def set_default_stats(self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized):
        if self.collect_stats:
            with torch.no_grad():
                self.initial_avg_query_norm = torch.mean(self.get_norm(query_emb)).item()
                self.initial_avg_ref_norm = torch.mean(self.get_norm(ref_emb)).item()
                self.final_avg_query_norm = torch.mean(self.get_norm(query_emb_normalized)).item()
                self.final_avg_ref_norm = torch.mean(self.get_norm(ref_emb_normalized)).item()

    def check_shapes(self, query_emb, ref_emb):
        if query_emb.ndim != 2 or ref_emb is not None and ref_emb.ndim != 2:
            raise ValueError('embeddings must be a 2D tensor of shape (batch_size, embedding_size)')


class LpDistance(BaseDistance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == torch.float16:
            rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, dim=0)
            output = torch.zeros(rows.size(), dtype=dtype, device=device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output
        else:
            return torch.cdist(query_emb, ref_emb, p=self.p)

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.nn.functional.pairwise_distance(query_emb, ref_emb, p=self.p)


class ModuleWithRecordsAndDistance(ModuleWithRecords):

    def __init__(self, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = self.get_default_distance() if distance is None else distance

    def get_default_distance(self):
        return LpDistance(p=2)


class BaseReducer(ModuleWithRecords):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_to_recordable_attributes(name='losses_size', is_stat=True)

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        assert len(loss_dict) == 1
        loss_name = list(loss_dict.keys())[0]
        loss_info = loss_dict[loss_name]
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(losses, loss_indices, reduction_type, kwargs, embeddings, labels)
        return loss_val

    def unpack_loss_info(self, loss_info):
        return loss_info['losses'], loss_info['indices'], loss_info['reduction_type'], {}

    def reduce_the_loss(self, losses, loss_indices, reduction_type, kwargs, embeddings, labels):
        self.set_losses_size_stat(losses)
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def get_reduction_func(self, reduction_type):
        return getattr(self, '{}_reduction'.format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, 'assert_sizes_{}'.format(reduction_type))(losses, loss_indices)

    def zero_loss(self, embeddings):
        return torch.sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        if not torch.is_tensor(losses) and losses == 0:
            return True
        return False

    def assert_sizes_already_reduced(self, losses, loss_indices):
        pass

    def assert_sizes_element(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert torch.is_tensor(loss_indices)
        assert len(losses) == len(loss_indices)

    def assert_sizes_pair(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 2
        assert all(torch.is_tensor(x) for x in loss_indices)
        assert len(losses) == len(loss_indices[0]) == len(loss_indices[1])

    def assert_sizes_pos_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_neg_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_triplet(self, losses, loss_indices):
        assert torch.is_tensor(losses)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)

    def set_losses_size_stat(self, losses):
        if self.collect_stats:
            if not torch.is_tensor(losses) or losses.ndim == 0:
                self.losses_size = 1
            else:
                self.losses_size = len(losses)


class DoNothingReducer(BaseReducer):

    def forward(self, loss_dict, embeddings, labels):
        return loss_dict


class MeanReducer(BaseReducer):

    def element_reduction(self, losses, *_):
        return torch.mean(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def triplet_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)


class MultipleReducers(BaseReducer):

    def __init__(self, reducers, default_reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.reducers = torch.nn.ModuleDict(reducers)
        self.default_reducer = MeanReducer() if default_reducer is None else default_reducer

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        sub_losses = torch.zeros(len(loss_dict), dtype=embeddings.dtype, device=embeddings.device)
        loss_count = 0
        for loss_name, loss_info in loss_dict.items():
            input_dict = {loss_name: loss_info}
            if loss_name in self.reducers:
                loss_val = self.reducers[loss_name](input_dict, embeddings, labels)
            else:
                loss_val = self.default_reducer(input_dict, embeddings, labels)
            sub_losses[loss_count] = loss_val
            loss_count += 1
        return self.sub_loss_reduction(sub_losses, embeddings, labels)

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        return torch.sum(sub_losses)


class ModuleWithRecordsAndReducer(ModuleWithRecords):

    def __init__(self, reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.set_reducer(reducer)

    def get_default_reducer(self):
        return MeanReducer()

    def set_reducer(self, reducer):
        if isinstance(reducer, (MultipleReducers, DoNothingReducer)):
            self.reducer = reducer
        elif len(self.sub_loss_names()) == 1:
            self.reducer = self.get_default_reducer() if reducer is None else copy.deepcopy(reducer)
        else:
            reducer_dict = {}
            for k in self.sub_loss_names():
                reducer_dict[k] = self.get_default_reducer() if reducer is None else copy.deepcopy(reducer)
            self.reducer = MultipleReducers(reducer_dict)

    def sub_loss_names(self):
        return ['loss']


class ModuleWithRecordsReducerAndDistance(ModuleWithRecordsAndReducer, ModuleWithRecordsAndDistance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BaseMetricLossFunction(EmbeddingRegularizerMixin, ModuleWithRecordsReducerAndDistance):

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def forward(self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

    def zero_loss(self):
        return {'losses': 0, 'indices': None, 'reduction_type': 'already_reduced'}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        return ['loss']

    def sub_loss_names(self):
        return self._sub_loss_names() + self.all_regularization_loss_names()

    def all_regularization_loss_names(self):
        reg_names = []
        for base_class in inspect.getmro(self.__class__):
            base_class_name = base_class.__name__
            mixin_keyword = 'RegularizerMixin'
            if base_class_name.endswith(mixin_keyword):
                descriptor = base_class_name.replace(mixin_keyword, '').lower()
                if getattr(self, '{}_regularizer'.format(descriptor)):
                    reg_names.extend(base_class.regularization_loss_names(self))
        return reg_names


def all_gather(x):
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        rank = torch.distributed.get_rank()
        x_list = [torch.ones_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(x_list, x.contiguous())
        x_list.pop(rank)
        return torch.cat(x_list, dim=0)
    return None


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def all_gather_embeddings_and_labels(emb, labels):
    if not is_distributed():
        return None, None
    ref_emb = all_gather(emb)
    ref_labels = all_gather(labels) if labels is not None else None
    return ref_emb, ref_labels


def gather(emb, labels):
    device = emb.device
    if labels is not None:
        labels = c_f.to_device(labels, device=device)
    dist_emb, dist_labels = all_gather_embeddings_and_labels(emb, labels)
    all_emb = torch.cat([emb, dist_emb], dim=0)
    all_labels = torch.cat([labels, dist_labels], dim=0) if dist_labels is not None else None
    return all_emb, all_labels, labels


def gather_emb_and_ref(emb, labels, ref_emb=None, ref_labels=None):
    all_emb, all_labels, labels = gather(emb, labels)
    all_ref_emb, all_ref_labels = None, None
    if ref_emb is not None:
        all_ref_emb, all_ref_labels, _ = gather(ref_emb, ref_labels)
    return all_emb, all_labels, all_ref_emb, all_ref_labels, labels


def gather_enqueue_mask(enqueue_mask, device):
    if enqueue_mask is None:
        return enqueue_mask
    enqueue_mask = c_f.to_device(enqueue_mask, device=device)
    return torch.cat([enqueue_mask, all_gather(enqueue_mask)], dim=0)


def get_indices_tuple(labels, ref_labels, embeddings=None, ref_emb=None, miner=None):
    device = labels.device
    curr_batch_idx = torch.arange(len(labels), device=device)
    if miner:
        indices_tuple = miner(embeddings, labels, ref_emb, ref_labels)
    else:
        indices_tuple = lmu.get_all_pairs_indices(labels, ref_labels)
    return lmu.remove_self_comparisons(indices_tuple, curr_batch_idx, len(ref_labels))


def select_ref_or_regular(regular, ref):
    return regular if ref is None else ref


class DistributedLossWrapper(torch.nn.Module):

    def __init__(self, loss, efficient=False):
        super().__init__()
        if not isinstance(loss, (BaseMetricLossFunction, CrossBatchMemory)):
            raise TypeError('The input loss must extend BaseMetricLossFunction or CrossBatchMemory')
        if isinstance(loss, CrossBatchMemory) and efficient:
            raise ValueError('CrossBatchMemory with efficient=True is not currently supported')
        self.loss = loss
        self.efficient = efficient

    def forward(self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None, enqueue_mask=None):
        if not is_distributed():
            warnings.warn('DistributedLossWrapper is being used in a non-distributed setting. Returning the loss as is.')
            return self.loss(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        world_size = torch.distributed.get_world_size()
        common_args = [embeddings, labels, indices_tuple, ref_emb, ref_labels, world_size]
        if isinstance(self.loss, CrossBatchMemory):
            return self.forward_cross_batch(*common_args, enqueue_mask)
        return self.forward_regular_loss(*common_args)

    def forward_regular_loss(self, emb, labels, indices_tuple, ref_emb, ref_labels, world_size):
        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple, ref_emb, ref_labels)
        all_emb, all_labels, all_ref_emb, all_ref_labels, labels = gather_emb_and_ref(emb, labels, ref_emb, ref_labels)
        if self.efficient:
            if all_labels is not None:
                all_labels = select_ref_or_regular(all_labels, all_ref_labels)
            all_emb = select_ref_or_regular(all_emb, all_ref_emb)
            if indices_tuple is None:
                indices_tuple = get_indices_tuple(labels, all_labels)
            loss = self.loss(emb, labels, indices_tuple, all_emb, all_labels)
        else:
            loss = self.loss(all_emb, all_labels, indices_tuple, all_ref_emb, all_ref_labels)
        return loss * world_size

    def forward_cross_batch(self, emb, labels, indices_tuple, ref_emb, ref_labels, world_size, enqueue_mask):
        if ref_emb is not None or ref_labels is not None:
            raise ValueError('CrossBatchMemory is not compatible with ref_emb and ref_labels')
        if world_size <= 1:
            return self.loss(emb, labels, indices_tuple, enqueue_mask)
        all_emb, all_labels, _, _, _ = gather_emb_and_ref(emb, labels, ref_emb, ref_labels)
        enqueue_mask = gather_enqueue_mask(enqueue_mask, emb.device)
        loss = self.loss(all_emb, all_labels, indices_tuple, enqueue_mask)
        return loss * world_size


class BaseMiner(ModuleWithRecordsAndDistance):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(list_of_names=['num_pos_pairs', 'num_neg_pairs', 'num_triplets'], is_stat=True)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        raise NotImplementedError

    def output_assertion(self, output):
        """
        Args:
            output: the output of self.mine
        This asserts that the mining function is outputting
        properly formatted indices. The default is to require a tuple representing
        a,p,n indices or a1,p,a2,n indices within a batch of embeddings.
        For example, a tuple of (anchors, positives, negatives) will be
        (torch.tensor, torch.tensor, torch.tensor)
        """
        if len(output) == 3:
            self.num_triplets = len(output[0])
            assert self.num_triplets == len(output[1]) == len(output[2])
        elif len(output) == 4:
            self.num_pos_pairs = len(output[0])
            self.num_neg_pairs = len(output[2])
            assert self.num_pos_pairs == len(output[1])
            assert self.num_neg_pairs == len(output[3])
        else:
            raise TypeError

    def forward(self, embeddings, labels, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
        Does any necessary preprocessing, then does mining, and then checks the
        shape of the mining output before returning it
        """
        self.reset_stats()
        with torch.no_grad():
            c_f.check_shapes(embeddings, labels)
            labels = c_f.to_device(labels, embeddings)
            ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
            mining_output = self.mine(embeddings, labels, ref_emb, ref_labels)
        self.output_assertion(mining_output)
        return mining_output


class DistributedMinerWrapper(torch.nn.Module):

    def __init__(self, miner, efficient=False):
        super().__init__()
        if not isinstance(miner, BaseMiner):
            raise TypeError('The input miner must extend BaseMiner')
        self.miner = miner
        self.efficient = efficient

    def forward(self, emb, labels, ref_emb=None, ref_labels=None):
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return self.miner(emb, labels, ref_emb, ref_labels)
        all_emb, all_labels, all_ref_emb, all_ref_labels, labels = gather_emb_and_ref(emb, labels, ref_emb, ref_labels)
        if self.efficient:
            all_labels = select_ref_or_regular(all_labels, all_ref_labels)
            all_emb = select_ref_or_regular(all_emb, all_ref_emb)
            return get_indices_tuple(labels, all_labels, emb, all_emb, self.miner)
        else:
            return self.miner(all_emb, all_labels, all_ref_emb, all_ref_labels)


def compute_distance_matrix_unit_l2(a, b, eps=1e-06):
    """
    computes pairwise Euclidean distance and return a N x N matrix
    """
    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = ((1.0 - dmat + eps) * 2.0).pow(0.5)
    return dmat


def find_hard_negatives(dmat, output_index=True, empirical_thresh=0.0):
    """
    a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ...
    """
    r, c = dmat.size()
    if not output_index:
        pos = torch.zeros(max(r, c))
        pos[:min(r, c)] = dmat.diag()
    dmat = dmat + torch.eye(r, c) * 99999
    dmat[dmat < empirical_thresh] = 99999
    min_a, min_p = torch.zeros(max(r, c)), torch.zeros(max(r, c))
    min_a[:c], _ = torch.min(dmat, dim=0)
    min_p[:r], _ = torch.min(dmat, dim=1)
    if not output_index:
        neg = torch.min(min_a, min_p)
        return pos, neg


class OriginalImplementationDynamicSoftMarginLoss(nn.Module):

    def __init__(self, is_binary=False, momentum=0.01, max_dist=None, nbins=512):
        """
        is_binary: true if learning binary descriptor
        momentum: weight assigned to the histogram computed from the current batch
        max_dist: maximum possible distance in the feature space
        nbins: number of bins to discretize the PDF
        """
        super(OriginalImplementationDynamicSoftMarginLoss, self).__init__()
        self._is_binary = is_binary
        if max_dist is None:
            max_dist = 2.0
        self._momentum = momentum
        self._max_val = max_dist
        self._min_val = -max_dist
        self.register_buffer('histogram', torch.ones(nbins))
        self._stats_initialized = False
        self.current_step = None

    def _compute_distances(self, x, labels=None):
        return self._compute_l2_distances(x, labels=labels)

    def _compute_l2_distances(self, x, labels=None):
        if labels is None:
            cnt = x.size(0) // 2
            a = x[:cnt, :]
            p = x[cnt:, :]
            dmat = compute_distance_matrix_unit_l2(a, p)
            return find_hard_negatives(dmat, output_index=False, empirical_thresh=0.008)
        else:
            dmat = compute_distance_matrix_unit_l2(x, x)
            dmat.fill_diagonal_(0)
            anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(None, labels, labels, t_per_anchor='all')
            return dmat[anchor_idx, positive_idx], dmat[anchor_idx, negative_idx]

    def _compute_histogram(self, x, momentum):
        """
        update the histogram using the current batch
        """
        num_bins = self.histogram.size(0)
        x_detached = x.detach()
        self.bin_width = (self._max_val - self._min_val) / num_bins
        lo = torch.floor((x_detached - self._min_val) / self.bin_width).long()
        hi = (lo + 1).clamp(min=0, max=num_bins - 1)
        hist = x.new_zeros(num_bins)
        alpha = 1.0 - (x_detached - self._min_val - lo.float() * self.bin_width) / self.bin_width
        hist.index_add_(0, lo, alpha)
        hist.index_add_(0, hi, 1.0 - alpha)
        hist = hist / (hist.sum() + 1e-06)
        self.histogram = c_f.to_device(self.histogram, tensor=hist, dtype=hist.dtype)
        self.histogram = (1.0 - momentum) * self.histogram + momentum * hist

    def _compute_stats(self, pos_dist, neg_dist):
        hist_val = pos_dist - neg_dist
        if self._stats_initialized:
            self._compute_histogram(hist_val, self._momentum)
        else:
            self._compute_histogram(hist_val, 1.0)
            self._stats_initialized = True

    def forward(self, x, labels=None):
        distances = self._compute_distances(x, labels=labels)
        if not self._is_binary:
            pos_dist, neg_dist = distances
            self._compute_stats(pos_dist, neg_dist)
            hist_var = pos_dist - neg_dist
        else:
            pos_dist, neg_dist, pos_dist_b, neg_dist_b = distances
            self._compute_stats(pos_dist_b, neg_dist_b)
            hist_var = pos_dist_b - neg_dist_b
        PDF = self.histogram / self.histogram.sum()
        CDF = PDF.cumsum(0)
        bin_idx = torch.floor((hist_var - self._min_val) / self.bin_width).long()
        weight = CDF[bin_idx]
        loss = (hist_var * weight).mean()
        return loss


def dSoftBinning(D, mid, Delta):
    side1 = (D > mid - Delta).type(D.dtype)
    side2 = (D <= mid).type(D.dtype)
    ind1 = side1 * side2
    side1 = (D > mid).type(D.dtype)
    side2 = (D <= mid + Delta).type(D.dtype)
    ind2 = side1 * side2
    return (ind1 - ind2) / Delta


def softBinning(D, mid, Delta):
    y = 1 - torch.abs(D - mid) / Delta
    return torch.max(torch.tensor([0], dtype=D.dtype), y)


class OriginalImplementationFastAP(torch.autograd.Function):
    """
    FastAP - autograd function definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank",
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019

    NOTE:
        Given a input batch, FastAP does not sample triplets from it as it's not
        a triplet-based method. Therefore, FastAP does not take a Sampler as input.
        Rather, we specify how the input batch is selected.
    """

    @staticmethod
    def forward(ctx, input, target, num_bins):
        """
        Args:
            input:     torch.Tensor(N x embed_dim), embedding matrix
            target:    torch.Tensor(N x 1), class labels
            num_bins:  int, number of bins in distance histogram
        """
        N = target.size()[0]
        assert input.size()[0] == N, "Batch size donesn't match!"
        Y = target.unsqueeze(1)
        Aff = 2 * (Y == Y.t()).type(input.dtype) - 1
        Aff.masked_fill_(torch.eye(N, N).bool(), 0)
        I_pos = (Aff > 0).type(input.dtype)
        I_neg = (Aff < 0).type(input.dtype)
        N_pos = torch.sum(I_pos, 1)
        dist2 = 2 - 2 * torch.mm(input, input.t())
        Delta = torch.tensor(4.0 / num_bins)
        Z = torch.linspace(0.0, 4.0, steps=num_bins + 1)
        L = Z.size()[0]
        h_pos = torch.zeros((N, L), dtype=input.dtype)
        h_neg = torch.zeros((N, L), dtype=input.dtype)
        for l in range(L):
            pulse = softBinning(dist2, Z[l], Delta)
            h_pos[:, l] = torch.sum(pulse * I_pos, 1)
            h_neg[:, l] = torch.sum(pulse * I_neg, 1)
        H_pos = torch.cumsum(h_pos, 1)
        h = h_pos + h_neg
        H = torch.cumsum(h, 1)
        FastAP = h_pos * H_pos / H
        FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
        FastAP = torch.sum(FastAP, 1) / N_pos
        FastAP = FastAP[~torch.isnan(FastAP)]
        loss = 1 - torch.mean(FastAP)
        ctx.save_for_backward(input, target)
        ctx.Z = Z
        ctx.Delta = Delta
        ctx.dist2 = dist2
        ctx.I_pos = I_pos
        ctx.I_neg = I_neg
        ctx.h_pos = h_pos
        ctx.h_neg = h_neg
        ctx.H_pos = H_pos
        ctx.N_pos = N_pos
        ctx.h = h
        ctx.H = H
        ctx.L = torch.tensor(L)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        Z = Variable(ctx.Z, requires_grad=False)
        Delta = Variable(ctx.Delta, requires_grad=False)
        dist2 = Variable(ctx.dist2, requires_grad=False)
        I_pos = Variable(ctx.I_pos, requires_grad=False)
        I_neg = Variable(ctx.I_neg, requires_grad=False)
        h = Variable(ctx.h, requires_grad=False)
        H = Variable(ctx.H, requires_grad=False)
        h_pos = Variable(ctx.h_pos, requires_grad=False)
        h_neg = Variable(ctx.h_neg, requires_grad=False)
        H_pos = Variable(ctx.H_pos, requires_grad=False)
        N_pos = Variable(ctx.N_pos, requires_grad=False)
        L = Z.size()[0]
        H2 = torch.pow(H, 2)
        H_neg = H - H_pos
        LTM1 = torch.tril(torch.ones(L, L), -1)
        tmp1 = h_pos * H_neg / H2
        tmp1[torch.isnan(tmp1)] = 0
        d_AP_h_pos = (H_pos * H + h_pos * H_neg) / H2
        d_AP_h_pos = d_AP_h_pos + torch.mm(tmp1, LTM1)
        d_AP_h_pos = d_AP_h_pos / N_pos.repeat(L, 1).t()
        d_AP_h_pos[torch.isnan(d_AP_h_pos) | torch.isinf(d_AP_h_pos)] = 0
        LTM0 = torch.tril(torch.ones(L, L), 0)
        tmp2 = -h_pos * H_pos / H2
        tmp2[torch.isnan(tmp2)] = 0
        d_AP_h_neg = torch.mm(tmp2, LTM0)
        d_AP_h_neg = d_AP_h_neg / N_pos.repeat(L, 1).t()
        d_AP_h_neg[torch.isnan(d_AP_h_neg) | torch.isinf(d_AP_h_neg)] = 0
        d_AP_x = 0
        for l in range(L):
            dpulse = dSoftBinning(dist2, Z[l], Delta)
            dpulse[torch.isnan(dpulse) | torch.isinf(dpulse)] = 0
            ddp = dpulse * I_pos
            ddn = dpulse * I_neg
            alpha_p = torch.diag(d_AP_h_pos[:, l])
            alpha_n = torch.diag(d_AP_h_neg[:, l])
            Ap = torch.mm(ddp, alpha_p) + torch.mm(alpha_p, ddp)
            An = torch.mm(ddn, alpha_n) + torch.mm(alpha_n, ddn)
            d_AP_x = d_AP_x - torch.mm(input.t(), Ap + An)
        grad_input = -d_AP_x
        return grad_input.t(), None, None


class OriginalImplementationFastAPLoss(torch.nn.Module):
    """
    FastAP - loss layer definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank",
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    """

    def __init__(self, num_bins=10):
        super(OriginalImplementationFastAPLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, batch, labels):
        return OriginalImplementationFastAP.apply(batch, labels, self.num_bins)


class OriginalImplementationHistogramLoss(torch.nn.Module):

    def __init__(self, num_steps, cuda=True):
        super(OriginalImplementationHistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.cuda = cuda
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        if self.cuda:
            self.t = self.t

    def forward(self, features, classes):

        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            inds = c_f.to_device(inds, tensor=s_repeat_floor)
            self.t = c_f.to_device(self.t, tensor=s_repeat_floor)
            indsa = (s_repeat_floor - (self.t - self.step) > -self.eps) & (s_repeat_floor - (self.t - self.step) < self.eps) & inds
            assert indsa.nonzero().size()[0] == size, 'Another number of bins should be used'
            zeros = torch.zeros((1, indsa.size()[1]))
            if self.cuda:
                zeros = zeros
            indsb = torch.cat((indsa, zeros))[1:, :]
            s_repeat_[~(indsb | indsa)] = 0
            self.t = self.t
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step
            return s_repeat_.sum(1) / size
        classes_size = classes.size()[0]
        classes_eq = (classes.repeat(classes_size, 1) == classes.view(-1, 1).repeat(1, classes_size)).data
        dists = torch.mm(features, features.transpose(0, 1))
        assert (dists > 1 + self.eps).sum().item() + (dists < -1 - self.eps).sum().item() == 0, 'L2 normalization should be used'
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
        if self.cuda:
            s_inds = s_inds
        classes_eq = classes_eq
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()
        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1, err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1, err_msg='Not good negative histogram', verbose=True)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).bool()
        if self.cuda:
            histogram_pos_inds = histogram_pos_inds
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)
        return loss


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class OriginalInstanceLoss(nn.Module):

    def __init__(self, gamma=1) ->None:
        super().__init__()
        self.gamma = gamma

    def forward(self, feature, label=None) ->Tensor:
        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature * self.gamma, torch.t(normed_feature))
        if label is None:
            sim_label = torch.arange(sim1.size(0)).detach()
        else:
            _, sim_label = torch.unique(label, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label)
        return loss


def pairwise_similarity(x, y=None):
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
    else:
        y_t = torch.transpose(x, 0, 1)
    dist = torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


class OriginalImplementationManifoldLoss(Module):

    def __init__(self, proxies, alpha, lambdaC=1.0, distance=F.cosine_similarity):
        """
        proxies : P x D , proxy embeddings (one proxy per class randomly initialized, instance of nn.Parameter)
        alpha : float, random walk parameter
        lambdaC : float, regularization weight
        distance : func, distance function to use
        """
        super(OriginalImplementationManifoldLoss, self).__init__()
        self.alpha = alpha
        self.lambdaC = lambdaC
        self.proxy = proxies / proxies.norm(p=2)
        self.nb_proxy = proxies.size(0)
        self.d = distance

    def get_Matrix(self, x):
        """
        x : B x D , feature embeddings
        return
            A: B x B the approximated rank matrix
        """
        W = pairwise_similarity(x)
        W = torch.exp(W / 0.5)
        Y = torch.eye(len(W), dtype=W.dtype, device=x.device)
        W = W - W * Y
        D = torch.diag(torch.pow(torch.sum(W, dim=1), -0.5))
        D[D == float('Inf')] = 0.0
        S = torch.mm(torch.mm(D, W), D)
        dt = S.dtype
        L = torch.inverse(Y.float() - self.alpha * S.float())
        L = L
        A = (1 - self.alpha) * torch.mm(L, Y)
        return A

    def forward(self, fvec, fLvec, fvecs_add=None):
        """
        fvec : B1 x D , current batch of feature embedding
        fLvec : B1 , current batch of GT labels
        fvecs_add : B2 x D , batch of additionnal contextual features to fill the manifold
        """
        fLvec = fLvec.tolist()
        N = len(fLvec)
        if fvecs_add is not None:
            fvec = torch.cat((fvec, self.proxy, fvecs_add), 0)
        else:
            fvec = torch.cat((fvec, self.proxy), 0)
        fvec = fvec / fvec.norm(p=2, dim=1).view(-1, 1)
        A = self.get_Matrix(fvec)
        A_p = A[N:N + self.nb_proxy].clone()
        A = A[:N]
        loss_intrinsic = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)
        loss_context = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)
        for i in range(N):
            loss_neg1_intrinsic = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)
            loss_neg1_context = torch.zeros(1, dtype=fvec.dtype, device=fvec.device)
            dist_pos = self.d(torch.unsqueeze(A[i], 0), torch.unsqueeze(A_p[fLvec[i]], 0))
            for j in range(self.nb_proxy):
                if fLvec[i] != j:
                    val1_context = self.d(torch.unsqueeze(A[i], 0), torch.unsqueeze(A_p[j], 0)) - dist_pos
                    val1_intrinsic = A[i, N + j] - A[i, N + fLvec[i]]
                    if val1_context > 0:
                        loss_neg1_context += torch.exp(val1_context)
                    if val1_intrinsic > 0:
                        loss_neg1_intrinsic += torch.exp(val1_intrinsic)
            loss_intrinsic += torch.log(1.0 + loss_neg1_intrinsic)
            loss_context += torch.log(1.0 + loss_neg1_context)
        loss_intrinsic /= N
        loss_context /= N
        return loss_intrinsic + self.lambdaC * loss_context


class TrustedImplementationP2SActivationLayer(nn.Module):
    """Output layer that produces cos	heta between activation vector x
    and class vector w_j

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors
                (i.e., number of classes)

    Usage example:
      batch_size = 64
      input_dim = 10
      class_num = 5

      l_layer = P2SActivationLayer(input_dim, class_num)
      l_loss = P2SGradLoss()

      data = torch.rand(batch_size, input_dim, requires_grad=True)
      target = (torch.rand(batch_size) * class_num).clamp(0, class_num-1)
      target = target.to(torch.long)

      scores = l_layer(data)
      loss = l_loss(scores, target)

      loss.backward()
    """

    def __init__(self, in_dim, out_dim):
        super(TrustedImplementationP2SActivationLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-05).mul_(100000.0)
        return

    def forward(self, input_feat):
        """
        Compute P2SGrad activation

        input:
        ------
          input_feat: tensor (batch_size, input_dim)

        output:
        -------
          tensor (batch_size, output_dim)

        """
        w = self.weight.renorm(2, 1, 1e-05).mul(100000.0)
        w = c_f.to_device(w, tensor=input_feat, dtype=input_feat.dtype)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        inner_wx = input_feat.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
        return cos_theta


class TrustedImplementationP2SGradLoss(nn.Module):
    """P2SGradLoss() MSE loss between output and target one-hot vectors

    See usage in __doc__ of P2SActivationLayer
    """

    def __init__(self):
        super(TrustedImplementationP2SGradLoss, self).__init__()
        self.m_loss = nn.MSELoss()

    def forward(self, input_score, target):
        """
        input
        -----
          input_score: tensor (batch_size, class_num)
                 cos \\theta given by P2SActivationLayer(input_feat)
          target: tensor (batch_size)
                 target[i] is the target class index of the i-th sample

        output
        ------
          loss: scaler
        """
        with torch.no_grad():
            index = torch.zeros_like(input_score)
            index = c_f.to_device(index, tensor=input_score, dtype=torch.long)
            target = c_f.to_device(target, tensor=input_score, dtype=torch.long)
            index.scatter_(1, target.data.view(-1, 1), 1)
        index = index
        loss = self.m_loss(input_score, index)
        return loss


class OriginalImplementationPNP(torch.nn.Module):

    def __init__(self, b, alpha, anneal, variant, bs, classes):
        super(OriginalImplementationPNP, self).__init__()
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant
        self.batch_size = bs
        self.num_id = classes
        self.samples_per_class = int(bs / classes)
        mask = 1.0 - torch.eye(self.batch_size)
        for i in range(self.num_id):
            mask[i * self.samples_per_class:(i + 1) * self.samples_per_class, i * self.samples_per_class:(i + 1) * self.samples_per_class] = 0
        self.mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)

    def forward(self, batch):
        dtype, device = batch.dtype, batch.device
        self.mask = self.mask.type(dtype)
        sim_all = self.compute_aff(batch)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) * self.mask
        sim_all_rk = torch.sum(sim_sg, dim=-1)
        if self.variant == 'PNP-D_s':
            sim_all_rk = torch.log(1 + sim_all_rk)
        elif self.variant == 'PNP-D_q':
            sim_all_rk = 1 / (1 + sim_all_rk) ** self.alpha
        elif self.variant == 'PNP-I_u':
            sim_all_rk = (1 + sim_all_rk) * torch.log(1 + sim_all_rk)
        elif self.variant == 'PNP-I_b':
            b = self.b
            sim_all_rk = 1 / b ** 2 * (b * sim_all_rk - torch.log(1 + b * sim_all_rk))
        elif self.variant == 'PNP-O':
            pass
        else:
            raise Exception('variantation <{}> not available!'.format(self.variant))
        loss = torch.zeros(1).type(dtype)
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            neg_divide = torch.sum(sim_all_rk[ind * group:(ind + 1) * group, ind * group:(ind + 1) * group] / group)
            loss = loss + neg_divide / self.batch_size
        if self.variant == 'PNP-D_q':
            return 1 - loss
        else:
            return loss

    def sigmoid(self, tensor, temp=1.0):
        """temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def compute_aff(self, x):
        """computes the affinity matrix between an input vector and itself"""
        return torch.mm(x, x.t())


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = torch.FloatTensor(T)
    return T


class OriginalImplementationProxyAnchor(torch.nn.Module):

    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        with_pos_proxies = torch.where(P_one_hot.sum(dim=0) != 0)[0]
        num_valid_proxies = len(with_pos_proxies)
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term
        return loss


class OriginalImplementationSoftTriple(nn.Module):

    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(OriginalImplementationSoftTriple, self).__init__()
        self.la = la
        self.gamma = 1.0 / gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN * K))
        self.weight = torch.zeros(cN * K, cN * K, dtype=torch.bool)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i * K + j, i * K + j + 1:(i + 1) * K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc * self.gamma, dim=2)
        simClass = torch.sum(prob * simStruc, dim=2)
        marginM = torch.zeros(simClass.shape, dtype=input.dtype)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la * (simClass - marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            small_val = c_f.small_val(input.dtype)
            simCenterMasked = torch.clamp(2.0 * simCenter[self.weight], max=2)
            reg = torch.sum(torch.sqrt(2.0 + small_val - simCenterMasked)) / (self.cN * self.K * (self.K - 1.0))
            return lossClassify + self.tau * reg
        else:
            return lossClassify


class ToyMpModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.relu(self.net1(x))
        return self.net2(x)


class TextModel(torch.nn.Module):

    def forward(self, list_of_text):
        return torch.randn(len(list_of_text), 32)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (OriginalInstanceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4])], {})),
    (TextModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

