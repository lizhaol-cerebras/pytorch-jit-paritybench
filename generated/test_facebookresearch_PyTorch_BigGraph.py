
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


from torch.utils import cpp_extension


from collections import defaultdict


import torch


from typing import Sequence


import logging


import random


import time


from functools import partial


from typing import Dict


from typing import Iterable


from typing import List


from typing import Mapping


from typing import NamedTuple


from typing import Tuple


from typing import Union


import numpy as np


import torch.multiprocessing as mp


import torch.nn as nn


from torch.optim import Adagrad


from abc import ABC


from abc import abstractmethod


from typing import Any


from typing import Callable


from typing import Optional


import re


from typing import Generator


from typing import Set


import torch.distributed as td


import torch.multiprocessing


import uuid


from enum import Enum


from itertools import chain


from typing import ClassVar


from typing import Counter


from types import TracebackType


from typing import ContextManager


from typing import Iterator


from typing import Type


from torch import nn as nn


from torch.nn import functional as F


import torch.nn.functional as F


import queue


from functools import lru_cache


from torch.optim import Optimizer


import math


from typing import TypeVar


from typing import MutableMapping


FloatTensorType = torch.Tensor


class AbstractLossFunction(nn.Module, ABC):
    """Calculate weighted loss of scores for positive and negative pairs.

    The inputs are a 1-D tensor of size P containing scores for positive pairs
    of entities (i.e., those among which an edge exists) and a P x N tensor
    containing scores for negative pairs (i.e., where no edge should exist). The
    pairs of entities corresponding to pos_scores[i] and to neg_scores[i,j] have
    at least one endpoint in common. The output is the loss value these scores
    induce. If the method supports weighting (as is the case for the logistic
    loss) all positive scores will be weighted by the same weight and so will
    all the negative ones.
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, pos_scores: 'FloatTensorType', neg_scores: 'FloatTensorType', weight: 'Optional[FloatTensorType]') ->FloatTensorType:
        pass


def match_shape(tensor: 'torch.Tensor', *expected_shape: Union[int, type(Ellipsis)]) ->Union[None, int, Tuple[int, ...]]:
    """Compare the given tensor's shape with what you expect it to be.

    This function serves two goals: it can be used both to assert that the size
    of a tensor (or part of it) is what it should be, and to query for the size
    of the unknown dimensions. The former result can be achieved with:

        >>> match_shape(t, 2, 3, 4)

    which is similar to

        >>> assert t.size() == (2, 3, 4)

    except that it doesn't use an assert (and is thus not stripped when the code
    is optimized) and that it raises a TypeError (instead of an AssertionError)
    with an informative error message. It works with any number of positional
    arguments, including zero. If a dimension's size is not known beforehand
    pass a -1: no check will be performed and the size will be returned.

        >>> t = torch.empty(2, 3, 4)
        >>> match_shape(t, 2, -1, 4)
        3
        >>> match_shape(t, -1, 3, -1)
        (2, 4)

    If the number of dimensions isn't known beforehand, an ellipsis can be used
    as a placeholder for any number of dimensions (including zero). Their sizes
    won't be returned.

        >>> t = torch.empty(2, 3, 4)
        >>> match_shape(t, ..., 3, -1)
        4

    """
    if not all(isinstance(d, int) or d is Ellipsis for d in expected_shape):
        raise RuntimeError("Some arguments aren't ints or ellipses: %s" % (expected_shape,))
    actual_shape = tensor.size()
    error = TypeError("Shape doesn't match: (%s) != (%s)" % (', '.join('%d' % d for d in actual_shape), ', '.join('...' if d is Ellipsis else '*' if d < 0 else '%d' % d for d in expected_shape)))
    if Ellipsis not in expected_shape:
        if len(actual_shape) != len(expected_shape):
            raise error
    else:
        if expected_shape.count(Ellipsis) > 1:
            raise RuntimeError('Two or more ellipses in %s' % (tuple(expected_shape),))
        if len(actual_shape) < len(expected_shape) - 1:
            raise error
        pos = expected_shape.index(Ellipsis)
        expected_shape = expected_shape[:pos] + actual_shape[pos:pos + 1 - len(expected_shape)] + expected_shape[pos + 1:]
    unknown_dims: 'List[int]' = []
    for actual_dim, expected_dim in zip(actual_shape, expected_shape):
        if expected_dim < 0:
            unknown_dims.append(actual_dim)
            continue
        if actual_dim != expected_dim:
            raise error
    if not unknown_dims:
        return None
    if len(unknown_dims) == 1:
        return unknown_dims[0]
    return tuple(unknown_dims)


class LogisticLossFunction(AbstractLossFunction):

    def forward(self, pos_scores: 'FloatTensorType', neg_scores: 'FloatTensorType', weight: 'Optional[FloatTensorType]') ->FloatTensorType:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)
        neg_weight = 1 / num_neg if num_neg > 0 else 0
        if weight is not None:
            match_shape(weight, num_pos)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_scores.new_ones(()).expand(num_pos), reduction='sum', weight=weight)
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_scores.new_zeros(()).expand(num_pos, num_neg), reduction='sum', weight=weight.unsqueeze(-1) if weight is not None else None)
        loss = pos_loss + neg_weight * neg_loss
        return loss


class RankingLossFunction(AbstractLossFunction):

    def __init__(self, *, margin, **kwargs):
        super().__init__()
        self.margin = margin

    def forward(self, pos_scores: 'FloatTensorType', neg_scores: 'FloatTensorType', weight: 'Optional[FloatTensorType]') ->FloatTensorType:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)
        if num_pos == 0 or num_neg == 0:
            return torch.zeros((), device=pos_scores.device, requires_grad=True)
        if weight is not None:
            match_shape(weight, num_pos)
            loss_per_sample = F.margin_ranking_loss(neg_scores, pos_scores.unsqueeze(1), target=pos_scores.new_full((1, 1), -1, dtype=torch.float), margin=self.margin, reduction='none')
            loss = (loss_per_sample * weight.unsqueeze(-1)).sum()
        else:
            loss = F.margin_ranking_loss(neg_scores, pos_scores.unsqueeze(1), target=pos_scores.new_full((1, 1), -1, dtype=torch.float), margin=self.margin, reduction='sum')
        return loss


class SoftmaxLossFunction(AbstractLossFunction):

    def forward(self, pos_scores: 'FloatTensorType', neg_scores: 'FloatTensorType', weight: 'Optional[FloatTensorType]') ->FloatTensorType:
        num_pos = match_shape(pos_scores, -1)
        num_neg = match_shape(neg_scores, num_pos, -1)
        if num_pos == 0 or num_neg == 0:
            return torch.zeros((), device=pos_scores.device, requires_grad=True)
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores.logsumexp(dim=1, keepdim=True)], dim=1)
        if weight is not None:
            loss_per_sample = F.cross_entropy(scores, pos_scores.new_zeros((num_pos,), dtype=torch.long), reduction='none')
            match_shape(weight, num_pos)
            loss_per_sample = loss_per_sample * weight
        else:
            loss_per_sample = F.cross_entropy(scores, pos_scores.new_zeros((num_pos,), dtype=torch.long), reduction='sum')
        return loss_per_sample.sum()


class AbstractEmbedding(nn.Module, ABC):

    @abstractmethod
    def forward(self, input_: 'EntityList') ->FloatTensorType:
        pass

    @abstractmethod
    def get_all_entities(self) ->FloatTensorType:
        pass

    @abstractmethod
    def sample_entities(self, *dims: int) ->FloatTensorType:
        pass


class SimpleEmbedding(AbstractEmbedding):

    def __init__(self, weight: 'nn.Parameter', max_norm: 'Optional[float]'=None):
        super().__init__()
        self.weight: 'nn.Parameter' = weight
        self.max_norm: 'Optional[float]' = max_norm

    def forward(self, input_: 'EntityList') ->FloatTensorType:
        return self.get(input_.to_tensor())

    def get(self, input_: 'LongTensorType') ->FloatTensorType:
        return F.embedding(input_, self.weight, max_norm=self.max_norm, sparse=True)

    def get_all_entities(self) ->FloatTensorType:
        return self.get(torch.arange(self.weight.size(0), dtype=torch.long, device=self.weight.device))

    def sample_entities(self, *dims: int) ->FloatTensorType:
        return self.get(torch.randint(low=0, high=self.weight.size(0), size=dims, device=self.weight.device))


class FeaturizedEmbedding(AbstractEmbedding):

    def __init__(self, weight: 'nn.Parameter', max_norm: 'Optional[float]'=None):
        super().__init__()
        self.weight: 'nn.Parameter' = weight
        self.max_norm: 'Optional[float]' = max_norm

    def forward(self, input_: 'EntityList') ->FloatTensorType:
        return self.get(input_.to_tensor_list())

    def get(self, input_: 'TensorList') ->FloatTensorType:
        if input_.size(0) == 0:
            return torch.empty((0, self.weight.size(1)))
        return F.embedding_bag(input_.data.long(), self.weight, input_.offsets[:-1], max_norm=self.max_norm, sparse=True)

    def get_all_entities(self) ->FloatTensorType:
        raise NotImplementedError('Cannot list all entities for featurized entities')

    def sample_entities(self, *dims: int) ->FloatTensorType:
        raise NotImplementedError('Cannot sample entities for featurized entities.')


class AbstractComparator(nn.Module, ABC):
    """Calculate scores between pairs of given vectors in a certain space.

    The input consists of four tensors each representing a set of vectors: one
    set for each pair of the product between <left-hand side vs right-hand side>
    and <positive vs negative>. Each of these sets is chunked into the same
    number of chunks. The chunks have all the same size within each set, but
    different sets may have chunks of different sizes (except the two positive
    sets, which have chunks of the same size). All the vectors have the same
    number of dimensions. In short, the four tensor have these sizes:

        L+: C x P x D     R+: C x P x D     L-: C x L x D     R-: C x R x D

    The output consists of three tensors:
    - One for the scores between the corresponding pairs in L+ and R+. That is,
      for each chunk on one side, each vector of that chunk is compared only
      with the corresponding vector in the corresponding chunk on the other
      side. Think of it as the "inner" product of the two sides, or a matching.
    - Two for the scores between R+ and L- and between L+ and R-, where for each
      pair of corresponding chunks, all the vectors on one side are compared
      with all the vectors on the other side. Think of it as a per-chunk "outer"
      product, or a complete bipartite graph.
    Hence the sizes of the three output tensors are:

        ⟨L+,R+⟩: C x P     R+ ⊗ L-: C x P x L     L+ ⊗ R-: C x P x R

    Some comparators may need to peform a certain operation in the same way on
    all input vectors (say, normalizing them) before starting to compare them.
    When some vectors are used as both positives and negatives, the operation
    should ideally only be performed once. For that to occur, comparators expose
    a prepare method that the user should call on the vectors before passing
    them to the forward method, taking care of calling it only once on
    duplicated inputs.

    """

    @abstractmethod
    def prepare(self, embs: 'FloatTensorType') ->FloatTensorType:
        pass

    @abstractmethod
    def forward(self, lhs_pos: 'FloatTensorType', rhs_pos: 'FloatTensorType', lhs_neg: 'FloatTensorType', rhs_neg: 'FloatTensorType') ->Tuple[FloatTensorType, FloatTensorType, FloatTensorType]:
        pass


class DotComparator(AbstractComparator):

    def prepare(self, embs: 'FloatTensorType') ->FloatTensorType:
        return embs

    def forward(self, lhs_pos: 'FloatTensorType', rhs_pos: 'FloatTensorType', lhs_neg: 'FloatTensorType', rhs_neg: 'FloatTensorType') ->Tuple[FloatTensorType, FloatTensorType, FloatTensorType]:
        num_chunks, num_pos_per_chunk, dim = match_shape(lhs_pos, -1, -1, -1)
        match_shape(rhs_pos, num_chunks, num_pos_per_chunk, dim)
        match_shape(lhs_neg, num_chunks, -1, dim)
        match_shape(rhs_neg, num_chunks, -1, dim)
        pos_scores = (lhs_pos.float() * rhs_pos.float()).sum(-1)
        lhs_neg_scores = torch.bmm(rhs_pos, lhs_neg.transpose(-1, -2))
        rhs_neg_scores = torch.bmm(lhs_pos, rhs_neg.transpose(-1, -2))
        return pos_scores, lhs_neg_scores, rhs_neg_scores


class CosComparator(AbstractComparator):

    def prepare(self, embs: 'FloatTensorType') ->FloatTensorType:
        norm = embs.norm(2, dim=-1)
        return embs * norm.reciprocal().unsqueeze(-1)

    def forward(self, lhs_pos: 'FloatTensorType', rhs_pos: 'FloatTensorType', lhs_neg: 'FloatTensorType', rhs_neg: 'FloatTensorType') ->Tuple[FloatTensorType, FloatTensorType, FloatTensorType]:
        num_chunks, num_pos_per_chunk, dim = match_shape(lhs_pos, -1, -1, -1)
        match_shape(rhs_pos, num_chunks, num_pos_per_chunk, dim)
        match_shape(lhs_neg, num_chunks, -1, dim)
        match_shape(rhs_neg, num_chunks, -1, dim)
        pos_scores = (lhs_pos.float() * rhs_pos.float()).sum(-1)
        lhs_neg_scores = torch.bmm(rhs_pos, lhs_neg.transpose(-1, -2))
        rhs_neg_scores = torch.bmm(lhs_pos, rhs_neg.transpose(-1, -2))
        return pos_scores, lhs_neg_scores, rhs_neg_scores


def batched_all_pairs_squared_l2_dist(a: 'FloatTensorType', b: 'FloatTensorType') ->FloatTensorType:
    """For each batch, return the squared L2 distance between each pair of vectors

    Let A and B be tensors of shape NxM_AxD and NxM_BxD, each containing N*M_A
    and N*M_B vectors of dimension D grouped in N batches of size M_A and M_B.
    For each batch, for each vector of A and each vector of B, return the sum
    of the squares of the differences of their components.

    """
    num_chunks, num_a, dim = match_shape(a, -1, -1, -1)
    num_b = match_shape(b, num_chunks, -1, dim)
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)
    res = torch.baddbmm(b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2).add_(a_squared.unsqueeze(-1))
    match_shape(res, num_chunks, num_a, num_b)
    return res


def batched_all_pairs_l2_dist(a: 'FloatTensorType', b: 'FloatTensorType') ->FloatTensorType:
    squared_res = batched_all_pairs_squared_l2_dist(a, b)
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res


class L2Comparator(AbstractComparator):

    def prepare(self, embs: 'FloatTensorType') ->FloatTensorType:
        return embs

    def forward(self, lhs_pos: 'FloatTensorType', rhs_pos: 'FloatTensorType', lhs_neg: 'FloatTensorType', rhs_neg: 'FloatTensorType') ->Tuple[FloatTensorType, FloatTensorType, FloatTensorType]:
        num_chunks, num_pos_per_chunk, dim = match_shape(lhs_pos, -1, -1, -1)
        match_shape(rhs_pos, num_chunks, num_pos_per_chunk, dim)
        match_shape(lhs_neg, num_chunks, -1, dim)
        match_shape(rhs_neg, num_chunks, -1, dim)
        pos_scores = (lhs_pos.float() - rhs_pos.float()).pow_(2).sum(dim=-1).clamp_min_(1e-30).sqrt_().neg()
        lhs_neg_scores = batched_all_pairs_l2_dist(rhs_pos, lhs_neg).neg()
        rhs_neg_scores = batched_all_pairs_l2_dist(lhs_pos, rhs_neg).neg()
        return pos_scores, lhs_neg_scores, rhs_neg_scores


class SquaredL2Comparator(AbstractComparator):

    def prepare(self, embs: 'FloatTensorType') ->FloatTensorType:
        return embs

    def forward(self, lhs_pos: 'FloatTensorType', rhs_pos: 'FloatTensorType', lhs_neg: 'FloatTensorType', rhs_neg: 'FloatTensorType') ->Tuple[FloatTensorType, FloatTensorType, FloatTensorType]:
        num_chunks, num_pos_per_chunk, dim = match_shape(lhs_pos, -1, -1, -1)
        match_shape(rhs_pos, num_chunks, num_pos_per_chunk, dim)
        match_shape(lhs_neg, num_chunks, -1, dim)
        match_shape(rhs_neg, num_chunks, -1, dim)
        pos_scores = (lhs_pos.float() - rhs_pos.float()).pow_(2).sum(dim=-1).neg()
        lhs_neg_scores = batched_all_pairs_squared_l2_dist(rhs_pos, lhs_neg).neg()
        rhs_neg_scores = batched_all_pairs_squared_l2_dist(lhs_pos, rhs_neg).neg()
        return pos_scores, lhs_neg_scores, rhs_neg_scores


class BiasedComparator(AbstractComparator):

    def __init__(self, base_comparator):
        super().__init__()
        self.base_comparator = base_comparator

    def prepare(self, embs: 'FloatTensorType') ->FloatTensorType:
        return torch.cat([embs[..., :1], self.base_comparator.prepare(embs[..., 1:])], dim=-1)

    def forward(self, lhs_pos: 'FloatTensorType', rhs_pos: 'FloatTensorType', lhs_neg: 'FloatTensorType', rhs_neg: 'FloatTensorType') ->Tuple[FloatTensorType, FloatTensorType, FloatTensorType]:
        num_chunks, num_pos_per_chunk, dim = match_shape(lhs_pos, -1, -1, -1)
        match_shape(rhs_pos, num_chunks, num_pos_per_chunk, dim)
        match_shape(lhs_neg, num_chunks, -1, dim)
        match_shape(rhs_neg, num_chunks, -1, dim)
        pos_scores, lhs_neg_scores, rhs_neg_scores = self.base_comparator.forward(lhs_pos[..., 1:], rhs_pos[..., 1:], lhs_neg[..., 1:], rhs_neg[..., 1:])
        lhs_pos_bias = lhs_pos[..., 0]
        rhs_pos_bias = rhs_pos[..., 0]
        pos_scores += lhs_pos_bias
        pos_scores += rhs_pos_bias
        lhs_neg_scores += rhs_pos_bias.unsqueeze(-1)
        lhs_neg_scores += lhs_neg[..., 0].unsqueeze(-2)
        rhs_neg_scores += lhs_pos_bias.unsqueeze(-1)
        rhs_neg_scores += rhs_neg[..., 0].unsqueeze(-2)
        return pos_scores, lhs_neg_scores, rhs_neg_scores


LongTensorType = torch.Tensor


Mask = List[Tuple[Union[int, slice, Sequence[int], LongTensorType], ...]]


class Negatives(Enum):
    NONE = 'none'
    UNIFORM = 'uniform'
    BATCH_UNIFORM = 'batch_uniform'
    ALL = 'all'


class Scores(NamedTuple):
    lhs_pos: 'FloatTensorType'
    rhs_pos: 'FloatTensorType'
    lhs_neg: 'FloatTensorType'
    rhs_neg: 'FloatTensorType'


T = TypeVar('T')


class Side(Enum):
    LHS = 0
    RHS = 1

    def pick(self, lhs: 'T', rhs: 'T') ->T:
        if self is Side.LHS:
            return lhs
        elif self is Side.RHS:
            return rhs
        else:
            raise NotImplementedError('Unknown side: %s' % self)


def ceil_of_ratio(num: 'int', den: 'int') ->int:
    return (num - 1) // den + 1


logger = logging.getLogger('torchbiggraph')


class MultiRelationEmbedder(nn.Module):
    """
    A multi-relation embedding model.

    Graph embedding on multiple relations over multiple entity types. Each
    relation consists of a lhs and rhs entity type, and optionally a relation
    operator (which is a learned multiplicative vector - see e.g.
    https://arxiv.org/abs/1510.04935)

    The model includes the logic for training using a ranking loss over a mixture
    of negatives sampled from the batch and uniformly from the entities. An
    optimization is used for negative sampling, where each batch is divided into
    sub-batches of size num_batch_negs, which are used as negative samples against
    each other. Each of these sub-batches also receives num_uniform_negs (common)
    negative samples sampled uniformly from the entities of the lhs and rhs types.
    """
    EMB_PREFIX = 'emb_'

    def __init__(self, default_dim: 'int', relations: 'List[RelationSchema]', entities: 'Dict[str, EntitySchema]', num_batch_negs: 'int', num_uniform_negs: 'int', disable_lhs_negs: 'bool', disable_rhs_negs: 'bool', lhs_operators: 'Sequence[Optional[Union[AbstractOperator, AbstractDynamicOperator]]]', rhs_operators: 'Sequence[Optional[Union[AbstractOperator, AbstractDynamicOperator]]]', comparator: 'AbstractComparator', regularizer: 'AbstractRegularizer', global_emb: 'bool'=False, max_norm: 'Optional[float]'=None, num_dynamic_rels: 'int'=0, half_precision: 'bool'=False) ->None:
        super().__init__()
        self.relations: 'List[RelationSchema]' = relations
        self.entities: 'Dict[str, EntitySchema]' = entities
        self.num_dynamic_rels: 'int' = num_dynamic_rels
        if num_dynamic_rels > 0:
            assert len(relations) == 1
        self.lhs_operators: 'nn.ModuleList' = nn.ModuleList(lhs_operators)
        self.rhs_operators: 'nn.ModuleList' = nn.ModuleList(rhs_operators)
        self.num_batch_negs: 'int' = num_batch_negs
        self.num_uniform_negs: 'int' = num_uniform_negs
        self.disable_lhs_negs = disable_lhs_negs
        self.disable_rhs_negs = disable_rhs_negs
        self.comparator = comparator
        self.lhs_embs: 'nn.ParameterDict' = nn.ModuleDict()
        self.rhs_embs: 'nn.ParameterDict' = nn.ModuleDict()
        if global_emb:
            global_embs = nn.ParameterDict()
            for entity, entity_schema in entities.items():
                global_embs[self.EMB_PREFIX + entity] = nn.Parameter(torch.zeros((entity_schema.dimension or default_dim,)))
            self.global_embs = global_embs
        else:
            self.global_embs: 'Optional[nn.ParameterDict]' = None
        self.max_norm: 'Optional[float]' = max_norm
        self.half_precision = half_precision
        self.regularizer: 'Optional[AbstractRegularizer]' = regularizer

    def set_embeddings(self, entity: 'str', side: 'Side', weights: 'nn.Parameter') ->None:
        if self.entities[entity].featurized:
            emb = FeaturizedEmbedding(weights, max_norm=self.max_norm)
        else:
            emb = SimpleEmbedding(weights, max_norm=self.max_norm)
        side.pick(self.lhs_embs, self.rhs_embs)[self.EMB_PREFIX + entity] = emb

    def set_all_embeddings(self, holder: 'EmbeddingHolder', bucket: 'Bucket') ->None:
        for entity in holder.lhs_unpartitioned_types:
            self.set_embeddings(entity, Side.LHS, holder.unpartitioned_embeddings[entity])
        for entity in holder.rhs_unpartitioned_types:
            self.set_embeddings(entity, Side.RHS, holder.unpartitioned_embeddings[entity])
        for entity in holder.lhs_partitioned_types:
            self.set_embeddings(entity, Side.LHS, holder.partitioned_embeddings[entity, bucket.lhs])
        for entity in holder.rhs_partitioned_types:
            self.set_embeddings(entity, Side.RHS, holder.partitioned_embeddings[entity, bucket.rhs])

    def clear_all_embeddings(self) ->None:
        self.lhs_embs.clear()
        self.rhs_embs.clear()

    def adjust_embs(self, embs: 'FloatTensorType', rel: 'Union[int, LongTensorType]', entity_type: 'str', operator: 'Union[None, AbstractOperator, AbstractDynamicOperator]') ->FloatTensorType:
        if self.global_embs is not None:
            if not isinstance(rel, int):
                raise RuntimeError('Cannot have global embs with dynamic rels')
            embs += self.global_embs[self.EMB_PREFIX + entity_type]
        if operator is not None:
            if self.num_dynamic_rels > 0:
                embs = operator(embs, rel)
            else:
                embs = operator(embs)
        embs = self.comparator.prepare(embs)
        if self.half_precision and embs.is_cuda:
            embs = embs.half()
        return embs

    def prepare_negatives(self, pos_input: 'EntityList', pos_embs: 'FloatTensorType', module: 'AbstractEmbedding', type_: 'Negatives', num_uniform_neg: 'int', rel: 'Union[int, LongTensorType]', entity_type: 'str', operator: 'Union[None, AbstractOperator, AbstractDynamicOperator]') ->Tuple[FloatTensorType, Mask]:
        """Given some chunked positives, set up chunks of negatives.

        This function operates on one side (left-hand or right-hand) at a time.
        It takes all the information about the positives on that side (the
        original input value, the corresponding embeddings, and the module used
        to convert one to the other). It then produces negatives for that side
        according to the specified mode. The positive embeddings come in in
        chunked form and the negatives are produced within each of these chunks.
        The negatives can be either none, or the positives from the same chunk,
        or all the possible entities. In the second mode, uniformly-sampled
        entities can also be appended to the per-chunk negatives (each chunk
        having a different sample). This function returns both the chunked
        embeddings of the negatives and a mask of the same size as the chunked
        positives-vs-negatives scores, whose non-zero elements correspond to the
        scores that must be ignored.

        """
        num_pos = len(pos_input)
        num_chunks, chunk_size, dim = match_shape(pos_embs, -1, -1, -1)
        last_chunk_size = num_pos - (num_chunks - 1) * chunk_size
        ignore_mask: 'Mask' = []
        if type_ is Negatives.NONE:
            neg_embs = pos_embs.new_empty((num_chunks, 0, dim))
        elif type_ is Negatives.UNIFORM:
            uniform_neg_embs = module.sample_entities(num_chunks, num_uniform_neg)
            neg_embs = self.adjust_embs(uniform_neg_embs, rel, entity_type, operator)
        elif type_ is Negatives.BATCH_UNIFORM:
            neg_embs = pos_embs
            if num_uniform_neg > 0:
                try:
                    uniform_neg_embs = module.sample_entities(num_chunks, num_uniform_neg)
                except NotImplementedError:
                    pass
                else:
                    neg_embs = torch.cat([pos_embs, self.adjust_embs(uniform_neg_embs, rel, entity_type, operator)], dim=1)
            chunk_indices = torch.arange(chunk_size, dtype=torch.long, device=pos_embs.device)
            last_chunk_indices = chunk_indices[:last_chunk_size]
            ignore_mask.append((slice(num_chunks - 1), chunk_indices, chunk_indices))
            ignore_mask.append((-1, last_chunk_indices, last_chunk_indices))
            ignore_mask.append((-1, slice(last_chunk_size), slice(last_chunk_size, chunk_size)))
        elif type_ is Negatives.ALL:
            pos_input_ten = pos_input.to_tensor()
            neg_embs = self.adjust_embs(module.get_all_entities().expand(num_chunks, -1, dim), rel, entity_type, operator)
            if num_uniform_neg > 0:
                logger.warning('Adding uniform negatives makes no sense when already using all negatives')
            chunk_indices = torch.arange(chunk_size, dtype=torch.long, device=pos_embs.device)
            last_chunk_indices = chunk_indices[:last_chunk_size]
            ignore_mask.append((torch.arange(num_chunks - 1, dtype=torch.long, device=pos_embs.device).unsqueeze(1), chunk_indices.unsqueeze(0), pos_input_ten[:-last_chunk_size].view(num_chunks - 1, chunk_size)))
            ignore_mask.append((-1, last_chunk_indices, pos_input_ten[-last_chunk_size:]))
        else:
            raise NotImplementedError('Unknown negative type %s' % type_)
        return neg_embs, ignore_mask

    def forward(self, edges: 'EdgeList') ->Scores:
        num_pos = len(edges)
        chunk_size: 'int'
        lhs_negatives: 'Negatives'
        lhs_num_uniform_negs: 'int'
        rhs_negatives: 'Negatives'
        rhs_num_uniform_negs: 'int'
        if self.num_dynamic_rels > 0:
            if edges.has_scalar_relation_type():
                raise TypeError('Need relation for each positive pair')
            relation_idx = 0
        else:
            if not edges.has_scalar_relation_type():
                raise TypeError('All positive pairs must come from the same relation')
            relation_idx = edges.get_relation_type_as_scalar()
        relation = self.relations[relation_idx]
        lhs_module: 'AbstractEmbedding' = self.lhs_embs[self.EMB_PREFIX + relation.lhs]
        rhs_module: 'AbstractEmbedding' = self.rhs_embs[self.EMB_PREFIX + relation.rhs]
        lhs_pos: 'FloatTensorType' = lhs_module(edges.lhs)
        rhs_pos: 'FloatTensorType' = rhs_module(edges.rhs)
        if relation.all_negs:
            chunk_size = num_pos
            negative_sampling_method = Negatives.ALL
        elif self.num_batch_negs == 0:
            chunk_size = min(self.num_uniform_negs, num_pos)
            negative_sampling_method = Negatives.UNIFORM
        else:
            chunk_size = min(self.num_batch_negs, num_pos)
            negative_sampling_method = Negatives.BATCH_UNIFORM
        lhs_negative_sampling_method = negative_sampling_method
        rhs_negative_sampling_method = negative_sampling_method
        if self.disable_lhs_negs:
            lhs_negative_sampling_method = Negatives.NONE
        if self.disable_rhs_negs:
            rhs_negative_sampling_method = Negatives.NONE
        if self.num_dynamic_rels == 0:
            if self.lhs_operators[relation_idx] is not None:
                raise RuntimeError('In non-dynamic relation mode there should be only a right-hand side operator')
            pos_scores, lhs_neg_scores, rhs_neg_scores, reg = self.forward_direction_agnostic(edges.lhs, edges.rhs, edges.get_relation_type(), relation.lhs, relation.rhs, None, self.rhs_operators[relation_idx], lhs_module, rhs_module, lhs_pos, rhs_pos, chunk_size, lhs_negative_sampling_method, rhs_negative_sampling_method)
            lhs_pos_scores = rhs_pos_scores = pos_scores
        else:
            lhs_pos_scores, lhs_neg_scores, _, l_reg = self.forward_direction_agnostic(edges.lhs, edges.rhs, edges.get_relation_type(), relation.lhs, relation.rhs, None, self.rhs_operators[relation_idx], lhs_module, rhs_module, lhs_pos, rhs_pos, chunk_size, lhs_negative_sampling_method, Negatives.NONE)
            rhs_pos_scores, rhs_neg_scores, _, r_reg = self.forward_direction_agnostic(edges.rhs, edges.lhs, edges.get_relation_type(), relation.rhs, relation.lhs, None, self.lhs_operators[relation_idx], rhs_module, lhs_module, rhs_pos, lhs_pos, chunk_size, rhs_negative_sampling_method, Negatives.NONE)
            if r_reg is None or l_reg is None:
                reg = None
            else:
                reg = l_reg + r_reg
        return Scores(lhs_pos_scores, rhs_pos_scores, lhs_neg_scores, rhs_neg_scores), reg

    def forward_direction_agnostic(self, src: 'EntityList', dst: 'EntityList', rel: 'Union[int, LongTensorType]', src_entity_type: 'str', dst_entity_type: 'str', src_operator: 'Union[None, AbstractOperator, AbstractDynamicOperator]', dst_operator: 'Union[None, AbstractOperator, AbstractDynamicOperator]', src_module: 'AbstractEmbedding', dst_module: 'AbstractEmbedding', src_pos: 'FloatTensorType', dst_pos: 'FloatTensorType', chunk_size: 'int', src_negative_sampling_method: 'Negatives', dst_negative_sampling_method: 'Negatives'):
        num_pos = len(src)
        assert len(dst) == num_pos
        src_pos = self.adjust_embs(src_pos, rel, src_entity_type, src_operator)
        dst_pos = self.adjust_embs(dst_pos, rel, dst_entity_type, dst_operator)
        num_chunks = ceil_of_ratio(num_pos, chunk_size)
        src_dim = src_pos.size(-1)
        dst_dim = dst_pos.size(-1)
        if num_pos < num_chunks * chunk_size:
            src_padding = src_pos.new_zeros(()).expand((num_chunks * chunk_size - num_pos, src_dim))
            src_pos = torch.cat((src_pos, src_padding), dim=0)
            dst_padding = dst_pos.new_zeros(()).expand((num_chunks * chunk_size - num_pos, dst_dim))
            dst_pos = torch.cat((dst_pos, dst_padding), dim=0)
        src_pos = src_pos.view((num_chunks, chunk_size, src_dim))
        dst_pos = dst_pos.view((num_chunks, chunk_size, dst_dim))
        src_neg, src_ignore_mask = self.prepare_negatives(src, src_pos, src_module, src_negative_sampling_method, self.num_uniform_negs, rel, src_entity_type, src_operator)
        dst_neg, dst_ignore_mask = self.prepare_negatives(dst, dst_pos, dst_module, dst_negative_sampling_method, self.num_uniform_negs, rel, dst_entity_type, dst_operator)
        pos_scores, src_neg_scores, dst_neg_scores = self.comparator(src_pos, dst_pos, src_neg, dst_neg)
        pos_scores = pos_scores.float()
        src_neg_scores = src_neg_scores.float()
        dst_neg_scores = dst_neg_scores.float()
        for ignore_mask in src_ignore_mask:
            src_neg_scores[ignore_mask] = -1000000000.0
        for ignore_mask in dst_ignore_mask:
            dst_neg_scores[ignore_mask] = -1000000000.0
        pos_scores = pos_scores.flatten(0, 1)[:num_pos]
        src_neg_scores = src_neg_scores.flatten(0, 1)[:num_pos]
        dst_neg_scores = dst_neg_scores.flatten(0, 1)[:num_pos]
        reg = None
        if self.regularizer is not None:
            assert (src_operator is None) != (dst_operator is None), 'Exactly one of src or dst operator should be None'
            operator = src_operator if src_operator is not None else dst_operator
            if self.num_dynamic_rels > 0:
                reg = self.regularizer.forward_dynamic(src_pos, dst_pos, operator, rel)
            else:
                reg = self.regularizer.forward(src_pos, dst_pos, operator)
        return pos_scores, src_neg_scores, dst_neg_scores, reg


class AbstractOperator(nn.Module, ABC):
    """Perform the same operation on many vectors.

    Given a tensor containing a set of vectors, perform the same operation on
    all of them, with a common set of parameters. The dimension of these vectors
    will be given at initialization (so that any parameter can be initialized).
    The input will be a tensor with at least one dimension. The last dimension
    will contain the vectors. The output is a tensor that will have the same
    size as the input.

    """

    def __init__(self, dim: 'int'):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, embeddings: 'FloatTensorType') ->FloatTensorType:
        pass

    def get_operator_params_for_reg(self) ->Optional[FloatTensorType]:
        raise NotImplementedError('Regularizer not implemented for this operator')

    def prepare_embs_for_reg(self, embs: 'FloatTensorType') ->FloatTensorType:
        return embs.abs()


class IdentityOperator(AbstractOperator):

    def forward(self, embeddings: 'FloatTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return embeddings

    def get_operator_params_for_reg(self) ->Optional[FloatTensorType]:
        return None


class DiagonalOperator(AbstractOperator):

    def __init__(self, dim: 'int'):
        super().__init__(dim)
        self.diagonal = nn.Parameter(torch.ones((self.dim,)))

    def forward(self, embeddings: 'FloatTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return self.diagonal * embeddings

    def get_operator_params_for_reg(self) ->Optional[FloatTensorType]:
        return self.diagonal.abs()


class TranslationOperator(AbstractOperator):

    def __init__(self, dim: 'int'):
        super().__init__(dim)
        self.translation = nn.Parameter(torch.zeros((self.dim,)))

    def forward(self, embeddings: 'FloatTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return embeddings + self.translation

    def get_operator_params_for_reg(self) ->Optional[FloatTensorType]:
        return self.translation.abs()


class LinearOperator(AbstractOperator):

    def __init__(self, dim: 'int'):
        super().__init__(dim)
        self.linear_transformation = nn.Parameter(torch.eye(self.dim))

    def forward(self, embeddings: 'FloatTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return torch.matmul(self.linear_transformation, embeddings.unsqueeze(-1)).squeeze(-1)


class AffineOperator(AbstractOperator):

    def __init__(self, dim: 'int'):
        super().__init__(dim)
        self.linear_transformation = nn.Parameter(torch.eye(self.dim))
        self.translation = nn.Parameter(torch.zeros((self.dim,)))

    def forward(self, embeddings: 'FloatTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return torch.matmul(self.linear_transformation, embeddings.unsqueeze(-1)).squeeze(-1) + self.translation

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        param_key = '%slinear_transformation' % prefix
        old_param_key = '%srotation' % prefix
        if old_param_key in state_dict:
            state_dict[param_key] = state_dict.pop(old_param_key).transpose(-1, -2).contiguous()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ComplexDiagonalOperator(AbstractOperator):

    def __init__(self, dim: 'int'):
        super().__init__(dim)
        if dim % 2 != 0:
            raise ValueError('Need even dimension as 1st half is real and 2nd half is imaginary coordinates')
        self.real = nn.Parameter(torch.ones((self.dim // 2,)))
        self.imag = nn.Parameter(torch.zeros((self.dim // 2,)))

    def forward(self, embeddings: 'FloatTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        real_a = embeddings[..., :self.dim // 2]
        imag_a = embeddings[..., self.dim // 2:]
        real_b = self.real
        imag_b = self.imag
        prod = torch.empty_like(embeddings)
        prod[..., :self.dim // 2] = real_a * real_b - imag_a * imag_b
        prod[..., self.dim // 2:] = real_a * imag_b + imag_a * real_b
        return prod

    def get_operator_params_for_reg(self) ->Optional[FloatTensorType]:
        return torch.sqrt(self.real ** 2 + self.imag ** 2)

    def prepare_embs_for_reg(self, embs: 'FloatTensorType') ->FloatTensorType:
        assert embs.shape[-1] == self.dim
        real, imag = embs[..., :self.dim // 2], embs[..., self.dim // 2:]
        return torch.sqrt(real ** 2 + imag ** 2)


class AbstractDynamicOperator(nn.Module, ABC):
    """Perform different operations on many vectors.

    The inputs are a tensor containing a set of vectors and another tensor
    specifying, for each vector, which operation to apply to it. The output has
    the same size as the first input and contains the outputs of the operations
    applied to the input vectors. The different operations are identified by
    integers in a [0, N) range. They are all of the same type (say, translation)
    but each one has its own set of parameters. The dimension of the vectors and
    the total number of operations that need to be supported are provided at
    initialization. The first tensor can have any number of dimensions (>= 1).

    """

    def __init__(self, dim: 'int', num_operations: 'int'):
        super().__init__()
        self.dim = dim
        self.num_operations = num_operations

    @abstractmethod
    def forward(self, embeddings: 'FloatTensorType', operator_idxs: 'LongTensorType') ->FloatTensorType:
        pass

    def get_operator_params_for_reg(self, operator_idxs: 'LongTensorType') ->Optional[FloatTensorType]:
        raise NotImplementedError('Regularizer not implemented for this operator')

    def prepare_embs_for_reg(self, embs: 'FloatTensorType') ->FloatTensorType:
        return embs.abs()


class IdentityDynamicOperator(AbstractDynamicOperator):

    def forward(self, embeddings: 'FloatTensorType', operator_idxs: 'LongTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return embeddings

    def get_operator_params_for_reg(self, operator_idxs: 'LongTensorType') ->Optional[FloatTensorType]:
        return None


class DiagonalDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: 'int', num_operations: 'int'):
        super().__init__(dim, num_operations)
        self.diagonals = nn.Parameter(torch.ones((self.num_operations, self.dim)))

    def forward(self, embeddings: 'FloatTensorType', operator_idxs: 'LongTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return self.diagonals[operator_idxs] * embeddings

    def get_operator_params_for_reg(self, operator_idxs: 'LongTensorType') ->Optional[FloatTensorType]:
        return self.diagonals[operator_idxs].abs()


class TranslationDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: 'int', num_operations: 'int'):
        super().__init__(dim, num_operations)
        self.translations = nn.Parameter(torch.zeros((self.num_operations, self.dim)))

    def forward(self, embeddings: 'FloatTensorType', operator_idxs: 'LongTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return embeddings + self.translations[operator_idxs]

    def get_operator_params_for_reg(self, operator_idxs: 'LongTensorType') ->Optional[FloatTensorType]:
        return self.translations[operator_idxs].abs()


class LinearDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: 'int', num_operations: 'int'):
        super().__init__(dim, num_operations)
        self.linear_transformations = nn.Parameter(torch.diag_embed(torch.ones(()).expand(num_operations, dim)))

    def forward(self, embeddings: 'FloatTensorType', operator_idxs: 'LongTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return torch.matmul(self.linear_transformations[operator_idxs], embeddings.unsqueeze(-1)).squeeze(-1)


class AffineDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: 'int', num_operations: 'int'):
        super().__init__(dim, num_operations)
        self.linear_transformations = nn.Parameter(torch.diag_embed(torch.ones(()).expand(num_operations, dim)))
        self.translations = nn.Parameter(torch.zeros((self.num_operations, self.dim)))

    def forward(self, embeddings: 'FloatTensorType', operator_idxs: 'LongTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return torch.matmul(self.linear_transformations[operator_idxs], embeddings.unsqueeze(-1)).squeeze(-1) + self.translations[operator_idxs]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        param_key = '%slinear_transformations' % prefix
        old_param_key = '%srotations' % prefix
        if old_param_key in state_dict:
            state_dict[param_key] = state_dict.pop(old_param_key).transpose(-1, -2).contiguous()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ComplexDiagonalDynamicOperator(AbstractDynamicOperator):

    def __init__(self, dim: 'int', num_operations: 'int'):
        super().__init__(dim, num_operations)
        if dim % 2 != 0:
            raise ValueError('Need even dimension as 1st half is real and 2nd half is imaginary coordinates')
        self.real = nn.Parameter(torch.ones((self.num_operations, self.dim // 2)))
        self.imag = nn.Parameter(torch.zeros((self.num_operations, self.dim // 2)))

    def forward(self, embeddings: 'FloatTensorType', operator_idxs: 'LongTensorType') ->FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        real_a = embeddings[..., :self.dim // 2]
        imag_a = embeddings[..., self.dim // 2:]
        real_b = self.real[operator_idxs]
        imag_b = self.imag[operator_idxs]
        prod = torch.empty_like(embeddings)
        prod[..., :self.dim // 2] = real_a * real_b - imag_a * imag_b
        prod[..., self.dim // 2:] = real_a * imag_b + imag_a * real_b
        return prod

    def get_operator_params_for_reg(self, operator_idxs) ->Optional[FloatTensorType]:
        return torch.sqrt(self.real[operator_idxs] ** 2 + self.imag[operator_idxs] ** 2)

    def prepare_embs_for_reg(self, embs: 'FloatTensorType') ->FloatTensorType:
        assert embs.shape[-1] == self.dim
        real, imag = embs[..., :self.dim // 2], embs[..., self.dim // 2:]
        return torch.sqrt(real ** 2 + imag ** 2)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AffineOperator,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ComplexDiagonalOperator,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (CosComparator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DiagonalOperator,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DotComparator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (IdentityOperator,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (L2Comparator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (LinearOperator,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SquaredL2Comparator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (TranslationOperator,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

