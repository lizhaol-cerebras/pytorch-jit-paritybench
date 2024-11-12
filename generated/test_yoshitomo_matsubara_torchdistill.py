
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


import math


from torch import nn


import logging


import time


import numpy as np


import pandas as pd


import torch


from torch.backends import cudnn


from torch.nn import functional


from torch import distributed as dist


from torch.nn import DataParallel


from torch.nn.parallel import DistributedDataParallel


from torch.utils.data._utils.collate import default_collate


from torchvision.models.detection.keypoint_rcnn import KeypointRCNN


from torchvision.models.detection.mask_rcnn import MaskRCNN


import copy


import random


import torch.utils.data


from torchvision.datasets import CocoDetection


from torchvision.transforms import functional


from collections import defaultdict


import torch.distributed as dist


import torchvision


from torch.utils.data.sampler import BatchSampler


from torch.utils.data.sampler import Sampler


from torch.utils.model_zoo import tqdm


from torchvision import transforms as T


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import Resize


from torchvision.transforms import functional as F


from torchvision.transforms.functional import InterpolationMode


from torchvision import models


from collections import OrderedDict


from torch.nn import Sequential


from torch.nn import ModuleList


from torch.nn import Module


from torch.nn import Parameter


from collections import namedtuple


from collections import abc


from torch.nn.parallel.scatter_gather import gather


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


from types import BuiltinFunctionType


from types import BuiltinMethodType


from types import FunctionType


from torch.utils.data import DataLoader


from torch.utils.data import random_split


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import Dataset


from torch.nn.functional import adaptive_avg_pool2d


from torch.nn.functional import adaptive_max_pool2d


from torch.nn.functional import normalize


from torch.nn.functional import cosine_similarity


from collections import deque


from logging import FileHandler


from logging import Formatter


from typing import Any


from typing import Tuple


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from torchvision.models.densenet import _DenseBlock


from torchvision.models.densenet import _Transition


from typing import Type


from typing import Callable


from typing import Union


from typing import List


from typing import Optional


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import conv1x1


from torchvision.models import densenet169


from torchvision.models import densenet201


from torchvision.models import inception_v3


from torchvision.models import resnet50


from torchvision.models import resnet101


from torchvision.models import resnet152


from torch.hub import load_state_dict_from_url


from torchvision.models.detection.faster_rcnn import FasterRCNN


from torchvision.ops import MultiScaleRoIAlign


from torchvision.models import resnet


from torchvision.models.detection.backbone_utils import BackboneWithFPN


from torchvision.ops import misc as misc_nn_ops


from torch.nn import SyncBatchNorm


from torch.jit.annotations import Tuple


from torch.jit.annotations import List


from torch.optim.lr_scheduler import LambdaLR


class KDLoss4Transformer(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, temperature, alpha=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha

    def compute_soft_loss(self, student_logits, teacher_logits):
        return super().forward(torch.log_softmax(student_logits / self.temperature, dim=1), torch.softmax(teacher_logits / self.temperature, dim=1))

    def compute_hard_loss(self, logits, positions, ignored_index):
        return functional.cross_entropy(logits, positions, reduction=self.cel_reduction, ignore_index=ignored_index)

    def forward(self, student_output, teacher_output, targets=None, *args, **kwargs):
        soft_loss = self.compute_soft_loss(student_output.logits, teacher_output.logits)
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = student_output.loss
        return self.alpha * hard_loss + self.beta * self.temperature ** 2 * soft_loss


MIDDLE_LEVEL_LOSS_DICT = dict()


LOSS_WRAPPER_DICT = dict()


def get_loss_wrapper(mid_level_loss, criterion_wrapper_config):
    """
    Gets a registered loss wrapper module.

    :param mid_level_loss: middle-level loss module.
    :type mid_level_loss: nn.Module
    :param criterion_wrapper_config: loss wrapper configuration to identify and instantiate the registered loss wrapper class.
    :type criterion_wrapper_config: dict
    :return: registered loss wrapper class or function to instantiate it.
    :rtype: nn.Module
    """
    wrapper_key = criterion_wrapper_config['key']
    args = criterion_wrapper_config.get('args', None)
    kwargs = criterion_wrapper_config.get('kwargs', None)
    if args is None:
        args = list()
    if kwargs is None:
        kwargs = dict()
    if wrapper_key in LOSS_WRAPPER_DICT:
        return LOSS_WRAPPER_DICT[wrapper_key](mid_level_loss, *args, **kwargs)
    raise ValueError('No loss wrapper `{}` registered'.format(wrapper_key))


LOW_LEVEL_LOSS_DICT = dict()


def get_low_level_loss(key, **kwargs):
    """
    Gets a registered (low-level) loss module.

    :param key: unique key to identify the registered loss class/function.
    :type key: str
    :return: registered loss class or function to instantiate it.
    :rtype: nn.Module
    """
    if key in LOSS_DICT:
        return LOSS_DICT[key](**kwargs)
    elif key in LOW_LEVEL_LOSS_DICT:
        return LOW_LEVEL_LOSS_DICT[key](**kwargs)
    raise ValueError('No loss `{}` registered'.format(key))


def get_mid_level_loss(mid_level_criterion_config, criterion_wrapper_config=None):
    """
    Gets a registered middle-level loss module.

    :param mid_level_criterion_config: middle-level loss configuration to identify and instantiate the registered middle-level loss class.
    :type mid_level_criterion_config: dict
    :param criterion_wrapper_config: middle-level loss configuration to identify and instantiate the registered middle-level loss class.
    :type criterion_wrapper_config: dict
    :return: registered middle-level loss class or function to instantiate it.
    :rtype: nn.Module
    """
    loss_key = mid_level_criterion_config['key']
    mid_level_loss = MIDDLE_LEVEL_LOSS_DICT[loss_key](**mid_level_criterion_config['kwargs']) if loss_key in MIDDLE_LEVEL_LOSS_DICT else get_low_level_loss(loss_key, **mid_level_criterion_config['kwargs'])
    if criterion_wrapper_config is None or len(criterion_wrapper_config) == 0:
        return mid_level_loss
    return get_loss_wrapper(mid_level_loss, criterion_wrapper_config)


class AbstractLoss(nn.Module):
    """
    An abstract loss module.

    :meth:`forward` and :meth:`__str__` should be overridden by all subclasses.

    :param sub_terms: loss module configurations.
    :type sub_terms: dict or None

    .. code-block:: YAML
       :caption: An example yaml of ``sub_terms``

        sub_terms:
          ce:
            criterion:
              key: 'CrossEntropyLoss'
              kwargs:
                reduction: 'mean'
            criterion_wrapper:
              key: 'SimpleLossWrapper'
              kwargs:
                input:
                  is_from_teacher: False
                  module_path: '.'
                  io: 'output'
                target:
                  uses_label: True
            weight: 1.0
    """

    def __init__(self, sub_terms=None, **kwargs):
        super().__init__()
        term_dict = dict()
        if sub_terms is not None:
            for loss_name, loss_config in sub_terms.items():
                sub_criterion_or_config = loss_config['criterion']
                sub_criterion = sub_criterion_or_config if isinstance(sub_criterion_or_config, nn.Module) else get_mid_level_loss(sub_criterion_or_config, loss_config.get('criterion_wrapper', None))
                term_dict[loss_name] = sub_criterion, loss_config['weight']
        self.term_dict = term_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function is not implemented')

    def __str__(self):
        raise NotImplementedError('forward function is not implemented')


class WeightedSumLoss(AbstractLoss):
    """
    A weighted sum (linear combination) of mid-/low-level loss modules.

    If ``model_term`` contains a numerical value with ``weight`` key, it will be a multiplier :math:`W_{model}`
    for the sum of model-driven loss values :math:`\\sum_{i} L_{model, i}`.

    .. math:: L_{total} = W_{model} \\cdot (\\sum_{i} L_{model, i}) + \\sum_{k} W_{sub, k} \\cdot L_{sub, k}

    :param model_term: model-driven loss module configurations.
    :type model_term: dict or None
    :param sub_terms: loss module configurations.
    :type sub_terms: dict or None
    """

    def __init__(self, model_term=None, sub_terms=None, **kwargs):
        super().__init__(sub_terms=sub_terms, **kwargs)
        if model_term is None:
            model_term = dict()
        self.model_loss_factor = model_term.get('weight', None)

    def forward(self, io_dict, model_loss_dict, targets):
        loss_dict = dict()
        student_io_dict = io_dict['student']
        teacher_io_dict = io_dict['teacher']
        for loss_name, (criterion, factor) in self.term_dict.items():
            loss_dict[loss_name] = factor * criterion(student_io_dict, teacher_io_dict, targets)
        sub_total_loss = sum(loss for loss in loss_dict.values()) if len(loss_dict) > 0 else 0
        if self.model_loss_factor is None or isinstance(self.model_loss_factor, (int, float)) and self.model_loss_factor == 0:
            return sub_total_loss
        if isinstance(self.model_loss_factor, dict):
            model_loss = sum([(self.model_loss_factor[k] * v) for k, v in model_loss_dict.items()])
            return sub_total_loss + model_loss
        return sub_total_loss + self.model_loss_factor * sum(model_loss_dict.values() if len(model_loss_dict) > 0 else [])

    def __str__(self):
        desc = 'Loss = '
        tuple_list = [(self.model_loss_factor, 'ModelLoss')] if self.model_loss_factor is not None and self.model_loss_factor != 0 else list()
        tuple_list.extend([(factor, criterion) for criterion, factor in self.term_dict.values()])
        desc += ' + '.join(['{} * {}'.format(factor, criterion) for factor, criterion in tuple_list])
        return desc


class SimpleLossWrapper(nn.Module):
    """
    A simple loss wrapper module designed to use low-level loss modules (e.g., loss modules in PyTorch)
    in torchdistill's pipelines.

    :param low_level_loss: low-level loss module e.g., torch.nn.CrossEntropyLoss.
    :type low_level_loss: nn.Module
    :param kwargs: kwargs to configure what the wrapper passes ``low_level_loss``.
    :type kwargs: dict or None

    .. code-block:: YAML
       :caption: An example YAML to instantiate :class:`SimpleLossWrapper`.

        criterion_wrapper:
          key: 'SimpleLossWrapper'
          kwargs:
            input:
              is_from_teacher: False
              module_path: '.'
              io: 'output'
            target:
              uses_label: True
    """

    def __init__(self, low_level_loss, **kwargs):
        super().__init__()
        self.low_level_loss = low_level_loss
        input_config = kwargs['input']
        self.is_input_from_teacher = input_config['is_from_teacher']
        self.input_module_path = input_config['module_path']
        self.input_key = input_config['io']
        target_config = kwargs.get('target', dict())
        self.uses_label = target_config.get('uses_label', False)
        self.is_target_from_teacher = target_config.get('is_from_teacher', None)
        self.target_module_path = target_config.get('module_path', None)
        self.target_key = target_config.get('io', None)

    @staticmethod
    def extract_value(io_dict, path, key):
        return io_dict[path][key]

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        input_batch = self.extract_value(teacher_io_dict if self.is_input_from_teacher else student_io_dict, self.input_module_path, self.input_key)
        if self.target_module_path is None and self.target_key is None:
            target_batch = targets
        else:
            target_batch = self.extract_value(teacher_io_dict if self.is_target_from_teacher else student_io_dict, self.target_module_path, self.target_key)
        return self.low_level_loss(input_batch, target_batch, *args, **kwargs)

    def __str__(self):
        return self.low_level_loss.__str__()


class DictLossWrapper(SimpleLossWrapper):
    """
    A dict-based wrapper module designed to use low-level loss modules (e.g., loss modules in PyTorch)
    in torchdistill's pipelines. This is a subclass of :class:`SimpleLossWrapper` and useful for models whose forward
    output is dict.

    :param low_level_loss: low-level loss module e.g., torch.nn.CrossEntropyLoss.
    :type low_level_loss: nn.Module
    :param weights: dict contains keys that match the model's output dict keys and corresponding loss weights.
    :type weights: dict
    :param kwargs: kwargs to configure what the wrapper passes ``low_level_loss``.
    :type kwargs: dict or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`DictLossWrapper` for deeplabv3_resnet50 in torchvision, whose default output is a dict of outputs from its main and auxiliary branches with keys 'out' and 'aux' respectively.

        criterion_wrapper:
          key: 'DictLossWrapper'
          kwargs:
            input:
              is_from_teacher: False
              module_path: '.'
              io: 'output'
            target:
              uses_label: True
            weights:
              out: 1.0
              aux: 0.5
    """

    def __init__(self, low_level_loss, weights, **kwargs):
        super().__init__(low_level_loss, **kwargs)
        self.weights = weights

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        input_batch = self.extract_value(teacher_io_dict if self.is_input_from_teacher else student_io_dict, self.input_module_path, self.input_key)
        if self.target_module_path is None and self.target_key is None:
            target_batch = targets
        else:
            target_batch = self.extract_value(teacher_io_dict if self.is_target_from_teacher else student_io_dict, self.target_module_path, self.target_key)
        loss = None
        for key, weight in self.weights.items():
            sub_loss = self.low_level_loss(input_batch[key], target_batch, *args, **kwargs)
            if loss is None:
                loss = weight * sub_loss
            else:
                loss += weight * sub_loss
        return loss

    def __str__(self):
        return str(self.weights) + ' * ' + self.low_level_loss.__str__()


class KDLoss(nn.KLDivLoss):
    """
    A standard knowledge distillation (KD) loss module.

    .. math::

       L_{KD} = \\alpha \\cdot L_{CE} + (1 - \\alpha) \\cdot \\tau^2 \\cdot L_{KL}

    Geoffrey Hinton, Oriol Vinyals, Jeff Dean: `"Distilling the Knowledge in a Neural Network" <https://arxiv.org/abs/1503.02531>`_ @ NIPS 2014 Deep Learning and Representation Learning Workshop (2014)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param temperature: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type temperature: float
    :param alpha: balancing factor for :math:`L_{CE}`, cross-entropy.
    :type alpha: float
    :param beta: balancing factor (default: :math:`1 - \\alpha`) for :math:`L_{KL}`, KL divergence between class-probability distributions softened by :math:`\\tau`.
    :type beta: float or None
    :param reduction: ``reduction`` for KLDivLoss. If ``reduction`` = 'batchmean', CrossEntropyLoss's ``reduction`` will be 'mean'.
    :type reduction: str or None
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io, temperature, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def forward(self, student_io_dict, teacher_io_dict, targets=None, *args, **kwargs):
        student_logits = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_logits = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        soft_loss = super().forward(torch.log_softmax(student_logits / self.temperature, dim=1), torch.softmax(teacher_logits / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = self.cross_entropy_loss(student_logits, targets)
        return self.alpha * hard_loss + self.beta * self.temperature ** 2 * soft_loss


def _extract_feature_map(io_dict, feature_map_config):
    io_type = feature_map_config['io']
    module_path = feature_map_config['path']
    return io_dict[module_path][io_type]


class FSPLoss(nn.Module):
    """
    A loss module for the flow of solution procedure (FSP) matrix.

    Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim: `"A Gift From Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning" <https://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017)

    :param fsp_pairs: configuration of teacher-student module pairs to compute the loss for the FSP matrix.
    :type fsp_pairs: dict

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`FSPLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'FSPLoss'
          kwargs:
            fsp_pairs:
              pair1:
                teacher_first:
                  io: 'input'
                  path: 'layer1'
                teacher_second:
                  io: 'output'
                  path: 'layer1'
                student_first:
                  io: 'input'
                  path: 'layer1'
                student_second:
                  io: 'output'
                  path: 'layer1'
                weight: 1
              pair2:
                teacher_first:
                  io: 'input'
                  path: 'layer2.1'
                teacher_second:
                  io: 'output'
                  path: 'layer2'
                student_first:
                  io: 'input'
                  path: 'layer2.1'
                student_second:
                  io: 'output'
                  path: 'layer2'
                weight: 1
    """

    def __init__(self, fsp_pairs, **kwargs):
        super().__init__()
        self.fsp_pairs = fsp_pairs

    @staticmethod
    def compute_fsp_matrix(first_feature_map, second_feature_map):
        first_h, first_w = first_feature_map.shape[2:4]
        second_h, second_w = second_feature_map.shape[2:4]
        target_h, target_w = min(first_h, second_h), min(first_w, second_w)
        if first_h > target_h or first_w > target_w:
            first_feature_map = adaptive_max_pool2d(first_feature_map, (target_h, target_w))
        if second_h > target_h or second_w > target_w:
            second_feature_map = adaptive_max_pool2d(second_feature_map, (target_h, target_w))
        first_feature_map = first_feature_map.flatten(2)
        second_feature_map = second_feature_map.flatten(2)
        hw = first_feature_map.shape[2]
        return torch.matmul(first_feature_map, second_feature_map.transpose(1, 2)) / hw

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        fsp_loss = 0
        batch_size = None
        for pair_name, pair_config in self.fsp_pairs.items():
            student_first_feature_map = _extract_feature_map(student_io_dict, pair_config['student_first'])
            student_second_feature_map = _extract_feature_map(student_io_dict, pair_config['student_second'])
            student_fsp_matrices = self.compute_fsp_matrix(student_first_feature_map, student_second_feature_map)
            teacher_first_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher_first'])
            teacher_second_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher_second'])
            teacher_fsp_matrices = self.compute_fsp_matrix(teacher_first_feature_map, teacher_second_feature_map)
            factor = pair_config.get('weight', 1)
            fsp_loss += factor * (student_fsp_matrices - teacher_fsp_matrices).norm(dim=1).sum()
            if batch_size is None:
                batch_size = student_first_feature_map.shape[0]
        return fsp_loss / batch_size


class ATLoss(nn.Module):
    """
    A loss module for attention transfer (AT). Referred to https://github.com/szagoruyko/attention-transfer/blob/master/utils.py

    Sergey Zagoruyko, Nikos Komodakis: `"Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer" <https://openreview.net/forum?id=Sks9_ajex>`_ @ ICLR 2017 (2017)

    :param at_pairs: configuration of teacher-student module pairs to compute the loss for attention transfer.
    :type at_pairs: dict
    :param mode: reference to follow 'paper' or 'code'.
    :type mode: dict

    .. warning::
        There is a discrepancy between Eq. (2) in the paper and `the authors' implementation <https://github.com/szagoruyko/attention-transfer/blob/893df5488f93691799f082a70e2521a9dc2ddf2d/utils.py#L18-L23>`_
        as pointed out in `a paper <https://link.springer.com/chapter/10.1007/978-3-030-76423-4_3>`_ and `an issue at the repository <https://github.com/szagoruyko/attention-transfer/issues/34>`_.
        Use ``mode`` = 'paper' instead of 'code' if you want to follow the equations in the paper.

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`ATLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'ATLoss'
          kwargs:
            at_pairs:
              pair1:
                teacher:
                  io: 'output'
                  path: 'layer3'
                student:
                  io: 'output'
                  path: 'layer3'
                weight: 1
              pair2:
                teacher:
                  io: 'output'
                  path: 'layer4'
                student:
                  io: 'output'
                  path: 'layer4'
                weight: 1
            mode: 'code'
    """

    def __init__(self, at_pairs, mode='code', **kwargs):
        super().__init__()
        self.at_pairs = at_pairs
        self.mode = mode
        if mode not in ('code', 'paper'):
            raise ValueError('mode `{}` is not expected'.format(mode))

    @staticmethod
    def attention_transfer_paper(feature_map):
        return normalize(feature_map.pow(2).sum(1).flatten(1))

    def compute_at_loss_paper(self, student_feature_map, teacher_feature_map):
        at_student = self.attention_transfer_paper(student_feature_map)
        at_teacher = self.attention_transfer_paper(teacher_feature_map)
        return torch.norm(at_student - at_teacher, dim=1).sum()

    @staticmethod
    def attention_transfer(feature_map):
        return normalize(feature_map.pow(2).mean(1).flatten(1))

    def compute_at_loss(self, student_feature_map, teacher_feature_map):
        at_student = self.attention_transfer(student_feature_map)
        at_teacher = self.attention_transfer(teacher_feature_map)
        return (at_student - at_teacher).pow(2).mean()

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        at_loss = 0
        batch_size = None
        for pair_name, pair_config in self.at_pairs.items():
            student_feature_map = _extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('weight', 1)
            if self.mode == 'paper':
                at_loss += factor * self.compute_at_loss_paper(student_feature_map, teacher_feature_map)
            else:
                at_loss += factor * self.compute_at_loss(student_feature_map, teacher_feature_map)
            if batch_size is None:
                batch_size = len(student_feature_map)
        return at_loss / batch_size if self.mode == 'paper' else at_loss


class PKTLoss(nn.Module):
    """
    A loss module for probabilistic knowledge transfer (PKT). Refactored https://github.com/passalis/probabilistic_kt/blob/master/nn/pkt.py

    Nikolaos Passalis, Anastasios Tefas: `"Learning Deep Representations with Probabilistic Knowledge Transfer" <https://openaccess.thecvf.com/content_ECCV_2018/html/Nikolaos_Passalis_Learning_Deep_Representations_ECCV_2018_paper.html>`_ @ ECCV 2018 (2018)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param eps: constant to avoid zero division.
    :type eps: float

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`PKTLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'PKTLoss'
          kwargs:
            student_module_path: 'fc'
            student_module_io: 'input'
            teacher_module_path: 'fc'
            teacher_module_io: 'input'
            eps: 0.0000001
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io, eps=1e-07):
        super().__init__()
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.eps = eps

    def cosine_similarity_loss(self, student_outputs, teacher_outputs):
        norm_s = torch.sqrt(torch.sum(student_outputs ** 2, dim=1, keepdim=True))
        student_outputs = student_outputs / (norm_s + self.eps)
        student_outputs[student_outputs != student_outputs] = 0
        norm_t = torch.sqrt(torch.sum(teacher_outputs ** 2, dim=1, keepdim=True))
        teacher_outputs = teacher_outputs / (norm_t + self.eps)
        teacher_outputs[teacher_outputs != teacher_outputs] = 0
        student_similarity = torch.mm(student_outputs, student_outputs.transpose(0, 1))
        teacher_similarity = torch.mm(teacher_outputs, teacher_outputs.transpose(0, 1))
        student_similarity = (student_similarity + 1.0) / 2.0
        teacher_similarity = (teacher_similarity + 1.0) / 2.0
        student_similarity = student_similarity / torch.sum(student_similarity, dim=1, keepdim=True)
        teacher_similarity = teacher_similarity / torch.sum(teacher_similarity, dim=1, keepdim=True)
        return torch.mean(teacher_similarity * torch.log((teacher_similarity + self.eps) / (student_similarity + self.eps)))

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_penultimate_outputs = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_penultimate_outputs = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        return self.cosine_similarity_loss(student_penultimate_outputs, teacher_penultimate_outputs)


class FTLoss(nn.Module):
    """
    A loss module for factor transfer (FT). This loss module is used at the 2nd stage of FT method.

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param p: the order of norm.
    :type p: int
    :param reduction: loss reduction type.
    :type reduction: str
    :param paraphraser_path: teacher model's paraphrase module path.
    :type paraphraser_path: str
    :param translator_path: student model's translator module path.
    :type translator_path: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`FTLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using auxiliary modules :class:`torchdistill.models.wrapper.Teacher4FactorTransfer` and :class:`torchdistill.models.wrapper.Student4FactorTransfer`.

        criterion:
          key: 'FTLoss'
          kwargs:
            p: 1
            reduction: 'mean'
            paraphraser_path: 'paraphraser'
            translator_path: 'translator'
    """

    def __init__(self, p=1, reduction='mean', paraphraser_path='paraphraser', translator_path='translator', **kwargs):
        super().__init__()
        self.norm_p = p
        self.paraphraser_path = paraphraser_path
        self.translator_path = translator_path
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        paraphraser_flat_outputs = teacher_io_dict[self.paraphraser_path]['output'].flatten(1)
        translator_flat_outputs = student_io_dict[self.translator_path]['output'].flatten(1)
        norm_paraphraser_flat_outputs = paraphraser_flat_outputs / paraphraser_flat_outputs.norm(dim=1).unsqueeze(1)
        norm_translator_flat_outputs = translator_flat_outputs / translator_flat_outputs.norm(dim=1).unsqueeze(1)
        if self.norm_p == 1:
            return nn.functional.l1_loss(norm_translator_flat_outputs, norm_paraphraser_flat_outputs, reduction=self.reduction)
        ft_loss = torch.norm(norm_translator_flat_outputs - norm_paraphraser_flat_outputs, self.norm_p, dim=1)
        return ft_loss.mean() if self.reduction == 'mean' else ft_loss.sum()


class AltActTransferLoss(nn.Module):
    """
    A loss module for distillation of activation boundaries (DAB). Refactored https://github.com/bhheo/AB_distillation/blob/master/cifar10_AB_distillation.py

    Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi: `"Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons" <https://ojs.aaai.org/index.php/AAAI/article/view/4264>`_ @ AAAI 2019 (2019)

    :param feature_pairs: configuration of teacher-student module pairs to compute the loss for distillation of activation boundaries.
    :type feature_pairs: dict
    :param margin: margin.
    :type margin: float
    :param reduction: loss reduction type.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`AltActTransferLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Connector4DAB`.

        criterion:
          key: 'AltActTransferLoss'
          kwargs:
            feature_pairs:
              pair1:
                teacher:
                  io: 'output'
                  path: 'layer1'
                student:
                  io: 'output'
                  path: 'connector_dict.connector1'
                weight: 1
              pair2:
                teacher:
                  io: 'output'
                  path: 'layer2'
                student:
                  io: 'output'
                  path: 'connector_dict.connector2'
                weight: 1
              pair3:
                teacher:
                  io: 'output'
                  path: 'layer3'
                student:
                  io: 'output'
                  path: 'connector_dict.connector3'
                weight: 1
              pair4:
                teacher:
                  io: 'output'
                  path: 'layer4'
                student:
                  io: 'output'
                  path: 'connector_dict.connector4'
                weight: 1
            margin: 1.0
            reduction: 'mean'
    """

    def __init__(self, feature_pairs, margin, reduction, **kwargs):
        super().__init__()
        self.feature_pairs = feature_pairs
        self.margin = margin
        self.reduction = reduction

    @staticmethod
    def compute_alt_act_transfer_loss(source, target, margin):
        loss = (source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() + (source - margin) ** 2 * ((source <= margin) & (target > 0)).float()
        return torch.abs(loss).sum()

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        dab_loss = 0
        batch_size = None
        for pair_name, pair_config in self.feature_pairs.items():
            student_feature_map = _extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('weight', 1)
            dab_loss += factor * self.compute_alt_act_transfer_loss(student_feature_map, teacher_feature_map, self.margin)
            if batch_size is None:
                batch_size = student_feature_map.shape[0]
        return dab_loss / batch_size if self.reduction == 'mean' else dab_loss


class RKDLoss(nn.Module):
    """
    A loss module for relational knowledge distillation (RKD). Refactored https://github.com/lenscloth/RKD/blob/master/metric/loss.py

    Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho: `"Relational Knowledge Distillation" <https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param student_output_path: student module path whose output is used in this loss module.
    :type student_output_path: str
    :param teacher_output_path: teacher module path whose output is used in this loss module.
    :type teacher_output_path: str
    :param dist_factor: weight on distance-based RKD loss.
    :type dist_factor: float
    :param angle_factor: weight on angle-based RKD loss.
    :type angle_factor: float
    :param reduction: ``reduction`` for SmoothL1Loss.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`RKDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'RKDLoss'
          kwargs:
            teacher_output_path: 'layer4'
            student_output_path: 'layer4'
            dist_factor: 1.0
            angle_factor: 2.0
            reduction: 'mean'
    """

    def __init__(self, student_output_path, teacher_output_path, dist_factor, angle_factor, reduction, **kwargs):
        super().__init__()
        self.student_output_path = student_output_path
        self.teacher_output_path = teacher_output_path
        self.dist_factor = dist_factor
        self.angle_factor = angle_factor
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction=reduction)

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

    def compute_rkd_distance_loss(self, teacher_flat_outputs, student_flat_outputs):
        if self.dist_factor is None or self.dist_factor == 0:
            return 0
        with torch.no_grad():
            t_d = self.pdist(teacher_flat_outputs, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
        d = self.pdist(student_flat_outputs, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        return self.smooth_l1_loss(d, t_d)

    def compute_rkd_angle_loss(self, teacher_flat_outputs, student_flat_outputs):
        if self.angle_factor is None or self.angle_factor == 0:
            return 0
        with torch.no_grad():
            td = teacher_flat_outputs.unsqueeze(0) - teacher_flat_outputs.unsqueeze(1)
            norm_td = normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        sd = student_flat_outputs.unsqueeze(0) - student_flat_outputs.unsqueeze(1)
        norm_sd = normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        return self.smooth_l1_loss(s_angle, t_angle)

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_flat_outputs = teacher_io_dict[self.teacher_output_path]['output'].flatten(1)
        student_flat_outputs = student_io_dict[self.student_output_path]['output'].flatten(1)
        rkd_distance_loss = self.compute_rkd_distance_loss(teacher_flat_outputs, student_flat_outputs)
        rkd_angle_loss = self.compute_rkd_angle_loss(teacher_flat_outputs, student_flat_outputs)
        return self.dist_factor * rkd_distance_loss + self.angle_factor * rkd_angle_loss


class VIDLoss(nn.Module):
    """
    A loss module for variational information distillation (VID). Referred to https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py

    Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai: `"Variational Information Distillation for Knowledge Transfer" <https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param feature_pairs: configuration of teacher-student module pairs to compute the loss for variational information distillation.
    :type feature_pairs: dict

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`VIDLoss` for a teacher-student pair of ResNet-50 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.VariationalDistributor4VID` for the student model.

        criterion:
          key: 'VIDLoss'
          kwargs:
            feature_pairs:
              pair1:
                teacher:
                  io: 'output'
                  path: 'layer1'
                student:
                  io: 'output'
                  path: 'regressor_dict.regressor1'
                weight: 1
              pair2:
                teacher:
                  io: 'output'
                  path: 'layer2'
                student:
                  io: 'output'
                  path: 'regressor_dict.regressor2'
                weight: 1
              pair3:
                teacher:
                  io: 'output'
                  path: 'layer3'
                student:
                  io: 'output'
                  path: 'regressor_dict.regressor3'
                weight: 1
              pair4:
                teacher:
                  io: 'output'
                  path: 'layer4'
                student:
                  io: 'output'
                  path: 'regressor_dict.regressor4'
                weight: 1
            margin: 1.0
    """

    def __init__(self, feature_pairs, **kwargs):
        super().__init__()
        self.feature_pairs = feature_pairs

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        vid_loss = 0
        for pair_name, pair_config in self.feature_pairs.items():
            pred_mean, pred_var = _extract_feature_map(student_io_dict, pair_config['student'])
            teacher_feature_map = _extract_feature_map(teacher_io_dict, pair_config['teacher'])
            factor = pair_config.get('weight', 1)
            neg_log_prob = 0.5 * ((pred_mean - teacher_feature_map) ** 2 / pred_var + torch.log(pred_var))
            vid_loss += factor * neg_log_prob.mean()
        return vid_loss


class CCKDLoss(nn.Module):
    """
    A loss module for correlation congruence for knowledge distillation (CCKD).

    Baoyun Peng, Xiao Jin, Jiaheng Liu, Dongsheng Li, Yichao Wu, Yu Liu, Shunfeng Zhou, Zhaoning Zhang: `"Correlation Congruence for Knowledge Distillation" <https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html>`_ @ ICCV 2019 (2019)

    :param student_linear_path: student model's linear module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CCKD`.
    :type student_linear_path: str
    :param teacher_linear_path: teacher model's linear module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CCKD`.
    :type teacher_linear_path: str
    :param kernel_config: kernel ('gaussian' or 'bilinear') configuration.
    :type kernel_config: dict
    :param reduction: loss reduction type.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`CCKDLoss` for a teacher-student pair of ResNet-50 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Linear4CCKD` for the teacher and student models.

        criterion:
          key: 'CCKDLoss'
          kwargs:
            teacher_linear_path: 'linear'
            student_linear_path: 'linear'
            kernel_params:
              key: 'gaussian'
              gamma: 0.4
              max_p: 2
            reduction: 'batchmean'
    """

    def __init__(self, student_linear_path, teacher_linear_path, kernel_config, reduction, **kwargs):
        super().__init__()
        self.student_linear_path = student_linear_path
        self.teacher_linear_path = teacher_linear_path
        self.kernel_type = kernel_config['type']
        if self.kernel_type == 'gaussian':
            self.gamma = kernel_config['gamma']
            self.max_p = kernel_config['max_p']
        elif self.kernel_type not in ('bilinear', 'gaussian'):
            raise ValueError('self.kernel_type `{}` is not expected'.format(self.kernel_type))
        self.reduction = reduction

    @staticmethod
    def compute_cc_mat_by_bilinear_pool(linear_outputs):
        return torch.matmul(linear_outputs, torch.t(linear_outputs))

    def compute_cc_mat_by_gaussian_rbf(self, linear_outputs):
        row_list = list()
        for index, linear_output in enumerate(linear_outputs):
            row = 1
            right_term = torch.matmul(linear_output, torch.t(linear_outputs))
            for p in range(1, self.max_p + 1):
                left_term = (2 * self.gamma) ** p / math.factorial(p)
                row += left_term * right_term ** p
            row *= math.exp(-2 * self.gamma)
            row_list.append(row.squeeze(0))
        return torch.stack(row_list)

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_path]['output']
        student_linear_outputs = student_io_dict[self.student_linear_path]['output']
        batch_size = teacher_linear_outputs.shape[0]
        if self.kernel_type == 'bilinear':
            teacher_cc = self.compute_cc_mat_by_bilinear_pool(teacher_linear_outputs)
            student_cc = self.compute_cc_mat_by_bilinear_pool(student_linear_outputs)
        elif self.kernel_type == 'gaussian':
            teacher_cc = self.compute_cc_mat_by_gaussian_rbf(teacher_linear_outputs)
            student_cc = self.compute_cc_mat_by_gaussian_rbf(student_linear_outputs)
        else:
            raise ValueError('self.kernel_type `{}` is not expected'.format(self.kernel_type))
        cc_loss = torch.dist(student_cc, teacher_cc, 2)
        return cc_loss / batch_size ** 2 if self.reduction == 'batchmean' else cc_loss


class SPKDLoss(nn.Module):
    """
    A loss module for similarity-preserving knowledge distillation (SPKD).

    Frederick Tung, Greg Mori: `"Similarity-Preserving Knowledge Distillation" <https://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html>`_ @ ICCV2019 (2019)

    :param student_output_path: student module path whose output is used in this loss module.
    :type student_output_path: str
    :param teacher_output_path: teacher module path whose output is used in this loss module.
    :type teacher_output_path: str
    :param reduction: loss reduction type.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`SPKDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision.

        criterion:
          key: 'SPKDLoss'
          kwargs:
            teacher_output_path: 'layer4'
            student_output_path: 'layer4'
            reduction: 'batchmean'
    """

    def __init__(self, student_output_path, teacher_output_path, reduction, **kwargs):
        super().__init__()
        self.student_output_path = student_output_path
        self.teacher_output_path = teacher_output_path
        self.reduction = reduction

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return normalize(torch.matmul(z, torch.t(z)), 1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_outputs = teacher_io_dict[self.teacher_output_path]['output']
        student_outputs = student_io_dict[self.student_output_path]['output']
        batch_size = teacher_outputs.shape[0]
        spkd_losses = self.compute_spkd_loss(teacher_outputs, student_outputs)
        spkd_loss = spkd_losses.sum()
        return spkd_loss / batch_size ** 2 if self.reduction == 'batchmean' else spkd_loss


def_logger = logging.getLogger()


logger = def_logger.getChild(__name__)


class CRDLoss(nn.Module):
    """
    A loss module for contrastive representation distillation (CRD). Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/criterion.py

    Yonglong Tian, Dilip Krishnan, Phillip Isola: `"Contrastive Representation Distillation" <https://openreview.net/forum?id=SkgpBJrtvS>`_ @ ICLR 2020 (2020)

    :param student_norm_module_path: student model's normalizer module path (:class:`torchdistill.models.wrapper.Normalizer4CRD` in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CRD`).
    :type student_norm_module_path: str
    :param student_empty_module_path: student model's empty module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CRD`.
    :type student_empty_module_path: str
    :param teacher_norm_module_path: teacher model's normalizer module path (:class:`torchdistill.models.wrapper.Normalizer4CRD` in an auxiliary wrapper :class:`torchdistill.models.wrapper.Linear4CRD`).
    :type teacher_norm_module_path: str
    :param input_size: number of input features.
    :type input_size: int
    :param output_size: number of output features.
    :type output_size: int
    :param num_negative_samples: number of negative samples.
    :type num_negative_samples: int
    :param num_samples: number of samples.
    :type num_samples: int
    :param temperature: temperature to adjust concentration level (not the temperature for :class:`KDLoss`).
    :type temperature: float
    :param momentum: momentum.
    :type momentum: float
    :param eps: eps.
    :type eps: float

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`CRDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Linear4CRD` for the teacher and student models.

        criterion:
          key: 'CRDLoss'
          kwargs:
            teacher_norm_module_path: 'normalizer'
            student_norm_module_path: 'normalizer'
            student_empty_module_path: 'empty'
            input_size: *feature_dim
            output_size: &num_samples 1281167
            num_negative_samples: *num_negative_samples
            num_samples: *num_samples
            temperature: 0.07
            momentum: 0.5
            eps: 0.0000001
    """

    def init_prob_alias(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        k = len(probs)
        self.probs = torch.zeros(k)
        self.alias = torch.zeros(k, dtype=torch.int64)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.probs[kk] = k * prob
            if self.probs[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            self.alias[small] = large
            self.probs[large] = self.probs[large] - 1.0 + self.probs[small]
            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        for last_one in (smaller + larger):
            self.probs[last_one] = 1

    def __init__(self, student_norm_module_path, student_empty_module_path, teacher_norm_module_path, input_size, output_size, num_negative_samples, num_samples, temperature=0.07, momentum=0.5, eps=1e-07):
        super().__init__()
        self.student_norm_module_path = student_norm_module_path
        self.student_empty_module_path = student_empty_module_path
        self.teacher_norm_module_path = teacher_norm_module_path
        self.eps = eps
        self.unigrams = torch.ones(output_size)
        self.num_negative_samples = num_negative_samples
        self.num_samples = num_samples
        self.register_buffer('params', torch.tensor([num_negative_samples, temperature, -1, -1, momentum]))
        stdv = 1.0 / math.sqrt(input_size / 3)
        self.register_buffer('memory_v1', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.probs, self.alias = None, None
        self.init_prob_alias(self.unigrams)

    def draw(self, n):
        k = self.alias.size(0)
        kk = torch.zeros(n, dtype=torch.long, device=self.prob.device).random_(0, k)
        prob = self.probs.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())
        return oq + oj

    def contrast_memory(self, student_embed, teacher_embed, pos_indices, contrast_idx=None):
        param_k = int(self.params[0].item())
        param_t = self.params[1].item()
        z_v1 = self.params[2].item()
        z_v2 = self.params[3].item()
        momentum = self.params[4].item()
        batch_size = student_embed.size(0)
        output_size = self.memory_v1.size(0)
        input_size = self.memory_v1.size(1)
        if contrast_idx is None:
            contrast_idx = self.draw(batch_size * (self.num_negative_samples + 1)).view(batch_size, -1)
            contrast_idx.select(1, 0).copy_(pos_indices.data)
        weight_v1 = torch.index_select(self.memory_v1, 0, contrast_idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batch_size, param_k + 1, input_size)
        out_v2 = torch.bmm(weight_v1, teacher_embed.view(batch_size, input_size, 1))
        out_v2 = torch.exp(torch.div(out_v2, param_t))
        weight_v2 = torch.index_select(self.memory_v2, 0, contrast_idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batch_size, param_k + 1, input_size)
        out_v1 = torch.bmm(weight_v2, student_embed.view(batch_size, input_size, 1))
        out_v1 = torch.exp(torch.div(out_v1, param_t))
        if z_v1 < 0:
            self.params[2] = out_v1.mean() * output_size
            z_v1 = self.params[2].clone().detach().item()
            logger.info('normalization constant z_v1 is set to {:.1f}'.format(z_v1))
        if z_v2 < 0:
            self.params[3] = out_v2.mean() * output_size
            z_v2 = self.params[3].clone().detach().item()
            logger.info('normalization constant z_v2 is set to {:.1f}'.format(z_v2))
        out_v1 = torch.div(out_v1, z_v1).contiguous()
        out_v2 = torch.div(out_v2, z_v2).contiguous()
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, pos_indices.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(student_embed, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, pos_indices, updated_v1)
            ab_pos = torch.index_select(self.memory_v2, 0, pos_indices.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(teacher_embed, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, pos_indices, updated_v2)
        return out_v1, out_v2

    def compute_contrast_loss(self, x):
        batch_size = x.shape[0]
        m = x.size(1) - 1
        pn = 1 / float(self.num_samples)
        p_pos = x.select(1, 0)
        log_d1 = torch.div(p_pos, p_pos.add(m * pn + self.eps)).log_()
        p_neg = x.narrow(1, 1, m)
        log_d0 = torch.div(p_neg.clone().fill_(m * pn), p_neg.add(m * pn + self.eps)).log_()
        loss = -(log_d1.sum(0) + log_d0.view(-1, 1).sum(0)) / batch_size
        return loss

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_linear_outputs = teacher_io_dict[self.teacher_norm_module_path]['output']
        student_linear_outputs = student_io_dict[self.student_norm_module_path]['output']
        supp_dict = student_io_dict[self.student_empty_module_path]['input']
        pos_idx, contrast_idx = supp_dict['pos_idx'], supp_dict.get('contrast_idx', None)
        device = student_linear_outputs.device
        pos_idx = pos_idx
        if contrast_idx is not None:
            contrast_idx = contrast_idx
        if device != self.probs.device:
            self.probs
            self.alias
            self
        out_s, out_t = self.contrast_memory(student_linear_outputs, teacher_linear_outputs, pos_idx, contrast_idx)
        student_contrast_loss = self.compute_contrast_loss(out_s)
        teacher_contrast_loss = self.compute_contrast_loss(out_t)
        loss = student_contrast_loss + teacher_contrast_loss
        return loss


class AuxSSKDLoss(nn.CrossEntropyLoss):
    """
    A loss module for self-supervision knowledge distillation (SSKD) that treats contrastive prediction as
    a self-supervision task (auxiliary task). This loss module is used at the 1st stage of SSKD method.
    Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py

    Guodong Xu, Ziwei Liu, Xiaoxiao Li, Chen Change Loy: `"Knowledge Distillation Meets Self-Supervision" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param module_path: model's self-supervision module path.
    :type module_path: str
    :param module_io: 'input' or 'output' of the module in the model.
    :type module_io: str
    :param reduction: ``reduction`` for CrossEntropyLoss.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`AuxSSKDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.SSWrapper4SSKD` for teacher model.

        criterion:
          key: 'AuxSSKDLoss'
          kwargs:
            module_path: 'ss_module'
            module_io: 'output'
            reduction: 'mean'
    """

    def __init__(self, module_path='ss_module', module_io='output', reduction='mean', **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.module_path = module_path
        self.module_io = module_io

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        ss_module_outputs = teacher_io_dict[self.module_path][self.module_io]
        device = ss_module_outputs.device
        batch_size = ss_module_outputs.shape[0]
        three_forth_batch_size = int(batch_size * 3 / 4)
        one_forth_batch_size = batch_size - three_forth_batch_size
        normal_indices = torch.arange(batch_size) % 4 == 0
        aug_indices = torch.arange(batch_size) % 4 != 0
        normal_rep = ss_module_outputs[normal_indices]
        aug_rep = ss_module_outputs[aug_indices]
        normal_rep = normal_rep.unsqueeze(2).expand(-1, -1, three_forth_batch_size).transpose(0, 2)
        aug_rep = aug_rep.unsqueeze(2).expand(-1, -1, one_forth_batch_size)
        cos_similarities = cosine_similarity(aug_rep, normal_rep, dim=1)
        targets = torch.arange(one_forth_batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        targets = targets[:three_forth_batch_size].long()
        return super().forward(cos_similarities, targets)


class SSKDLoss(nn.Module):
    """
    A loss module for self-supervision knowledge distillation (SSKD).
    This loss module is used at the 2nd stage of SSKD method. Refactored https://github.com/xuguodong03/SSKD/blob/master/student.py

    Guodong Xu, Ziwei Liu, Xiaoxiao Li, Chen Change Loy: `"Knowledge Distillation Meets Self-Supervision" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param student_linear_path: student model's linear module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.SSWrapper4SSKD`.
    :type student_linear_path: str
    :param teacher_linear_path: teacher model's linear module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.SSWrapper4SSKD`.
    :type teacher_linear_path: str
    :param student_ss_module_path: student model's self-supervision module path.
    :type student_ss_module_path: str
    :param teacher_ss_module_path: teacher model's self-supervision module path.
    :type teacher_ss_module_path: str
    :param kl_temp: temperature to soften teacher and student's class-probability distributions for KL divergence given original data.
    :type kl_temp: float
    :param ss_temp: temperature to soften teacher and student's self-supervision cosine similarities for KL divergence.
    :type ss_temp: float
    :param tf_temp: temperature to soften teacher and student's class-probability distributions for KL divergence given augmented data by transform.
    :type tf_temp: float
    :param ss_ratio: ratio of samples with the smallest error levels used for self-supervision.
    :type ss_ratio: float
    :param tf_ratio: ratio of samples with the smallest error levels used for transform.
    :type tf_ratio: float
    :param student_linear_module_io: 'input' or 'output' of the linear module in the student model.
    :type student_linear_module_io: str
    :param teacher_linear_module_io: 'input' or 'output' of the linear module in the teacher model.
    :type teacher_linear_module_io: str
    :param student_ss_module_io: 'input' or 'output' of the self-supervision module in the student model.
    :type student_ss_module_io: str
    :param teacher_ss_module_io: 'input' or 'output' of the self-supervision module in the teacher model.
    :type teacher_ss_module_io: str
    :param loss_weights: weights for 1) cross-entropy, 2) KL divergence for the original data, 3) KL divergence for self-supervision cosine similarities, and 4) KL divergence for the augmented data by transform.
    :type loss_weights: list[float] or None
    :param reduction: ``reduction`` for KLDivLoss. If ``reduction`` = 'batchmean', CrossEntropyLoss's ``reduction`` will be 'mean'.
    :type reduction: str or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`SSKDLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.SSWrapper4SSKD` for the teacher and student models.

        criterion:
          key: 'SSKDLoss'
          kwargs:
            student_linear_module_path: 'model.fc'
            teacher_linear_module_path: 'model.fc'
            student_ss_module_path: 'ss_module'
            teacher_ss_module_path: 'ss_module'
            kl_temp: 4.0
            ss_temp: 0.5
            tf_temp: 4.0
            ss_ratio: 0.75
            tf_ratio: 1.0
            loss_weights: [1.0, 0.9, 10.0, 2.7]
            reduction: 'batchmean'
    """

    def __init__(self, student_linear_module_path, teacher_linear_module_path, student_ss_module_path, teacher_ss_module_path, kl_temp, ss_temp, tf_temp, ss_ratio, tf_ratio, student_linear_module_io='output', teacher_linear_module_io='output', student_ss_module_io='output', teacher_ss_module_io='output', loss_weights=None, reduction='batchmean', **kwargs):
        super().__init__()
        self.loss_weights = [1.0, 1.0, 1.0, 1.0] if loss_weights is None else loss_weights
        self.kl_temp = kl_temp
        self.ss_temp = ss_temp
        self.tf_temp = tf_temp
        self.ss_ratio = ss_ratio
        self.tf_ratio = tf_ratio
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction)
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.student_linear_module_path = student_linear_module_path
        self.student_linear_module_io = student_linear_module_io
        self.teacher_linear_module_path = teacher_linear_module_path
        self.teacher_linear_module_io = teacher_linear_module_io
        self.student_ss_module_path = student_ss_module_path
        self.student_ss_module_io = student_ss_module_io
        self.teacher_ss_module_path = teacher_ss_module_path
        self.teacher_ss_module_io = teacher_ss_module_io

    @staticmethod
    def compute_cosine_similarities(ss_module_outputs, normal_indices, aug_indices, three_forth_batch_size, one_forth_batch_size):
        normal_feat = ss_module_outputs[normal_indices]
        aug_feat = ss_module_outputs[aug_indices]
        normal_feat = normal_feat.unsqueeze(2).expand(-1, -1, three_forth_batch_size).transpose(0, 2)
        aug_feat = aug_feat.unsqueeze(2).expand(-1, -1, one_forth_batch_size)
        return cosine_similarity(aug_feat, normal_feat, dim=1)

    def forward(self, student_io_dict, teacher_io_dict, targets, *args, **kwargs):
        student_linear_outputs = student_io_dict[self.student_linear_module_path][self.student_linear_module_io]
        teacher_linear_outputs = teacher_io_dict[self.teacher_linear_module_path][self.teacher_linear_module_io]
        device = student_linear_outputs.device
        batch_size = student_linear_outputs.shape[0]
        three_forth_batch_size = int(batch_size * 3 / 4)
        one_forth_batch_size = batch_size - three_forth_batch_size
        normal_indices = torch.arange(batch_size) % 4 == 0
        aug_indices = torch.arange(batch_size) % 4 != 0
        ce_loss = self.cross_entropy_loss(student_linear_outputs[normal_indices], targets)
        kl_loss = self.kldiv_loss(torch.log_softmax(student_linear_outputs[normal_indices] / self.kl_temp, dim=1), torch.softmax(teacher_linear_outputs[normal_indices] / self.kl_temp, dim=1))
        kl_loss *= self.kl_temp ** 2
        aug_knowledges = torch.softmax(teacher_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        aug_targets = targets.unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long()
        ranks = torch.argsort(aug_knowledges, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.tf_ratio)
        indices = indices[:correct_num + wrong_keep]
        distill_index_tf = torch.sort(indices)[0]
        student_ss_module_outputs = student_io_dict[self.student_ss_module_path][self.student_ss_module_io]
        teacher_ss_module_outputs = teacher_io_dict[self.teacher_ss_module_path][self.teacher_ss_module_io]
        s_cos_similarities = self.compute_cosine_similarities(student_ss_module_outputs, normal_indices, aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = self.compute_cosine_similarities(teacher_ss_module_outputs, normal_indices, aug_indices, three_forth_batch_size, one_forth_batch_size)
        t_cos_similarities = t_cos_similarities.detach()
        aug_targets = torch.arange(one_forth_batch_size).unsqueeze(1).expand(-1, 3).contiguous().view(-1)
        aug_targets = aug_targets[:three_forth_batch_size].long()
        ranks = torch.argsort(t_cos_similarities, dim=1, descending=True)
        ranks = torch.argmax(torch.eq(ranks, aug_targets.unsqueeze(1)).long(), dim=1)
        indices = torch.argsort(ranks)
        tmp = torch.nonzero(ranks, as_tuple=True)[0]
        wrong_num = tmp.numel()
        correct_num = three_forth_batch_size - wrong_num
        wrong_keep = int(wrong_num * self.ss_ratio)
        indices = indices[:correct_num + wrong_keep]
        distill_index_ss = torch.sort(indices)[0]
        ss_loss = self.kldiv_loss(torch.log_softmax(s_cos_similarities[distill_index_ss] / self.ss_temp, dim=1), torch.softmax(t_cos_similarities[distill_index_ss] / self.ss_temp, dim=1))
        ss_loss *= self.ss_temp ** 2
        log_aug_outputs = torch.log_softmax(student_linear_outputs[aug_indices] / self.tf_temp, dim=1)
        tf_loss = self.kldiv_loss(log_aug_outputs[distill_index_tf], aug_knowledges[distill_index_tf])
        tf_loss *= self.tf_temp ** 2
        total_loss = 0
        for loss_weight, loss in zip(self.loss_weights, [ce_loss, kl_loss, ss_loss, tf_loss]):
            total_loss += loss_weight * loss
        return total_loss


class PADL2Loss(nn.Module):
    """
    A loss module for prime-aware adaptive distillation (PAD) with L2 loss. This loss module is used at the 2nd stage of PAD method.

    Youcai Zhang, Zhonghao Lan, Yuchen Dai, Fangao Zeng, Yan Bai, Jie Chang, Yichen Wei: `"Prime-Aware Adaptive Distillation" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3317_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param student_embed_module_path: student model's embedding module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.VarianceBranch4PAD`.
    :type student_embed_module_path: str
    :param teacher_embed_module_path: teacher model's embedding module path.
    :type teacher_embed_module_path: str
    :param student_embed_module_io: 'input' or 'output' of the embedding module in the student model.
    :type student_embed_module_io: str
    :param teacher_embed_module_io: 'input' or 'output' of the embedding module in the teacher model.
    :type teacher_embed_module_io: str
    :param module_path: student model's variance estimator module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.VarianceBranch4PAD`.
    :type module_path: str
    :param module_io: 'input' or 'output' of the variance estimator module in the student model.
    :type module_io: str
    :param eps: constant to avoid zero division.
    :type eps: float
    :param reduction: loss reduction type.
    :type reduction: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`PADL2Loss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.VarianceBranch4PAD` for the student model.

        criterion:
          key: 'PADL2Loss'
          kwargs:
            student_embed_module_path: 'student_model.avgpool'
            student_embed_module_io: 'output'
            teacher_embed_module_path: 'avgpool'
            teacher_embed_module_io: 'output'
            module_path: 'var_estimator'
            module_io: 'output'
            eps: 0.000001
            reduction: 'mean'
    """

    def __init__(self, student_embed_module_path, teacher_embed_module_path, student_embed_module_io='output', teacher_embed_module_io='output', module_path='var_estimator', module_io='output', eps=1e-06, reduction='mean', **kwargs):
        super().__init__()
        self.student_embed_module_path = student_embed_module_path
        self.teacher_embed_module_path = teacher_embed_module_path
        self.student_embed_module_io = student_embed_module_io
        self.teacher_embed_module_io = teacher_embed_module_io
        self.module_path = module_path
        self.module_io = module_io
        self.eps = eps
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        log_variances = student_io_dict[self.module_path][self.module_io]
        student_embed_outputs = student_io_dict[self.student_embed_module_path][self.student_embed_module_io].flatten(1)
        teacher_embed_outputs = teacher_io_dict[self.teacher_embed_module_path][self.teacher_embed_module_io].flatten(1)
        squared_losses = torch.mean((teacher_embed_outputs - student_embed_outputs) ** 2 / (self.eps + torch.exp(log_variances)) + log_variances, dim=1)
        return squared_losses.mean() if self.reduction == 'mean' else squared_losses.sum()


class HierarchicalContextLoss(nn.Module):
    """
    A loss module for knowledge review (KR) method. Referred to https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia: `"Distilling Knowledge via Knowledge Review" <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html>`_ @ CVPR 2021 (2021)

    :param student_module_path: student model's module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Student4KnowledgeReview`.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param reduction: ``reduction`` for MSELoss.
    :type reduction: str or None
    :param output_sizes: output sizes of adaptive_avg_pool2d.
    :type output_sizes: list[int] or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`HierarchicalContextLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Student4KnowledgeReview` for the student model.

        criterion:
          key: 'HierarchicalContextLoss'
          kwargs:
            student_module_path: 'abf_modules.4'
            student_module_io: 'output'
            teacher_module_path: 'layer1.-1.relu'
            teacher_module_io: 'input'
            reduction: 'mean'
            output_sizes: [4, 2, 1]
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io, reduction='mean', output_sizes=None, **kwargs):
        super().__init__()
        if output_sizes is None:
            output_sizes = [4, 2, 1]
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.criteria = nn.MSELoss(reduction=reduction)
        self.output_sizes = output_sizes

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_features, _ = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_features = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        _, _, h, _ = student_features.shape
        loss = self.criteria(student_features, teacher_features)
        weight = 1.0
        total_weight = 1.0
        for k in self.output_sizes:
            if k >= h:
                continue
            proc_student_features = adaptive_avg_pool2d(student_features, (k, k))
            proc_teacher_features = adaptive_avg_pool2d(teacher_features, (k, k))
            weight /= 2.0
            loss += weight * self.criteria(proc_student_features, proc_teacher_features)
            total_weight += weight
        return loss / total_weight


class RegularizationLoss(nn.Module):
    """
    A regularization loss module.

    :param module_path: module path.
    :type module_path: str
    :param module_io: 'input' or 'output' of the module in the student model.
    :type module_io: str
    :param is_from_teacher: True if you use teacher's I/O dict. Otherwise, you use student's I/O dict.
    :type is_from_teacher: bool
    :param p: the order of norm.
    :type p: int
    """

    def __init__(self, module_path, io_type='output', is_from_teacher=False, p=1, **kwargs):
        super().__init__()
        self.module_path = module_path
        self.io_type = io_type
        self.is_from_teacher = is_from_teacher
        self.norm_p = p

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        io_dict = teacher_io_dict if self.is_from_teacher else student_io_dict
        z = io_dict[self.module_path][self.io_type]
        return z.norm(p=self.norm_p)


class KTALoss(nn.Module):
    """
    A loss module for knowledge translation and adaptation (KTA).
    This loss module is used at the 2nd stage of KTAAD method.

    Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan.: `"Knowledge Adaptation for Efficient Semantic Segmentation" <https://openaccess.thecvf.com/content_CVPR_2019/html/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param p: the order of norm for differences between normalized feature adapter's (flattened) output and knowledge translator's (flattened) output.
    :type p: int
    :param q: the order of norm for the denominator to normalize feature adapter (flattened) output.
    :type q: int
    :param reduction: loss reduction type.
    :type reduction: str
    :param knowledge_translator_path: knowledge translator module path.
    :type knowledge_translator_path: str
    :param feature_adapter_path: feature adapter module path.
    :type feature_adapter_path: str

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`KTALoss` for a teacher-student pair of DeepLabv3 with ResNet50 and LRASPP with MobileNet v3 (Large) in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Teacher4FactorTransfer` and :class:`torchdistill.models.wrapper.Student4KTAAD` for the teacher and student models.

        criterion:
          key: 'KTALoss'
          kwargs:
            p: 1
            q: 2
            reduction: 'mean'
            knowledge_translator_path: 'paraphraser.encoder'
            feature_adapter_path: 'feature_adapter'
    """

    def __init__(self, p=1, q=2, reduction='mean', knowledge_translator_path='paraphraser', feature_adapter_path='feature_adapter', **kwargs):
        super().__init__()
        self.norm_p = p
        self.norm_q = q
        self.knowledge_translator_path = knowledge_translator_path
        self.feature_adapter_path = feature_adapter_path
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        knowledge_translator_flat_outputs = teacher_io_dict[self.knowledge_translator_path]['output'].flatten(1)
        feature_adapter_flat_outputs = student_io_dict[self.feature_adapter_path]['output'].flatten(1)
        norm_knowledge_translator_flat_outputs = knowledge_translator_flat_outputs / knowledge_translator_flat_outputs.norm(p=self.norm_q, dim=1).unsqueeze(1)
        norm_feature_adapter_flat_outputs = feature_adapter_flat_outputs / feature_adapter_flat_outputs.norm(p=self.norm_q, dim=1).unsqueeze(1)
        if self.norm_p == 1:
            return nn.functional.l1_loss(norm_feature_adapter_flat_outputs, norm_knowledge_translator_flat_outputs, reduction=self.reduction)
        kta_loss = torch.norm(norm_feature_adapter_flat_outputs - norm_knowledge_translator_flat_outputs, self.norm_p, dim=1)
        return kta_loss.mean() if self.reduction == 'mean' else kta_loss.sum()


class AffinityLoss(nn.Module):
    """
    A loss module for affinity distillation in KTA. This loss module is used at the 2nd stage of KTAAD method.

    Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan.: `"Knowledge Adaptation for Efficient Semantic Segmentation" <https://openaccess.thecvf.com/content_CVPR_2019/html/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param student_module_path: student model's module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Student4KTAAD`.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.Teacher4FactorTransfer`.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param reduction: loss reduction type.
    :type reduction: str or None

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`AffinityLoss` for a teacher-student pair of DeepLabv3 with ResNet50 and LRASPP with MobileNet v3 (Large) in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Teacher4FactorTransfer` and :class:`torchdistill.models.wrapper.Student4KTAAD` for the teacher and student models.

        criterion:
          key: 'AffinityLoss'
          kwargs:
            student_module_path: 'affinity_adapter'
            student_module_io: 'output'
            teacher_module_path: 'paraphraser.encoder'
            teacher_module_io: 'output'
            reduction: 'mean'
    """

    def __init__(self, student_module_path, teacher_module_path, student_module_io='output', teacher_module_io='output', reduction='mean', **kwargs):
        super().__init__()
        self.student_module_path = student_module_path
        self.teacher_module_path = teacher_module_path
        self.student_module_io = student_module_io
        self.teacher_module_io = teacher_module_io
        self.reduction = reduction

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_flat_outputs = student_io_dict[self.student_module_path][self.student_module_io].flatten(2)
        teacher_flat_outputs = teacher_io_dict[self.teacher_module_path][self.teacher_module_io].flatten(2)
        batch_size, ch_size, hw = student_flat_outputs.shape
        student_flat_outputs = student_flat_outputs / student_flat_outputs.norm(p=2, dim=2).unsqueeze(-1)
        teacher_flat_outputs = teacher_flat_outputs / teacher_flat_outputs.norm(p=2, dim=2).unsqueeze(-1)
        total_squared_losses = torch.zeros(batch_size)
        for i in range(ch_size):
            total_squared_losses += ((torch.bmm(student_flat_outputs[:, i].unsqueeze(2), student_flat_outputs[:, i].unsqueeze(1)) - torch.bmm(teacher_flat_outputs[:, i].unsqueeze(2), teacher_flat_outputs[:, i].unsqueeze(1))) / hw).norm(p=2, dim=(1, 2))
        return total_squared_losses.mean() if self.reduction == 'mean' else total_squared_losses.sum()


class ChSimLoss(nn.Module):
    """
    A loss module for Inter-Channel Correlation for Knowledge Distillation (ICKD).
    Refactored https://github.com/ADLab-AutoDrive/ICKD/blob/main/ImageNet/torchdistill/losses/single.py

    Li Liu, Qingle Huang, Sihao Lin, Hongwei Xie, Bing Wang, Xiaojun Chang, Xiaodan Liang: `"Inter-Channel Correlation for Knowledge Distillation" <https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html>`_ @ ICCV 2021 (2021)

    :param feature_pairs: configuration of teacher-student module pairs to compute the L2 distance between the inter-channel correlation matrices of the student and the teacher.
    :type feature_pairs: dict

    .. code-block:: yaml
       :caption: An example YAML to instantiate :class:`ChSimLoss` for a teacher-student pair of ResNet-34 and ResNet-18 in torchvision, using an auxiliary module :class:`torchdistill.models.wrapper.Student4ICKD`.

        criterion:
          key: 'ChSimLoss'
          kwargs:
            feature_pairs:
              pair1:
                teacher:
                  io: 'output'
                  path: 'layer4'
                student:
                  io: 'output'
                  path: 'embed_dict.embed1'
                weight: 1
    """

    def __init__(self, feature_pairs, **kwargs):
        super().__init__()
        self.feature_pairs = feature_pairs
        self.smooth_l1_loss = nn.SmoothL1Loss()

    @staticmethod
    def batch_loss(f_s, f_t):
        bsz, ch = f_s.shape[0], f_s.shape[1]
        f_s = f_s.view(bsz, ch, -1)
        f_t = f_t.view(bsz, ch, -1)
        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)
        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)
        g_diff = emd_s - emd_t
        loss = (g_diff * g_diff).view(bsz, -1).sum() / (ch * bsz * bsz)
        return loss

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        chsim_loss = 0
        for pair_name, pair_config in self.feature_pairs.items():
            teacher_outputs = _extract_feature_map(teacher_io_dict, pair_config['teacher'])
            student_outputs = _extract_feature_map(student_io_dict, pair_config['student'])
            weight = pair_config.get('weight', 1)
            loss = self.batch_loss(student_outputs, teacher_outputs)
            chsim_loss += weight * loss
        return chsim_loss


class DISTLoss(nn.Module):
    """
    A loss module for Knowledge Distillation from A Stronger Teacher (DIST).
    Referred to https://github.com/hunto/image_classification_sota/blob/main/lib/models/losses/dist_kd.py

    Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu: `"Knowledge Distillation from A Stronger Teacher" <https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html>`_ @ NeurIPS 2022 (2022)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :param beta: balancing factor for inter-loss.
    :type beta: float
    :param gamma: balancing factor for intra-loss.
    :type gamma: float
    :param tau: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type tau: float
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io, beta=1.0, gamma=1.0, tau=1.0, eps=1e-08, **kwargs):
        super().__init__()
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = eps

    @staticmethod
    def pearson_correlation(y_s, y_t, eps):
        return cosine_similarity(y_s - y_s.mean(1).unsqueeze(1), y_t - y_t.mean(1).unsqueeze(1), eps=eps)

    def inter_class_relation(self, y_s, y_t):
        return 1 - self.pearson_correlation(y_s, y_t, self.eps).mean()

    def intra_class_relation(self, y_s, y_t):
        return self.inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_logits = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_logits = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        y_s = (student_logits / self.tau).softmax(dim=1)
        y_t = (teacher_logits / self.tau).softmax(dim=1)
        inter_loss = self.tau ** 2 * self.inter_class_relation(y_s, y_t)
        intra_loss = self.tau ** 2 * self.intra_class_relation(y_s, y_t)
        loss = self.beta * inter_loss + self.gamma * intra_loss
        return loss


class SRDLoss(nn.Module):
    """
    A loss module for Understanding the Role of the Projector in Knowledge Distillation.
    Referred to https://github.com/roymiles/Simple-Recipe-Distillation/blob/main/imagenet/torchdistill/losses/single.py

    Roy Miles, Krystian Mikolajczyk: `"Understanding the Role of the Projector in Knowledge Distillation" <https://arxiv.org/abs/2303.11098>`_ @ AAAI 2024 (2024)

    :param student_feature_module_path: student model's feature module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.SRDModelWrapper`.
    :type student_feature_module_path: str
    :param student_feature_module_io: 'input' or 'output' of the feature module in the student model.
    :type student_feature_module_io: str
    :param teacher_feature_module_path: teacher model's feature module path in an auxiliary wrapper :class:`torchdistill.models.wrapper.SRDModelWrapper`.
    :type teacher_feature_module_path: str
    :param teacher_feature_module_io: 'input' or 'output' of the feature module in the teacher model.
    :type teacher_feature_module_io: str
    :param student_linear_module_path: student model's linear module path.
    :type student_linear_module_path: str
    :param student_linear_module_io: 'input' or 'output' of the linear module in the student model.
    :type student_linear_module_io: str
    :param teacher_linear_module_path: teacher model's linear module path.
    :type teacher_linear_module_path: str
    :param teacher_linear_module_io: 'input' or 'output' of the linear module in the teacher model.
    :type teacher_linear_module_io: str
    :param exponent: exponent for feature distillation loss.
    :type exponent: float
    :param temperature: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type temperature: float
    :param reduction: loss reduction type.
    :type reduction: str or None
    """

    def __init__(self, student_feature_module_path, student_feature_module_io, teacher_feature_module_path, teacher_feature_module_io, student_linear_module_path, student_linear_module_io, teacher_linear_module_path, teacher_linear_module_io, exponent=1.0, temperature=1.0, reduction='batchmean', **kwargs):
        super().__init__()
        self.student_feature_module_path = student_feature_module_path
        self.student_feature_module_io = student_feature_module_io
        self.teacher_feature_module_path = teacher_feature_module_path
        self.teacher_feature_module_io = teacher_feature_module_io
        self.student_linear_module_path = student_linear_module_path
        self.student_linear_module_io = student_linear_module_io
        self.teacher_linear_module_path = teacher_linear_module_path
        self.teacher_linear_module_io = teacher_linear_module_io
        self.exponent = exponent
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction=reduction)

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        student_features = student_io_dict[self.student_feature_module_path][self.student_feature_module_io]
        teacher_features = teacher_io_dict[self.teacher_feature_module_path][self.teacher_feature_module_io]
        diff_features = torch.abs(student_features - teacher_features)
        feat_distill_loss = torch.log(diff_features.pow(self.exponent).sum())
        student_logits = student_io_dict[self.student_linear_module_path][self.student_linear_module_io]
        teacher_logits = teacher_io_dict[self.teacher_linear_module_path][self.teacher_linear_module_io]
        kl_loss = self.criterion(torch.log_softmax(student_logits / self.temperature, dim=1), torch.softmax(teacher_logits / self.temperature, dim=1))
        loss = 2 * feat_distill_loss + kl_loss
        return loss


class LogitStdKDLoss(nn.KLDivLoss):
    """
    A standard knowledge distillation (KD) loss module with logits standardization.

    Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, Xiaochun Cao: `"Logit Standardization in Knowledge Distillation" <https://arxiv.org/abs/2403.01427>`_ @ CVPR 2024 (2024)

    :param student_module_path: student model's logit module path.
    :type student_module_path: str
    :param student_module_io: 'input' or 'output' of the module in the student model.
    :type student_module_io: str
    :param teacher_module_path: teacher model's logit module path.
    :type teacher_module_path: str
    :param teacher_module_io: 'input' or 'output' of the module in the teacher model.
    :type teacher_module_io: str
    :param temperature: hyperparameter :math:`\\tau` to soften class-probability distributions.
    :type temperature: float
    :param eps: value added to the denominator for numerical stability.
    :type eps: float
    :param alpha: balancing factor for :math:`L_{CE}`, cross-entropy.
    :type alpha: float
    :param beta: balancing factor (default: :math:`1 - \\alpha`) for :math:`L_{KL}`, KL divergence between class-probability distributions softened by :math:`\\tau`.
    :type beta: float or None
    :param reduction: ``reduction`` for KLDivLoss. If ``reduction`` = 'batchmean', CrossEntropyLoss's ``reduction`` will be 'mean'.
    :type reduction: str or None
    """

    def __init__(self, student_module_path, student_module_io, teacher_module_path, teacher_module_io, temperature, eps=1e-07, alpha=None, beta=None, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.student_module_path = student_module_path
        self.student_module_io = student_module_io
        self.teacher_module_path = teacher_module_path
        self.teacher_module_io = teacher_module_io
        self.temperature = temperature
        self.eps = eps
        self.alpha = alpha
        self.beta = 1 - alpha if beta is None else beta
        cel_reduction = 'mean' if reduction == 'batchmean' else reduction
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=cel_reduction, **kwargs)

    def standardize(self, logits):
        return (logits - logits.mean(dim=-1, keepdims=True)) / (self.eps + logits.std(dim=-1, keepdims=True))

    def forward(self, student_io_dict, teacher_io_dict, targets=None, *args, **kwargs):
        student_logits = student_io_dict[self.student_module_path][self.student_module_io]
        teacher_logits = teacher_io_dict[self.teacher_module_path][self.teacher_module_io]
        soft_loss = super().forward(torch.log_softmax(self.standardize(student_logits) / self.temperature, dim=1), torch.softmax(self.standardize(teacher_logits) / self.temperature, dim=1))
        if self.alpha is None or self.alpha == 0 or targets is None:
            return soft_loss
        hard_loss = self.cross_entropy_loss(student_logits, targets)
        return self.alpha * hard_loss + self.beta * self.temperature ** 2 * soft_loss


class ConvReg(nn.Sequential):
    """
    `A convolutional regression for FitNets used in "Contrastive Representation Distillation" (CRD) <https://github.com/HobbitLong/RepDistiller/blob/34557d27282c83d49cff08b594944cf9570512bb/models/util.py#L131-L154>`_

    :param num_input_channels: ``in_channels`` for Conv2d.
    :type num_input_channels: int
    :param num_output_channels: ``out_channels`` for Conv2d.
    :type num_output_channels: int
    :param kernel_size: ``kernel_size`` for Conv2d.
    :type kernel_size: (int, int) or int
    :param stride: ``stride`` for Conv2d.
    :type stride: int
    :param padding: ``padding`` for Conv2d.
    :type padding: int
    :param uses_relu: if True, uses ReLU as the last module.
    :type uses_relu: bool
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_relu=True):
        module_dict = OrderedDict()
        module_dict['conv'] = nn.Conv2d(num_input_channels, num_output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        module_dict['bn'] = nn.BatchNorm2d(num_output_channels)
        if uses_relu:
            module_dict['relu'] = nn.ReLU(inplace=True)
        super().__init__(module_dict)


class DenseNet4Cifar(nn.Module):
    """
    DenseNet-BC model for CIFAR datasets. Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
    for CIFAR datasets, referring to https://github.com/liuzhuang13/DenseNet

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger: `"Densely Connected Convolutional Networks" <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html>`_ @ CVPR 2017 (2017).

    :param growth_rate: number of filters to add each layer (`k` in paper).
    :type growth_rate: int
    :param block_config: three numbers of layers in each pooling block.
    :type block_config: list[int]
    :param num_init_features: number of filters to learn in the first convolution layer.
    :type num_init_features: int
    :param bn_size: multiplicative factor for number of bottleneck layers. (i.e. bn_size * k features in the bottleneck layer)
    :type bn_size: int
    :param drop_rate: dropout rate after each dense layer.
    :type drop_rate: float
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param memory_efficient: if True, uses checkpointing. Much more memory efficient, but slower. Refer to `"the paper" <https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html>`_ for details.
    :type memory_efficient: bool
    """

    def __init__(self, growth_rate: 'int'=32, block_config: 'Tuple[int, int, int]'=(12, 12, 12), num_init_features: 'int'=64, bn_size: 'int'=4, drop_rate: 'float'=0, num_classes: 'int'=10, memory_efficient: 'bool'=False) ->None:
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, memory_efficient=memory_efficient)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: 'Tensor') ->Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class ResNet4Cifar(nn.Module):
    """
    ResNet model for CIFAR datasets. Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    for CIFAR datasets, referring to https://github.com/facebookarchive/fb.resnet.torch

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: `"Deep Residual Learning for Image Recognition" <https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html>`_ @ CVPR 2016 (2016).

    :param block: block class.
    :type block: BasicBlock
    :param layers: three numbers of layers in each pooling block.
    :type layers: list[int]
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param zero_init_residual: if True, zero-initializes the last BN in each residual branch
    :type zero_init_residual: bool
    :param groups: ``groups`` for Conv2d.
    :type groups: int
    :param width_per_group: base width for Conv2d.
    :type width_per_group: int
    :param replace_stride_with_dilation: indicates if we should replace the 2x2 stride with a dilated convolution instead.
    :type replace_stride_with_dilation: list[bool] or None
    :param norm_layer: normalization module class or callable object.
    :type norm_layer: typing.Callable or nn.Module or None
    """

    def __init__(self, block: 'Type[Union[BasicBlock]]', layers: 'List[int]', num_classes: 'int'=10, zero_init_residual: 'bool'=False, groups: 'int'=1, width_per_group: 'int'=64, replace_stride_with_dilation: 'Optional[List[bool]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: 'Type[Union[BasicBlock]]', planes: 'int', blocks: 'int', stride: 'int'=1, dilate: 'bool'=False) ->nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: 'Tensor') ->Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        return self._forward_impl(x)


class WideBasicBlock(nn.Module):
    """
    A basic block of Wide ResNet for CIFAR datasets.

    :param in_planes: number of input feature planes.
    :type in_planes: int
    :param planes: number of output feature planes.
    :type planes: int
    :param dropout_rate: dropout rate.
    :type dropout_rate: float
    :param stride: stride for Conv2d.
    :type stride: int
    """

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


class WideResNet4Cifar(nn.Module):
    """
    Wide ResNet (WRN) model for CIFAR datasets. Refactored https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    for CIFAR datasets, referring to https://github.com/szagoruyko/wide-residual-networks

    Sergey Zagoruyko, Nikos Komodakis: `"Wide Residual Networks" <https://bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html>`_ @ BMVC 2016 (2016)

    :param depth: depth.
    :type depth: int
    :param k: widening factor.
    :type k: int
    :param dropout_p: dropout rate.
    :type dropout_p: float
    :param block: block class.
    :type block: WideBasicBlock
    :param num_classes: number of classification classes.
    :type num_classes: int
    :param norm_layer: normalization module class or callable object.
    :type norm_layer: typing.Callable or nn.Module or None
    """

    def __init__(self, depth, k, dropout_p, block, num_classes, norm_layer=None):
        super().__init__()
        n = (depth - 4) / 6
        stage_sizes = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, stage_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_wide_layer(block, stage_sizes[1], n, dropout_p, 1)
        self.layer2 = self._make_wide_layer(block, stage_sizes[2], n, dropout_p, 2)
        self.layer3 = self._make_wide_layer(block, stage_sizes[3], n, dropout_p, 2)
        self.bn1 = norm_layer(stage_sizes[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_sizes[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _forward_impl(self, x: 'Tensor') ->Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        return self._forward_impl(x)


class BottleneckBase(nn.Module):

    def __init__(self, encoder, decoder, compressor=None, decompressor=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.compressor = compressor
        self.decompressor = decompressor

    def forward(self, x):
        z = self.encoder(x)
        if self.compressor is not None:
            z = self.compressor(z)
        if self.decompressor is not None:
            z = self.decompressor(z)
        return self.decoder(z)


class Bottleneck4DenseNet(BottleneckBase):
    """
    Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems
    """

    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None):
        modules = [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False), nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=2, stride=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False), nn.AvgPool2d(kernel_size=2, stride=2)]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class CustomDenseNet(nn.Module):

    def __init__(self, bottleneck, short_feature_names, org_densenet):
        super().__init__()
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_features_set = set(short_feature_names)
        if 'classifier' in short_features_set:
            short_features_set.remove('classifier')
        for child_name, child_module in org_densenet.features.named_children():
            if child_name in short_features_set:
                module_dict[child_name] = child_module
        self.features = nn.Sequential(module_dict)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = org_densenet.classifier

    def forward(self, x):
        z = self.features(x)
        z = self.relu(z)
        z = self.adaptive_avgpool(z)
        z = torch.flatten(z, 1)
        return self.classifier(z)


class Bottleneck4Inception3(BottleneckBase):
    """
    Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems
    """

    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None):
        modules = [nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False), nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 256, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=2, stride=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 192, kernel_size=2, stride=1, bias=False), nn.AvgPool2d(kernel_size=2, stride=1)]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class CustomInception3(nn.Sequential):

    def __init__(self, bottleneck, short_module_names, org_model):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        child_name_list = list()
        for child_name, child_module in org_model.named_children():
            if child_name in short_module_set:
                if len(child_name_list) > 0 and child_name_list[-1] == 'Conv2d_2b_3x3' and child_name == 'Conv2d_3b_1x1':
                    module_dict['maxpool1'] = nn.MaxPool2d(kernel_size=3, stride=2)
                    child_name_list.append('maxpool1')
                elif len(child_name_list) > 0 and child_name_list[-1] == 'Conv2d_4a_3x3' and child_name == 'Mixed_5b':
                    module_dict['maxpool2'] = nn.MaxPool2d(kernel_size=3, stride=2)
                    child_name_list.append('maxpool2')
                elif child_name == 'fc':
                    module_dict['adaptive_avgpool'] = nn.AdaptiveAvgPool2d((1, 1))
                    module_dict['dropout'] = nn.Dropout()
                    module_dict['flatten'] = nn.Flatten(1)
                module_dict[child_name] = child_module
                child_name_list.append(child_name)
        super().__init__(module_dict)


class Bottleneck4ResNet(BottleneckBase):
    """
    Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems
    """

    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None):
        modules = [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False), nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False), nn.AvgPool2d(kernel_size=2, stride=1)]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class CustomResNet(nn.Sequential):

    def __init__(self, bottleneck, short_module_names, org_resnet):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        for child_name, child_module in org_resnet.named_children():
            if child_name in short_module_set:
                if child_name == 'fc':
                    module_dict['flatten'] = nn.Flatten(1)
                module_dict[child_name] = child_module
        super().__init__(module_dict)


class Bottleneck4SmallResNet(BottleneckBase):
    """
    Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks
    """

    def __init__(self, bottleneck_channel, compressor=None, decompressor=None):
        encoder = nn.Sequential(nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False))
        decoder = nn.Sequential(nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 128, kernel_size=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 64, kernel_size=2, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 64, kernel_size=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


class Bottleneck4LargeResNet(BottleneckBase):
    """
    Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks
    """

    def __init__(self, bottleneck_channel, compressor=None, decompressor=None):
        encoder = nn.Sequential(nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False))
        decoder = nn.Sequential(nn.BatchNorm2d(bottleneck_channel), nn.ReLU(inplace=True), nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False), nn.BatchNorm2d(64), nn.Conv2d(64, 128, kernel_size=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 256, kernel_size=2, bias=False), nn.BatchNorm2d(256), nn.Conv2d(256, 256, kernel_size=2, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class AuxiliaryModelWrapper(nn.Module):
    """
    An abstract auxiliary model wrapper.

    :meth:`forward`, :meth:`secondary_forward`, and :meth:`post_epoch_process` should be overridden by all subclasses.
    """

    def __init__(self):
        super().__init__()

    def secondary_forward(self, *args, **kwargs):
        pass

    def post_epoch_process(self, *args, **kwargs):
        pass


class EmptyModule(AuxiliaryModelWrapper):
    """
    An empty auxiliary model wrapper. This module returns input as output and is useful when you want to replace
    your teacher/student model with an empty model for saving inference time.
    e.g., Multi-stage knowledge distillation may have some stages that do not require either teacher or student models.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return args[0] if isinstance(args, tuple) and len(args) == 1 else args


class Paraphraser4FactorTransfer(nn.Module):
    """
    Paraphraser for factor transfer (FT). This module is used at the 1st and 2nd stages of FT method.

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param k: paraphrase rate.
    :type k: float
    :param num_input_channels: number of input channels.
    :type num_input_channels: int
    :param kernel_size: ``kernel_size`` for Conv2d.
    :type kernel_size: int
    :param stride: ``stride`` for Conv2d.
    :type stride: int
    :param padding: ``padding`` for Conv2d.
    :type padding: int
    :param uses_bn: if True, uses BatchNorm2d.
    :type uses_bn: bool
    :param uses_decoder: if True, uses decoder in :meth:`forward`.
    :type uses_decoder: bool
    """

    @staticmethod
    def make_tail_modules(num_output_channels, uses_bn):
        leaky_relu = nn.LeakyReLU(0.1)
        if uses_bn:
            return [nn.BatchNorm2d(num_output_channels), leaky_relu]
        return [leaky_relu]

    @classmethod
    def make_enc_modules(cls, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_bn):
        return [nn.Conv2d(num_input_channels, num_output_channels, kernel_size, stride=stride, padding=padding), *cls.make_tail_modules(num_output_channels, uses_bn)]

    @classmethod
    def make_dec_modules(cls, num_input_channels, num_output_channels, kernel_size, stride, padding, uses_bn):
        return [nn.ConvTranspose2d(num_input_channels, num_output_channels, kernel_size, stride=stride, padding=padding), *cls.make_tail_modules(num_output_channels, uses_bn)]

    def __init__(self, k, num_input_channels, kernel_size=3, stride=1, padding=1, uses_bn=True, uses_decoder=True):
        super().__init__()
        self.paraphrase_rate = k
        num_enc_output_channels = int(num_input_channels * k)
        self.encoder = nn.Sequential(*self.make_enc_modules(num_input_channels, num_input_channels, kernel_size, stride, padding, uses_bn), *self.make_enc_modules(num_input_channels, num_enc_output_channels, kernel_size, stride, padding, uses_bn), *self.make_enc_modules(num_enc_output_channels, num_enc_output_channels, kernel_size, stride, padding, uses_bn))
        self.decoder = nn.Sequential(*self.make_dec_modules(num_enc_output_channels, num_enc_output_channels, kernel_size, stride, padding, uses_bn), *self.make_dec_modules(num_enc_output_channels, num_input_channels, kernel_size, stride, padding, uses_bn), *self.make_dec_modules(num_input_channels, num_input_channels, kernel_size, stride, padding, uses_bn))
        self.uses_decoder = uses_decoder

    def forward(self, z):
        if self.uses_decoder:
            return self.decoder(self.encoder(z))
        return self.encoder(z)


class Translator4FactorTransfer(nn.Sequential):
    """
    Translator for factor transfer (FT). This module is used at the 2nd stage of FT method.
    Note that "the student translator has the same three convolution layers as the paraphraser".

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param num_input_channels: number of input channels.
    :type num_input_channels: int
    :param kernel_size: ``kernel_size`` for Conv2d.
    :type kernel_size: int
    :param stride: ``stride`` for Conv2d.
    :type stride: int
    :param padding: ``padding`` for Conv2d.
    :type padding: int
    :param uses_bn: if True, uses BatchNorm2d.
    :type uses_bn: bool
    """

    def __init__(self, num_input_channels, num_output_channels, kernel_size=3, stride=1, padding=1, uses_bn=True):
        super().__init__(*Paraphraser4FactorTransfer.make_enc_modules(num_input_channels, num_input_channels, kernel_size, stride, padding, uses_bn), *Paraphraser4FactorTransfer.make_enc_modules(num_input_channels, num_output_channels, kernel_size, stride, padding, uses_bn), *Paraphraser4FactorTransfer.make_enc_modules(num_output_channels, num_output_channels, kernel_size, stride, padding, uses_bn))


AUXILIARY_MODEL_WRAPPER_DICT = dict()


def get_auxiliary_model_wrapper(key, *args, **kwargs):
    """
    Gets an auxiliary model wrapper from the auxiliary model wrapper registry.

    :param key: model key.
    :type key: str
    :return: auxiliary model wrapper.
    :rtype: nn.Module
    """
    if key in AUXILIARY_MODEL_WRAPPER_DICT:
        return AUXILIARY_MODEL_WRAPPER_DICT[key](*args, **kwargs)
    raise ValueError('No auxiliary model wrapper `{}` registered'.format(key))


def add_submodule(module, module_path, module_dict):
    """
    Recursively adds submodules to `module_dict`.

    :param module: module.
    :type module: nn.Module
    :param module_path: module path.
    :type module_path: str
    :param module_dict: module dict.
    :type module_dict: nn.ModuleDict or dict
    """
    module_names = module_path.split('.')
    module_name = module_names.pop(0)
    if len(module_names) == 0:
        if module_name in module_dict:
            raise KeyError('module_name `{}` is already used.'.format(module_name))
        module_dict[module_name] = module
        return
    next_module_path = '.'.join(module_names)
    sub_module_dict = module_dict.get(module_name, None)
    if module_name not in module_dict:
        sub_module_dict = OrderedDict()
        module_dict[module_name] = sub_module_dict
    add_submodule(module, next_module_path, sub_module_dict)


def build_sequential_container(module_dict):
    """
    Builds sequential container (nn.Sequential) from ``module_dict``.

    :param module_dict: module dict to build sequential to build a sequential container.
    :type module_dict: nn.ModuleDict or collections.OrderedDict
    :return: sequential container.
    :rtype: nn.Sequential
    """
    for key in module_dict.keys():
        value = module_dict[key]
        if isinstance(value, OrderedDict):
            value = build_sequential_container(value)
            module_dict[key] = value
        elif not isinstance(value, Module):
            raise ValueError('module type `{}` is not expected'.format(type(value)))
    return Sequential(module_dict)


def freeze_module_params(module):
    """
    Freezes parameters by setting requires_grad=False for all the parameters.

    :param module: module.
    :type module: nn.Module
    """
    if isinstance(module, Module):
        for param in module.parameters():
            param.requires_grad = False
    elif isinstance(module, Parameter):
        module.requires_grad = False


ADAPTATION_MODULE_DICT = dict()


def get_adaptation_module(key, *args, **kwargs):
    """
    Gets an adaptation module from the adaptation module registry.

    :param key: model key.
    :type key: str
    :return: adaptation module.
    :rtype: nn.Module
    """
    if key in ADAPTATION_MODULE_DICT:
        return ADAPTATION_MODULE_DICT[key](*args, **kwargs)
    elif key in MODULE_DICT:
        return MODULE_DICT[key](*args, **kwargs)
    raise ValueError('No adaptation module `{}` registered'.format(key))


def get_module(root_module, module_path):
    """
    Gets a module specified by ``module_path``.

    :param root_module: module.
    :type root_module: nn.Module
    :param module_path: module path for extracting the module from ``root_module``.
    :type module_path: str
    :return: module extracted from ``root_module`` if exists.
    :rtype: nn.Module or None
    """
    module_names = module_path.split('.')
    module = root_module
    for module_name in module_names:
        if not hasattr(module, module_name):
            if isinstance(module, (DataParallel, DistributedDataParallel)):
                module = module.module
                if not hasattr(module, module_name):
                    if isinstance(module, Sequential) and module_name.lstrip('-').isnumeric():
                        module = module[int(module_name)]
                    else:
                        logger.warning('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path, type(root_module).__name__))
                else:
                    module = getattr(module, module_name)
            elif isinstance(module, (Sequential, ModuleList)) and module_name.lstrip('-').isnumeric():
                module = module[int(module_name)]
            else:
                logger.warning('`{}` of `{}` could not be reached in `{}`'.format(module_name, module_path, type(root_module).__name__))
                return None
        else:
            module = getattr(module, module_name)
    return module


def redesign_model(org_model, model_config, model_label, model_type='original'):
    """
    Redesigns ``org_model`` and returns a new separate model e.g.,

    * prunes some modules from ``org_model``,
    * freezes parameters of some modules in ``org_model``, and
    * adds adaptation module(s) to ``org_model`` as a new separate model.

    .. note::
        The parameters and states of modules in ``org_model`` will be kept in a new redesigned model.

    :param org_model: original model to be redesigned.
    :type org_model: nn.Module
    :param model_config: configuration to redesign ``org_model``.
    :type model_config: dict
    :param model_label: model label (e.g., 'teacher', 'student') to be printed just for debugging purpose.
    :type model_label: str
    :param model_type: model type (e.g., 'original', name of model class, etc) to be printed just for debugging purpose.
    :type model_type: str
    :return: redesigned model.
    :rtype: nn.Module
    """
    frozen_module_path_set = set(model_config.get('frozen_modules', list()))
    module_paths = model_config.get('sequential', list())
    if not isinstance(module_paths, list) or len(module_paths) == 0:
        logger.info('Using the {} model'.format(model_type))
        if len(frozen_module_path_set) > 0:
            logger.info('Frozen module(s): {}'.format(frozen_module_path_set))
        isinstance_str = 'instance('
        for frozen_module_path in frozen_module_path_set:
            if frozen_module_path.startswith(isinstance_str) and frozen_module_path.endswith(')'):
                target_cls = nn.__dict__[frozen_module_path[len(isinstance_str):-1]]
                for m in org_model.modules():
                    if isinstance(m, target_cls):
                        freeze_module_params(m)
            else:
                module = get_module(org_model, frozen_module_path)
                freeze_module_params(module)
        return org_model
    logger.info('Redesigning the {} model with {}'.format(model_label, module_paths))
    if len(frozen_module_path_set) > 0:
        logger.info('Frozen module(s): {}'.format(frozen_module_path_set))
    module_dict = OrderedDict()
    adaptation_dict = model_config.get('adaptations', dict())
    for frozen_module_path in frozen_module_path_set:
        module = get_module(org_model, frozen_module_path)
        freeze_module_params(module)
    for module_path in module_paths:
        if module_path.startswith('+'):
            module_path = module_path[1:]
            adaptation_config = adaptation_dict[module_path]
            module = get_adaptation_module(adaptation_config['key'], **adaptation_config['kwargs'])
        else:
            module = get_module(org_model, module_path)
        if module_path in frozen_module_path_set:
            freeze_module_params(module)
        add_submodule(module, module_path, module_dict)
    return build_sequential_container(module_dict)


def build_auxiliary_model_wrapper(model_config, **kwargs):
    """
    Builds an auxiliary model wrapper for either teacher or student models.

    :param model_config: configuration to build the auxiliary model wrapper. Should contain either 'teacher_model' or `student_model'.
    :type model_config: dict
    :return: auxiliary model wrapper.
    :rtype: nn.Module
    """
    auxiliary_model_wrapper_config = model_config.get('auxiliary_model_wrapper', dict())
    auxiliary_model_wrapper_key = auxiliary_model_wrapper_config.get('key', None)
    if auxiliary_model_wrapper_key is None:
        return None
    auxiliary_model_wrapper_kwargs = auxiliary_model_wrapper_config.get('kwargs', None)
    if auxiliary_model_wrapper_kwargs is None:
        auxiliary_model_wrapper_kwargs = dict()
    elif 'teacher_model' in kwargs:
        kwargs['teacher_model'] = redesign_model(kwargs['teacher_model'], auxiliary_model_wrapper_config, 'teacher', 'pre-auxiliary')
    elif 'student_model' in kwargs:
        kwargs['student_model'] = redesign_model(kwargs['student_model'], auxiliary_model_wrapper_config, 'student', 'pre-auxiliary')
    return get_auxiliary_model_wrapper(auxiliary_model_wrapper_key, **kwargs, **auxiliary_model_wrapper_kwargs)


def check_if_wrapped(model):
    """
    Checks if a given model is wrapped by DataParallel or DistributedDataParallel.

    :param model: model.
    :type model: nn.Module
    :return: True if `model` is wrapped by either DataParallel or DistributedDataParallel.
    :rtype: bool
    """
    return isinstance(model, (DataParallel, DistributedDataParallel))


def load_module_ckpt(module, map_location, ckpt_file_path):
    """
    Loads checkpoint for ``module``.

    :param module: module to load checkpoint.
    :type module: nn.Module
    :param map_location: ``map_location`` for torch.load.
    :type map_location: torch.device or str or dict or typing.Callable
    :param ckpt_file_path: file path to load checkpoint.
    :type ckpt_file_path: str
    """
    state_dict = torch.load(ckpt_file_path, map_location=map_location)
    if check_if_wrapped(module):
        module.module.load_state_dict(state_dict)
    else:
        module.load_state_dict(state_dict)


def is_dist_avail_and_initialized():
    """
    Checks if distributed model is available and initialized.

    :return: True if distributed mode is available and initialized.
    :rtype: bool
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """
    Gets the rank of the current process in the provided ``group`` or the default group if none was provided.

    :return: rank of the current process in the provided ``group`` or the default group if none was provided.
    :rtype: int
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Checks if this is the main process.

    :return: True if this is the main process.
    :rtype: bool
    """
    return get_rank() == 0


def make_parent_dirs(file_path):
    """
    Makes parent directories.

    :param file_path: file path
    :type file_path: str
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_on_master(*args, **kwargs):
    """
    Use `torch.save` for `args` if this is the main process.

    :return: True if this is the main process.
    :rtype: bool
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def save_module_ckpt(module, ckpt_file_path):
    """
    Saves checkpoint of ``module``'s state dict.

    :param module: module to load checkpoint.
    :type module: nn.Module
    :param ckpt_file_path: file path to save checkpoint.
    :type ckpt_file_path: str
    """
    if is_main_process():
        make_parent_dirs(ckpt_file_path)
    state_dict = module.module.state_dict() if check_if_wrapped(module) else module.state_dict()
    save_on_master(state_dict, ckpt_file_path)


def get_frozen_param_names(module):
    """
    Gets collection of frozen parameter names.

    :param module: module.
    :type module: nn.Module
    :return: names of frozen parameters.
    :rtype: list[str]
    """
    return [name for name, param in module.named_parameters() if not param.requires_grad]


def get_updatable_param_names(module):
    """
    Gets collection of updatable parameter names.

    :param module: module.
    :type module: nn.Module
    :return: names of updatable parameters.
    :rtype: list[str]
    """
    return [name for name, param in module.named_parameters() if param.requires_grad]


def wrap_if_distributed(module, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
    """
    Wraps ``module`` with DistributedDataParallel if ``distributed`` = True and ``module`` has any updatable parameters.

    :param module: module to be wrapped.
    :type module: nn.Module
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    :return: wrapped module if ``distributed`` = True and it contains any updatable parameters.
    :rtype: nn.Module
    """
    module
    if distributed and len(get_updatable_param_names(module)) > 0:
        any_frozen = len(get_frozen_param_names(module)) > 0
        if find_unused_parameters is None:
            find_unused_parameters = any_frozen
        return DistributedDataParallel(module, device_ids=device_ids, find_unused_parameters=find_unused_parameters, **kwargs)
    return module


class Teacher4FactorTransfer(AuxiliaryModelWrapper):
    """
    An auxiliary teacher model wrapper for factor transfer (FT), including paraphraser :class:`Paraphraser4FactorTransfer`.

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param teacher_model: teacher model.
    :type teacher_model: nn.Module
    :param minimal: ``model_config`` for :meth:`build_auxiliary_model_wrapper` if you want to.
    :type minimal: dict or None
    :param input_module_path: path of module whose output is used as input to paraphraser.
    :type input_module_path: str
    :param paraphraser_kwargs: kwargs to instantiate :class:`Paraphraser4FactorTransfer`.
    :type paraphraser_kwargs: dict
    :param uses_decoder: ``uses_decoder`` for :class:`Paraphraser4FactorTransfer`.
    :type uses_decoder: bool
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, teacher_model, minimal, input_module_path, paraphraser_kwargs, paraphraser_ckpt, uses_decoder, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
        super().__init__()
        if minimal is None:
            minimal = dict()
        auxiliary_teacher_model_wrapper = build_auxiliary_model_wrapper(minimal, teacher_model=teacher_model)
        model_type = 'original'
        teacher_ref_model = teacher_model
        if auxiliary_teacher_model_wrapper is not None:
            teacher_ref_model = auxiliary_teacher_model_wrapper
            model_type = type(teacher_ref_model).__name__
        self.teacher_model = redesign_model(teacher_ref_model, minimal, 'teacher', model_type)
        self.input_module_path = input_module_path
        paraphraser = Paraphraser4FactorTransfer(uses_decoder=uses_decoder, **paraphraser_kwargs)
        self.paraphraser = wrap_if_distributed(paraphraser, device, device_ids, distributed, find_unused_parameters=find_unused_parameters)
        self.ckpt_file_path = paraphraser_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(self.paraphraser, map_location, self.ckpt_file_path)
        self.uses_decoder = uses_decoder

    def forward(self, *args):
        with torch.no_grad():
            return self.teacher_model(*args)

    def secondary_forward(self, io_dict):
        if self.uses_decoder and not self.paraphraser.training:
            self.paraphraser.train()
        self.paraphraser(io_dict[self.input_module_path]['output'])

    def post_epoch_process(self, *args, **kwargs):
        save_module_ckpt(self.paraphraser, self.ckpt_file_path)


class Student4FactorTransfer(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for factor transfer (FT), including translator :class:`Translator4FactorTransfer`.

    Jangho Kim, Seonguk Park, Nojun Kwak: `"Paraphrasing Complex Network: Network Compression via Factor Transfer" <https://papers.neurips.cc/paper_files/paper/2018/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html>`_ @ NeurIPS 2018 (2018)

    :param student_model: student model.
    :type student_model: nn.Module
    :param input_module_path: path of module whose output is used as input to paraphraser.
    :type input_module_path: str
    :param translator_kwargs: kwargs to instantiate :class:`Translator4FactorTransfer`.
    :type translator_kwargs: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, student_model, input_module_path, translator_kwargs, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters=find_unused_parameters)
        self.input_module_path = input_module_path
        self.translator = wrap_if_distributed(Translator4FactorTransfer(**translator_kwargs), device, device_ids, distributed, find_unused_parameters=find_unused_parameters)

    def forward(self, *args):
        return self.student_model(*args)

    def secondary_forward(self, io_dict):
        self.translator(io_dict[self.input_module_path]['output'])


class Connector4DAB(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper with connector for distillation of activation boundaries (DAB).

    Byeongho Heo, Minsik Lee, Sangdoo Yun, Jin Young Choi: `"Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons" <https://ojs.aaai.org/index.php/AAAI/article/view/4264>`_ @ AAAI 2019 (2019)

    :param student_model: student model.
    :type student_model: nn.Module
    :param connectors: connector keys and configurations.
    :type connectors: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    @staticmethod
    def build_connector(conv2d_kwargs, bn2d_kwargs=None):
        module_list = [nn.Conv2d(**conv2d_kwargs)]
        if bn2d_kwargs is not None and len(bn2d_kwargs) > 0:
            module_list.append(nn.BatchNorm2d(**bn2d_kwargs))
        return nn.Sequential(*module_list)

    def __init__(self, student_model, connectors, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        io_path_pairs = list()
        self.connector_dict = nn.ModuleDict()
        for connector_key, connector_config in connectors.items():
            connector = self.build_connector(connector_config['conv2d_kwargs'], connector_config.get('bn2d_kwargs', None))
            self.connector_dict[connector_key] = wrap_if_distributed(connector, device, device_ids, distributed, find_unused_parameters)
            io_path_pairs.append((connector_key, connector_config['io'], connector_config['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def secondary_forward(self, io_dict):
        for connector_key, io_type, module_path in self.io_path_pairs:
            self.connector_dict[connector_key](io_dict[module_path][io_type])


class Regressor4VID(nn.Module):
    """
    An auxiliary module for variational information distillation (VID).

    Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai: `"Variational Information Distillation for Knowledge Transfer" <https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param in_channels: number of input channels for the first convolution layer.
    :type in_channels: int
    :param mid_channels: number of output/input channels for the first/second convolution layer.
    :type mid_channels: int
    :param out_channels: number of output channels for the third convolution layer.
    :type out_channels: int
    :param eps: eps.
    :type eps: float
    :param init_pred_var: minimum variance introduced for numerical stability.
    :type init_pred_var: float
    """

    def __init__(self, in_channels, middle_channels, out_channels, eps, init_pred_var, **kwargs):
        super().__init__()
        self.regressor = nn.Sequential(nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, middle_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.soft_plus_param = nn.Parameter(np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(out_channels))
        self.eps = eps
        self.init_pred_var = init_pred_var

    def forward(self, student_feature_map):
        pred_mean = self.regressor(student_feature_map)
        pred_var = torch.log(1.0 + torch.exp(self.soft_plus_param)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        return pred_mean, pred_var


class VariationalDistributor4VID(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for variational information distillation (VID), including translator :class:`Regressor4VID`.

    Sungsoo Ahn, Shell Xu Hu, Andreas Damianou, Neil D. Lawrence, Zhenwen Dai: `"Variational Information Distillation for Knowledge Transfer" <https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param student_model: student model.
    :type student_model: nn.Module
    :param in_channels: number of input channels for the first convolution layer.
    :type in_channels: int
    :param regressors: regressor keys and configurations.
    :type regressors: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, student_model, regressors, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        io_path_pairs = list()
        self.regressor_dict = nn.ModuleDict()
        for regressor_key, regressor_config in regressors.items():
            regressor = Regressor4VID(**regressor_config['kwargs'])
            self.regressor_dict[regressor_key] = wrap_if_distributed(regressor, device, device_ids, distributed, find_unused_parameters)
            io_path_pairs.append((regressor_key, regressor_config['io'], regressor_config['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def secondary_forward(self, io_dict):
        for regressor_key, io_type, module_path in self.io_path_pairs:
            self.regressor_dict[regressor_key](io_dict[module_path][io_type])


class Linear4CCKD(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for correlation congruence for knowledge distillation (CCKD).
    Fully-connected layers cope with a mismatch of feature representations of teacher and student models.

    Baoyun Peng, Xiao Jin, Jiaheng Liu, Dongsheng Li, Yichao Wu, Yu Liu, Shunfeng Zhou, Zhaoning Zhang: `"Correlation Congruence for Knowledge Distillation" <https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Correlation_Congruence_for_Knowledge_Distillation_ICCV_2019_paper.html>`_ @ ICCV 2019 (2019)

    :param input_module: input module configuration.
    :type input_module: dict
    :param linear_kwargs: kwargs for Linear.
    :type linear_kwargs: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param teacher_model: teacher model.
    :type teacher_model: nn.Module or None
    :param student_model: student model.
    :type student_model: nn.Module or None
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, input_module, linear_kwargs, device, device_ids, distributed, teacher_model=None, student_model=None, find_unused_parameters=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        self.linear = wrap_if_distributed(nn.Linear(**linear_kwargs), device, device_ids, distributed, find_unused_parameters)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def secondary_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.linear(flat_outputs)


class Normalizer4CRD(nn.Module):
    """
    An auxiliary module for contrastive representation distillation (CRD).

    Yonglong Tian, Dilip Krishnan, Phillip Isola: `"Contrastive Representation Distillation" <https://openreview.net/forum?id=SkgpBJrtvS>`_ @ ICLR 2020 (2020)

    :param linear: linear module.
    :type linear: nn.Module
    :param power: the exponents.
    :type power: int
    """

    def __init__(self, linear, power=2):
        super().__init__()
        self.linear = linear
        self.power = power

    def forward(self, x):
        z = self.linear(x)
        norm = z.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = z.div(norm)
        return out


class Linear4CRD(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for contrastive representation distillation (CRD), including translator :class:`Normalizer4CRD`.
    Refactored https://github.com/HobbitLong/RepDistiller/blob/master/crd/memory.py

    Yonglong Tian, Dilip Krishnan, Phillip Isola: `"Contrastive Representation Distillation" <https://openreview.net/forum?id=SkgpBJrtvS>`_ @ ICLR 2020 (2020)

    :param input_module_path: path of module whose output will be flattened and then used as input to normalizer.
    :type input_module_path: str
    :param linear_kwargs: kwargs for Linear.
    :type linear_kwargs: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param power: ``power`` for :class:`Normalizer4CRD`.
    :type power: int
    :param teacher_model: teacher model.
    :type teacher_model: nn.Module or None
    :param student_model: student model.
    :type student_model: nn.Module or None
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, input_module_path, linear_kwargs, device, device_ids, distributed, power=2, teacher_model=None, student_model=None, find_unused_parameters=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.empty = nn.Sequential()
        self.input_module_path = input_module_path
        linear = nn.Linear(**linear_kwargs)
        self.normalizer = wrap_if_distributed(Normalizer4CRD(linear, power=power), device, device_ids, distributed, find_unused_parameters)

    def forward(self, x, supp_dict):
        self.empty(supp_dict)
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def secondary_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path]['output'], 1)
        self.normalizer(flat_outputs)


class HeadRCNN(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for head network distillation (HND) and generalized head network distillation (GHND).

    * Yoshitomo Matsubara, Sabur Baidya, Davide Callegaro, Marco Levorato, Sameer Singh: `"Distilled Split Deep Neural Networks for Edge-Assisted Real-Time Systems" <https://dl.acm.org/doi/10.1145/3349614.3356022>`_ @ MobiCom 2019 Workshop on Hot Topics in Video Analytics and Intelligent Edges (2019)
    * Yoshitomo Matsubara, Marco Levorato: `"Neural Compression and Filtering for Edge-assisted Real-time Object Detection in Challenged Networks"  <https://arxiv.org/abs/2007.15818>`_ @ ICPR 2020 (2021)

    :param head_rcnn: head R-CNN configuration as ``model_config`` in :meth:`torchdistill.models.util.redesign_model`.
    :type head_rcnn: dict
    :param kwargs: ``teacher_model`` or ``student_model`` keys must be included. If both ``teacher_model`` and ``student_model`` are provided, ``student_model`` will be prioritized.
    :type kwargs: dict
    """

    def __init__(self, head_rcnn, **kwargs):
        super().__init__()
        tmp_ref_model = kwargs.get('teacher_model', None)
        ref_model = kwargs.get('student_model', tmp_ref_model)
        if ref_model is None:
            raise ValueError('Either student_model or teacher_model has to be given.')
        self.transform = ref_model.transform
        self.seq = redesign_model(ref_model, head_rcnn, 'R-CNN', 'HeadRCNN')

    def forward(self, images, targets=None):
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)
        return self.seq(images.tensors)


class SSWrapper4SSKD(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for self-supervision knowledge distillation (SSKD).
    If both ``teacher_model`` and ``student_model`` are provided, ``student_model`` will be prioritized

    Guodong Xu, Ziwei Liu, Xiaoxiao Li, Chen Change Loy: `"Knowledge Distillation Meets Self-Supervision" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/898_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param input_module: input module configuration.
    :type input_module: dict
    :param feat_dim: number of input/output features for self-supervision module.
    :type feat_dim: int
    :param ss_module_ckpt: self-supervision module checkpoint file path.
    :type ss_module_ckpt: str
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param freezes_ss_module: if True, freezes self-supervision module.
    :type freezes_ss_module: bool
    :param teacher_model: teacher model.
    :type teacher_model: nn.Module or None
    :param student_model: student model.
    :type student_model: nn.Module or None
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, input_module, feat_dim, ss_module_ckpt, device, device_ids, distributed, freezes_ss_module=False, teacher_model=None, student_model=None, find_unused_parameters=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        self.model = teacher_model if is_teacher else student_model
        self.is_teacher = is_teacher
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        ss_module = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, feat_dim))
        self.ckpt_file_path = ss_module_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(ss_module, map_location, self.ckpt_file_path)
        self.ss_module = ss_module if is_teacher and freezes_ss_module else wrap_if_distributed(ss_module, device, device_ids, distributed, find_unused_parameters)

    def forward(self, x):
        if self.is_teacher:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)

    def secondary_forward(self, io_dict):
        flat_outputs = torch.flatten(io_dict[self.input_module_path][self.input_module_io], 1)
        self.ss_module(flat_outputs)

    def post_epoch_process(self, *args, **kwargs):
        save_module_ckpt(self.ss_module, self.ckpt_file_path)


class VarianceBranch4PAD(AuxiliaryModelWrapper):
    """
    An auxiliary teacher/student model wrapper for prime-aware adaptive distillation (PAD).

    Youcai Zhang, Zhonghao Lan, Yuchen Dai, Fangao Zeng, Yan Bai, Jie Chang, Yichen Wei: `"Prime-Aware Adaptive Distillation" <https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3317_ECCV_2020_paper.php>`_ @ ECCV 2020 (2020)

    :param student_model: student model.
    :type student_model: nn.Module
    :param input_module: input module configuration.
    :type input_module: dict
    :param feat_dim: number of input/output features for self-supervision module.
    :type feat_dim: int
    :param var_estimator_ckpt: variance estimator module checkpoint file path.
    :type var_estimator_ckpt: str
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, student_model, input_module, feat_dim, var_estimator_ckpt, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        var_estimator = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim))
        self.ckpt_file_path = var_estimator_ckpt
        if os.path.isfile(self.ckpt_file_path):
            map_location = {'cuda:0': 'cuda:{}'.format(device_ids[0])} if distributed else device
            load_module_ckpt(var_estimator, map_location, self.ckpt_file_path)
        self.var_estimator = wrap_if_distributed(var_estimator, device, device_ids, distributed, find_unused_parameters)

    def forward(self, x):
        return self.student_model(x)

    def secondary_forward(self, io_dict):
        embed_outputs = io_dict[self.input_module_path][self.input_module_io].flatten(1)
        self.var_estimator(embed_outputs)

    def post_epoch_process(self, *args, **kwargs):
        save_module_ckpt(self.var_estimator, self.ckpt_file_path)


class AttentionBasedFusion(nn.Module):
    """
    An auxiliary module for knowledge review (KR). Refactored https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia: `"Distilling Knowledge via Knowledge Review" <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html>`_ @ CVPR 2021 (2021)

    :param in_channels: number of input channels for the first convolution layer.
    :type in_channels: int
    :param mid_channels: number of output/input channels for the first/second convolution layer.
    :type mid_channels: int
    :param out_channels: number of output channels for the third convolution layer.
    :type out_channels: int
    """

    def __init__(self, in_channels, mid_channels, out_channels, uses_attention):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), nn.BatchNorm2d(mid_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels))
        self.attention_conv = None if not uses_attention else nn.Sequential(nn.Conv2d(mid_channels * 2, 2, kernel_size=1), nn.Sigmoid())
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)

    def forward(self, x, y=None, size=None):
        x = self.conv1(x)
        if self.attention_conv is not None:
            n, _, h, w = x.shape
            y = functional.interpolate(y, (size, size), mode='nearest')
            z = torch.cat([x, y], dim=1)
            z = self.attention_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        y = self.conv2(x)
        return y, x


class Student4KnowledgeReview(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for knowledge review (KR). Refactored https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Pengguang Chen, Shu Liu, Hengshuang Zhao, Jiaya Jia: `"Distilling Knowledge via Knowledge Review" <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html>`_ @ CVPR 2021 (2021)

    :param student_model: student model.
    :type student_model: nn.Module
    :param abfs: attention based fusion configurations.
    :type abfs: list[dict]
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, student_model, abfs, device, device_ids, distributed, sizes=None, find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        if sizes is None:
            sizes = [1, 7, 14, 28, 56]
        self.sizes = sizes
        abf_list = nn.ModuleList()
        num_abfs = len(abfs)
        io_path_pairs = list()
        for idx, abf_config in enumerate(abfs):
            abf = wrap_if_distributed(AttentionBasedFusion(uses_attention=idx < num_abfs - 1, **abf_config['kwargs']), device, device_ids, distributed, find_unused_parameters)
            abf_list.append(abf)
            io_path_pairs.append((abf_config['io'], abf_config['path']))
        self.abf_modules = abf_list[::-1]
        self.io_path_pairs = io_path_pairs[::-1]

    def forward(self, *args):
        return self.student_model(*args)

    def secondary_forward(self, io_dict):
        feature_maps = [io_dict[module_path][io_type] for io_type, module_path in self.io_path_pairs]
        out_features, res_features = self.abf_modules[0](feature_maps[0])
        if len(self.sizes) > 1:
            for features, abf, size in zip(feature_maps[1:], self.abf_modules[1:], self.sizes[1:]):
                out_features, res_features = abf(features, res_features, size)


class Student4KTAAD(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for knowledge translation and adaptation + affinity distillation (KTAAD).
    Refactored https://github.com/dvlab-research/ReviewKD/blob/master/ImageNet/models/reviewkd.py

    Tong He, Chunhua Shen, Zhi Tian, Dong Gong, Changming Sun, Youliang Yan.: `"Knowledge Adaptation for Efficient Semantic Segmentation" <https://openaccess.thecvf.com/content_CVPR_2019/html/He_Knowledge_Adaptation_for_Efficient_Semantic_Segmentation_CVPR_2019_paper.html>`_ @ CVPR 2019 (2019)

    :param student_model: student model.
    :type student_model: nn.Module
    :param input_module_path: path of module whose output is used as input to feature adapter and affinity adapter.
    :type input_module_path: str
    :param feature_adapter_config: feature adapter configuration.
    :type feature_adapter_config: dict
    :param affinity_adapter_config: affinity adapter configuration.
    :type affinity_adapter_config: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, student_model, input_module_path, feature_adapter_config, affinity_adapter_config, device, device_ids, distributed, find_unused_parameters=None, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        self.input_module_path = input_module_path
        feature_adapter = nn.Sequential(nn.Conv2d(**feature_adapter_config['conv_kwargs']), nn.BatchNorm2d(**feature_adapter_config['bn_kwargs']), nn.ReLU(**feature_adapter_config['relu_kwargs']))
        affinity_adapter = nn.Sequential(nn.Conv2d(**affinity_adapter_config['conv_kwargs']))
        self.feature_adapter = wrap_if_distributed(feature_adapter, device, device_ids, distributed, find_unused_parameters)
        self.affinity_adapter = wrap_if_distributed(affinity_adapter, device, device_ids, distributed, find_unused_parameters)

    def forward(self, *args):
        return self.student_model(*args)

    def secondary_forward(self, io_dict):
        feature_maps = io_dict[self.input_module_path]['output']
        self.feature_adapter(feature_maps)
        self.affinity_adapter(feature_maps)


class ChannelSimilarityEmbed(nn.Module):
    """
    An auxiliary module for Inter-Channel Correlation for Knowledge Distillation (ICKD). Refactored https://github.com/ADLab-AutoDrive/ICKD/blob/main/ImageNet/torchdistill/models/special.py

    Li Liu, Qingle Huang, Sihao Lin, Hongwei Xie, Bing Wang, Xiaojun Chang, Xiaodan Liang: `"Inter-Channel Correlation for Knowledge Distillation" <https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html>`_ @ ICCV 2021 (2021)

    :param in_channels: number of input channels for the convolution layer.
    :type in_channels: int
    :param out_channels: number of output channels for the convolution layer.
    :type out_channels: int
    """

    def __init__(self, in_channels=512, out_channels=128, **kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.l2norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.l2norm(x)
        return x


class Student4ICKD(AuxiliaryModelWrapper):
    """
    An auxiliary student model wrapper for Inter-Channel Correlation for Knowledge Distillation (ICKD).
    Referred to https://github.com/ADLab-AutoDrive/ICKD/blob/main/ImageNet/torchdistill/models/special.py

    Li Liu, Qingle Huang, Sihao Lin, Hongwei Xie, Bing Wang, Xiaojun Chang, Xiaodan Liang: `"Inter-Channel Correlation for Knowledge Distillation" <https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Exploring_Inter-Channel_Correlation_for_Diversity-Preserved_Knowledge_Distillation_ICCV_2021_paper.html>`_ @ ICCV 2021 (2021)

    :param student_model: student model.
    :type student_model: nn.Module
    :param embeddings: embeddings keys and configuration.
    :type embeddings: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    """

    def __init__(self, student_model, embeddings, device, device_ids, distributed, **kwargs):
        super().__init__()
        self.student_model = wrap_if_distributed(student_model, device, device_ids, distributed)
        io_path_pairs = list()
        self.embed_dict = nn.ModuleDict()
        for embed_key, embed_params in embeddings.items():
            embed = ChannelSimilarityEmbed(**embed_params)
            self.embed_dict[embed_key] = wrap_if_distributed(embed, device, device_ids, distributed)
            io_path_pairs.append((embed_key, embed_params['io'], embed_params['path']))
        self.io_path_pairs = io_path_pairs

    def forward(self, x):
        return self.student_model(x)

    def secondary_forward(self, io_dict):
        for embed_key, io_type, module_path in self.io_path_pairs:
            self.embed_dict[embed_key](io_dict[module_path][io_type])


class SRDModelWrapper(AuxiliaryModelWrapper):
    """
    An auxiliary model wrapper for Understanding the Role of the Projector in Knowledge Distillation.
    Referred to https://github.com/roymiles/Simple-Recipe-Distillation/blob/main/imagenet/torchdistill/losses/single.py

    Roy Miles, Krystian Mikolajczyk: `"Understanding the Role of the Projector in Knowledge Distillation" <https://arxiv.org/abs/2303.11098>`_ @ AAAI 2024 (2024)

    :param model: model.
    :type model: nn.Module
    :param input_module: input module configuration.
    :type input_module: dict
    :param linear_kwargs: nn.Linear keyword arguments.
    :type linear_kwargs: dict or None
    :param norm_kwargs: nn.BatchNorm1d keyword arguments.
    :type norm_kwargs: dict
    :param device: target device.
    :type device: torch.device
    :param device_ids: target device IDs.
    :type device_ids: list[int]
    :param distributed: whether to be in distributed training mode.
    :type distributed: bool
    :param teacher_model: teacher model.
    :type teacher_model: nn.Module or None
    :param student_model: student model.
    :type student_model: nn.Module or None
    :param find_unused_parameters: ``find_unused_parameters`` for DistributedDataParallel.
    :type find_unused_parameters: bool or None
    """

    def __init__(self, input_module, norm_kwargs, device, device_ids, distributed, linear_kwargs=None, teacher_model=None, student_model=None, find_unused_parameters=None, **kwargs):
        super().__init__()
        is_teacher = teacher_model is not None
        if not is_teacher:
            student_model = wrap_if_distributed(student_model, device, device_ids, distributed, find_unused_parameters)
        self.model = teacher_model if is_teacher else student_model
        self.input_module_path = input_module['path']
        self.input_module_io = input_module['io']
        self.linear = wrap_if_distributed(nn.Linear(**linear_kwargs), device, device_ids, distributed) if isinstance(linear_kwargs, dict) else None
        self.norm_layer = wrap_if_distributed(nn.BatchNorm1d(**norm_kwargs), device, device_ids, distributed)

    def forward(self, x):
        return self.model(x)

    def secondary_forward(self, io_dict):
        z = io_dict[self.input_module_path][self.input_module_io]
        z = z.mean(-1).mean(-1)
        if self.linear is not None:
            z = self.linear(z)
        self.norm_layer(z)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Bottleneck4DenseNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Bottleneck4Inception3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Bottleneck4LargeResNet,
     lambda: ([], {'bottleneck_channel': 4}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (Bottleneck4ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Bottleneck4SmallResNet,
     lambda: ([], {'bottleneck_channel': 4}),
     lambda: ([torch.rand([4, 64, 64, 64])], {})),
    (BottleneckBase,
     lambda: ([], {'encoder': torch.nn.ReLU(), 'decoder': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ChannelSimilarityEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 64, 64])], {})),
    (ConvReg,
     lambda: ([], {'num_input_channels': 4, 'num_output_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DenseNet4Cifar,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (EmptyModule,
     lambda: ([], {}),
     lambda: ([], {})),
    (Normalizer4CRD,
     lambda: ([], {'linear': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Paraphraser4FactorTransfer,
     lambda: ([], {'k': 4, 'num_input_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Regressor4VID,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4, 'eps': 4, 'init_pred_var': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Translator4FactorTransfer,
     lambda: ([], {'num_input_channels': 4, 'num_output_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WideBasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

