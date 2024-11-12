
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


import re


import torch


import copy


import scipy.sparse as sp


import numpy as np


from abc import ABCMeta


import torch.utils.data


from torch.utils.data.dataloader import default_collate


import collections


from itertools import repeat


from typing import List


import random


import inspect


from sklearn.preprocessing import StandardScaler


from collections import defaultdict


import scipy.io as sio


from itertools import product


import scipy.io


import time


from torch import Tensor


import pandas as pd


import warnings


import torch.nn.functional as F


import itertools


from collections import namedtuple


import torch.nn as nn


import torch.multiprocessing as mp


import math


from typing import Optional


from torch.utils.checkpoint import checkpoint


from torch.utils.checkpoint import get_device_states


from torch.utils.checkpoint import set_device_states


from torch import nn


from typing import Type


from typing import Any


from sklearn import preprocessing


from torch.nn.parameter import Parameter


from sklearn.cluster import SpectralClustering


from functools import partial


from scipy.linalg import block_diag


from torch.nn.modules.module import Module


from scipy.linalg import expm


from typing import Tuple


import functools


import logging


from torch.utils import checkpoint


from torch.nn import Module


import torch.nn.init as init


from torch.nn import CrossEntropyLoss


from torch.utils.cpp_extension import load


import matplotlib.cm as cm


import matplotlib.pyplot as plt


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from typing import Dict


from typing import Union


from typing import Callable


from sklearn.metrics import f1_score


import scipy


from functools import reduce


from scipy.special import iv


from torch.utils.data import DataLoader


import scipy.sparse.linalg as slinalg


from scipy.sparse import linalg


from sklearn.model_selection import StratifiedKFold


import scipy.sparse as sparse


import sklearn.preprocessing as preprocessing


from torch.utils.data import Sampler


from torch.utils.data import BatchSampler


from torch.utils.data import TensorDataset


from abc import abstractmethod


from sklearn.cluster import KMeans


from sklearn.linear_model import LogisticRegression


from sklearn.metrics import auc


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import roc_auc_score


from torch.nn import functional as F


from sklearn.multiclass import OneVsRestClassifier


from sklearn.metrics import accuracy_score


from sklearn.utils import shuffle as skshuffle


from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import KFold


from sklearn.svm import SVC


from sklearn.metrics.cluster import normalized_mutual_info_score


from scipy.optimize import linear_sum_assignment


import types


from scipy.sparse import lil_matrix


from sklearn.metrics.pairwise import cosine_similarity


from sklearn.preprocessing import normalize


from itertools import chain


from torch import optim


from sklearn import metrics


from sklearn.model_selection import ShuffleSplit


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import OneHotEncoder


from collections import Counter


from torch import optim as optim


from typing import Counter


from sklearn.utils.extmath import randomized_svd


import torch.multiprocessing


from scipy.sparse import csr_matrix


from sklearn.decomposition import PCA


from sklearn.manifold import TSNE


class linear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None, scheme=None, rp_ratio=2):
        if rp_ratio > 1:
            D = input.shape[1]
            rmat = (torch.bernoulli(torch.ones((D, D // rp_ratio)) * 0.5) * 2.0 - 1) * math.sqrt(1.0 / (D // rp_ratio))
            input_rp = torch.mm(input, rmat)
            quantized = quantize_activation(input_rp, scheme)
        else:
            quantized = quantize_activation(input, scheme)
        empty_cache(config.empty_cache_threshold)
        ctx.scheme = scheme
        if rp_ratio > 1:
            ctx.saved = quantized, weight, bias, rmat
            ctx.other_args = input_rp.shape
        else:
            ctx.saved = quantized, weight, bias
            ctx.other_args = input.shape
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scheme:
            ctx.scheme.set_scale(grad_output)
        q_input_shape = ctx.other_args
        if len(ctx.saved) == 4:
            quantized, weight, bias, rmat = ctx.saved
            input_rp = dequantize_activation(quantized, q_input_shape)
            input = torch.mm(input_rp, rmat.t())
            del quantized, ctx.saved, input_rp
        else:
            quantized, weight, bias = ctx.saved
            input = dequantize_activation(quantized, q_input_shape)
            del quantized, ctx.saved
        empty_cache(config.empty_cache_threshold)
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]
        grad_output_flatten = grad_output.view(-1, C_out)
        input_flatten = input.view(-1, C_in)
        grad_input = grad_output_flatten.mm(weight)
        grad_weight = grad_output_flatten.t().mm(input_flatten)
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        if ctx.scheme:
            ctx.scheme.if_allocate_perlayer()
        return grad_input, grad_weight, grad_bias, None, None


class QLinear(nn.Linear):
    num_layers = 0

    def __init__(self, input_features, output_features, bias=True, group=0, rp_ratio=2):
        super(QLinear, self).__init__(input_features, output_features, bias)
        if config.adaptive_conv_scheme:
            self.scheme = QScheme(self, group=group)
        else:
            self.scheme = None
        self.rp_ratio = rp_ratio

    def forward(self, input):
        if config.training:
            return linear.apply(input, self.weight, self.bias, self.scheme, self.rp_ratio)
        else:
            return super(QLinear, self).forward(input)


CONFIGS = {'fast_spmm': None, 'csrmhspmm': None, 'csr_edge_softmax': None, 'fused_gat_func': None, 'fast_spmm_cpu': None, 'spmm_flag': False, 'mh_spmm_flag': False, 'fused_gat_flag': False, 'spmm_cpu_flag': False}


def initialize_spmm():
    if CONFIGS['spmm_flag']:
        return
    CONFIGS['spmm_flag'] = True
    if torch.cuda.is_available():
        CONFIGS['fast_spmm'] = csrspmm

