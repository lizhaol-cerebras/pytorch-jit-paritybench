
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


import re


from typing import Optional


from typing import Union


import torch


from torch import Tensor


from torch import nn


from torch.nn import Module


from torch import BoolTensor


from torch import IntTensor


import matplotlib.pyplot as plt


import numpy as np


import matplotlib.animation as animation


from matplotlib.table import Table


from collections.abc import Iterable


from collections.abc import Iterator


from functools import partial


from itertools import chain


from typing import Any


from collections.abc import Sequence


from typing import Callable


from torch import tensor


from typing import List


from typing import no_type_check


from typing import Literal


from collections import OrderedDict


from collections.abc import Hashable


from collections.abc import Mapping


from copy import deepcopy


from typing import ClassVar


from typing import Dict


from torch.nn import ModuleDict


from collections.abc import Collection


import torch.distributed as dist


from types import ModuleType


from torch import distributed as dist


from functools import lru_cache


import copy


import math


import warnings


import torch.nn as nn


from torch.nn.functional import adaptive_max_pool2d


from torch.nn.functional import relu


from torch.nn.functional import softmax


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from itertools import permutations


from torch.linalg import norm


from math import ceil


from math import pi


from torch.nn.functional import pad


from torch.nn import functional as F


from itertools import combinations


from typing import cast


import inspect


from typing import NamedTuple


from torch.nn.functional import conv2d


from typing import TYPE_CHECKING


import itertools


import functools


from torch.nn.functional import conv3d


from torch.nn.functional import unfold


from torch import linalg


from torch.utils.data import DataLoader


from collections import Counter


from collections import defaultdict


from math import inf


from torch import stack


from torch.utils.data import Dataset


from enum import unique


import string


from torch.nn.functional import adaptive_avg_pool2d


from abc import ABC


from abc import abstractmethod


from collections.abc import Generator


from time import perf_counter


from itertools import product


from math import floor


from math import sqrt


from torch.nn import ModuleList


import typing


from torch.nn import Linear


import numpy


import random


from scipy.io import wavfile


from scipy.optimize import linear_sum_assignment


from torch.nn import Parameter


from scipy.special import expit as sigmoid


from sklearn.metrics import accuracy_score as sk_accuracy


from sklearn.metrics import confusion_matrix as sk_confusion_matrix


from scipy.special import softmax


from sklearn.metrics import roc_auc_score as sk_roc_auc_score


from sklearn.metrics import average_precision_score as sk_average_precision_score


from sklearn.metrics import cohen_kappa_score as sk_cohen_kappa


from scipy.spatial.distance import dice as sc_dice


from sklearn.metrics import f1_score as sk_f1_score


from sklearn.metrics import fbeta_score as sk_fbeta_score


import pandas as pd


from sklearn.metrics import hamming_loss as sk_hamming_loss


from sklearn.metrics import hinge_loss as sk_hinge


from sklearn.preprocessing import OneHotEncoder


from sklearn.metrics import jaccard_score as sk_jaccard_index


from sklearn.metrics import matthews_corrcoef as sk_matthews_corrcoef


from sklearn.metrics import precision_recall_curve as sk_precision_recall_curve


from sklearn.metrics import precision_score as sk_precision_score


from sklearn.metrics import recall_score as sk_recall_score


from sklearn.metrics import coverage_error as sk_coverage_error


from sklearn.metrics import label_ranking_average_precision_score as sk_label_ranking


from sklearn.metrics import label_ranking_loss as sk_label_ranking_loss


from sklearn.metrics import roc_curve as sk_roc_curve


from sklearn.datasets import make_blobs


from sklearn.metrics import adjusted_mutual_info_score as sklearn_ami


from sklearn.metrics import adjusted_rand_score as sklearn_adjusted_rand_score


from sklearn.metrics import mutual_info_score as sklearn_mutual_info_score


from sklearn.metrics import normalized_mutual_info_score as sklearn_nmi


from sklearn.metrics import rand_score as sklearn_rand_score


from sklearn.metrics.cluster import contingency_matrix as sklearn_contingency_matrix


from sklearn.metrics.cluster import entropy as sklearn_entropy


from sklearn.metrics.cluster import pair_confusion_matrix as sklearn_pair_confusion_matrix


from sklearn.metrics.cluster._supervised import _generalized_average as sklearn_generalized_average


from torch.multiprocessing import Pool


from torch.multiprocessing import set_sharing_strategy


from torch.multiprocessing import set_start_method


from sklearn.metrics import jaccard_score


from scipy.ndimage import uniform_filter


from scipy.linalg import sqrtm


import matplotlib


from torchvision.transforms import PILToTensor


from sklearn.metrics.pairwise import cosine_similarity


from sklearn.metrics.pairwise import euclidean_distances


from sklearn.metrics.pairwise import linear_kernel


from sklearn.metrics.pairwise import manhattan_distances


from sklearn.metrics.pairwise import pairwise_distances


from scipy.stats import pearsonr


from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


from sklearn.metrics import explained_variance_score


from scipy.stats import kendalltau


from scipy.stats import entropy


from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error


from sklearn.metrics import mean_absolute_percentage_error as sk_mean_abs_percentage_error


from sklearn.metrics import mean_squared_error as sk_mean_squared_error


from sklearn.metrics import mean_squared_log_error as sk_mean_squared_log_error


from sklearn.metrics._regression import _check_reg_targets


from sklearn.utils import check_consistent_length


from scipy.spatial.distance import minkowski as scipy_minkowski


from sklearn.metrics import r2_score as sk_r2score


from scipy.stats import rankdata


from scipy.stats import spearmanr


from sklearn.metrics import mean_tweedie_deviance


from numpy import array


from sklearn.metrics import roc_auc_score


from sklearn.metrics import label_ranking_average_precision_score


from sklearn.metrics import ndcg_score


from sklearn.metrics import f1_score


from scipy.ndimage import binary_erosion as scibinary_erosion


from scipy.ndimage import distance_transform_cdt as scidistance_transform_cdt


from scipy.ndimage import distance_transform_edt as scidistance_transform_edt


from scipy.ndimage import generate_binary_structure as scigenerate_binary_structure


from scipy.spatial import procrustes as scipy_procrustes


import torch.multiprocessing as mp


from sklearn.metrics import auc as _sk_auc


from sklearn.metrics import mean_squared_error


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


import time


from sklearn.metrics import accuracy_score

