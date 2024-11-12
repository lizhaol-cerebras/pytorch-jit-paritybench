
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


import numbers


from collections import defaultdict


import torch


from sklearn.metrics import matthews_corrcoef


from sklearn.metrics import f1_score


import torch.multiprocessing as mp


import copy


from functools import partial


import random


from itertools import chain


from itertools import groupby


import numpy as np


from sklearn.utils.class_weight import compute_class_weight


from torch.utils.data import Dataset


from torch.utils.data import Subset


from torch.utils.data import IterableDataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import KFold


from sklearn.model_selection import ShuffleSplit


from sklearn.model_selection import StratifiedShuffleSplit


from math import ceil


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from typing import Iterable


from torch.utils.data import ConcatDataset


from torch.utils.data import TensorDataset


import abc


import inspect


from abc import ABC


from inspect import signature


from random import randint


from numpy.random import random as random_float


from sklearn.preprocessing import StandardScaler


from functools import reduce


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from sklearn.metrics import mean_squared_error


from sklearn.metrics import r2_score


from sklearn.metrics import classification_report


from functools import wraps


import warnings


from typing import Generator


from typing import List


from typing import Union


import numpy


from torch import nn


from collections import OrderedDict


from torch.nn.parallel import DistributedDataParallel


from torch.nn import DataParallel


from typing import Tuple


from torch import optim


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


from torch.nn import BCEWithLogitsLoss


from torch.nn import NLLLoss


import torch.distributed as dist


from torch import multiprocessing as mp


from copy import deepcopy


import pandas as pd


import time


from torch.utils.data import SequentialSampler


logger = logging.getLogger(__name__)


def pick_single_fn(heads, fn_name):
    """ Iterates over heads and returns a static method called fn_name
    if and only if one head has a method of that name. If no heads have such a method, None is returned.
    If more than one head has such a method, an Exception is thrown"""
    merge_fns = []
    for h in heads:
        merge_fns.append(getattr(h, fn_name, None))
    merge_fns = [x for x in merge_fns if x is not None]
    if len(merge_fns) == 0:
        return None
    elif len(merge_fns) == 1:
        return merge_fns[0]
    else:
        raise Exception(f'More than one of the prediction heads have a {fn_name}() function')


def stack(list_of_lists):
    n_lists_final = len(list_of_lists[0])
    ret = [list() for _ in range(n_lists_final)]
    for l in list_of_lists:
        for i, x in enumerate(l):
            ret[i] += x
    return ret


class BaseAdaptiveModel:
    """
    Base Class for implementing AdaptiveModel with frameworks like PyTorch and ONNX.
    """
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific AdaptiveModel implementation.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def __init__(self, prediction_heads):
        self.prediction_heads = prediction_heads

    @classmethod
    def load(cls, **kwargs):
        """
        Load corresponding AdaptiveModel Class(AdaptiveModel/ONNXAdaptiveModel) based on the
        files in the load_dir.

        :param kwargs: arguments to pass for loading the model.
        :return: instance of a model
        """
        if (Path(kwargs['load_dir']) / 'model.onnx').is_file():
            model = cls.subclasses['ONNXAdaptiveModel'].load(**kwargs)
        else:
            model = cls.subclasses['AdaptiveModel'].load(**kwargs)
        return model

    def logits_to_preds(self, logits, **kwargs):
        """
        Get predictions from all prediction heads.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param label_maps: Maps from label encoding to label string
        :param label_maps: dict
        :return: A list of all predictions from all prediction heads
        """
        all_preds = []
        for head, logits_for_head in zip(self.prediction_heads, logits):
            preds = head.logits_to_preds(logits=logits_for_head, **kwargs)
            all_preds.append(preds)
        return all_preds

    def formatted_preds(self, logits, **kwargs):
        """
        Format predictions for inference.

        :param logits: model logits
        :type logits: torch.tensor
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        :return: predictions in the right format
        """
        n_heads = len(self.prediction_heads)
        if n_heads == 0:
            preds_final = self.language_model.formatted_preds(logits=logits, **kwargs)
        elif n_heads == 1:
            preds_final = []
            try:
                preds = kwargs['preds']
                temp = [y[0] for y in preds]
                preds_flat = [item for sublist in temp for item in sublist]
                kwargs['preds'] = preds_flat
            except KeyError:
                kwargs['preds'] = None
            head = self.prediction_heads[0]
            logits_for_head = logits[0]
            preds = head.formatted_preds(logits=logits_for_head, **kwargs)
            if type(preds) == list:
                preds_final += preds
            elif type(preds) == dict and 'predictions' in preds:
                preds_final.append(preds)
        else:
            preds_final = [list() for _ in range(n_heads)]
            preds = kwargs.get('preds')
            if preds is not None:
                preds_for_heads = stack(preds)
                logits_for_heads = [None] * n_heads
                del kwargs['preds']
            else:
                preds_for_heads = [None] * n_heads
                logits_for_heads = logits
            preds_final = [list() for _ in range(n_heads)]
            if not 'samples' in kwargs:
                samples = [s for b in kwargs['baskets'] for s in b.samples]
                kwargs['samples'] = samples
            for i, (head, preds_for_head, logits_for_head) in enumerate(zip(self.prediction_heads, preds_for_heads, logits_for_heads)):
                preds = head.formatted_preds(logits=logits_for_head, preds=preds_for_head, **kwargs)
                preds_final[i].append(preds)
            merge_fn = pick_single_fn(self.prediction_heads, 'merge_formatted_preds')
            if merge_fn:
                preds_final = merge_fn(preds_final)
        return preds_final

    def connect_heads_with_processor(self, tasks, require_labels=True):
        """
        Populates prediction head with information coming from tasks.

        :param tasks: A dictionary where the keys are the names of the tasks and the values are the details of the task (e.g. label_list, metric, tensor name)
        :param require_labels: If True, an error will be thrown when a task is not supplied with labels)
        :return:
        """
        if 'nextsentence' not in tasks:
            idx = None
            for i, ph in enumerate(self.prediction_heads):
                if ph.task_name == 'nextsentence':
                    idx = i
            if idx is not None:
                logger.info('Removing the NextSentenceHead since next_sent_pred is set to False in the BertStyleLMProcessor')
                del self.prediction_heads[i]
        for head in self.prediction_heads:
            head.label_tensor_name = tasks[head.task_name]['label_tensor_name']
            label_list = tasks[head.task_name]['label_list']
            if not label_list and require_labels:
                raise Exception(f"The task '{head.task_name}' is missing a valid set of labels")
            label_list = tasks[head.task_name]['label_list']
            head.label_list = label_list
            if 'RegressionHead' in str(type(head)):
                num_labels = 1
            else:
                num_labels = len(label_list)
            head.metric = tasks[head.task_name]['metric']

    @classmethod
    def _get_prediction_head_files(cls, load_dir, strict=True):
        load_dir = Path(load_dir)
        files = os.listdir(load_dir)
        model_files = [(load_dir / f) for f in files if '.bin' in f and 'prediction_head' in f]
        config_files = [(load_dir / f) for f in files if 'config.json' in f and 'prediction_head' in f]
        model_files.sort()
        config_files.sort()
        if strict:
            error_str = f'There is a mismatch in number of model files ({len(model_files)}) and config files ({len(config_files)}).This might be because the Language Model Prediction Head does not currently support saving and loading'
            assert len(model_files) == len(config_files), error_str
        logger.info(f'Found files for loading {len(model_files)} prediction heads')
        return model_files, config_files

