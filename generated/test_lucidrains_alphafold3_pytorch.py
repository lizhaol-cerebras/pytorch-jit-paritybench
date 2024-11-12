
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


import random


from math import pi


from math import sqrt


from itertools import product


from itertools import zip_longest


from functools import partial


from functools import wraps


from collections import namedtuple


import torch


from torch import nn


from torch import Tensor


from torch import tensor


from torch import is_tensor


from torch.amp import autocast


import torch.nn.functional as F


from torch.utils._pytree import tree_map


from torch.nn import Module


from torch.nn import ModuleList


from torch.nn import Linear


from torch.nn import Sequential


import copy


import numpy as np


from torch.utils.data import Sampler


from collections import defaultdict


from itertools import groupby


from collections.abc import Iterable


from torch import repeat_interleave


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import Dataset


import re


from torch.optim import Adam


from torch.optim import Optimizer


from torch.utils.data import DataLoader as OrigDataLoader


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import LRScheduler


import itertools


class TorchTyping:

    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: 'str'):
        return self.abstract_dtype[Tensor, shapes]


LinearNoBias = partial(nn.Linear, bias=False)


def exists(v):
    return v is not None


def identity(x, *args, **kwargs):
    """Return the input value."""
    return x


def max_neg_value(t: 'Tensor') ->Tensor:
    """Get the maximum negative value of Tensor based on its `dtype`.

    :param t: The Tensor.
    :return: The maximum negative value of its `dtype`.
    """
    return -torch.finfo(t.dtype).max


def softclamp(t: 'Tensor', value: 'float') ->Tensor:
    """Perform a soft clamp on a Tensor.

    :param t: The Tensor.
    :param value: The value to clamp to.
    :return: The soft clamped Tensor
    """
    return (t / value).tanh() * value


def get_gpu_type() ->str:
    """Return the type of GPU detected: NVIDIA, ROCm, or Unknown."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        if 'nvidia' in device_name:
            return 'NVIDIA GPU detected'
        elif 'amd' in device_name or 'gfx' in device_name:
            return 'ROCm GPU detected'
        else:
            return 'Unknown GPU type'
    else:
        return 'No GPU available'


MAX_CONCURRENT_TENSOR_ELEMENTS = int(2000000000.0) if 'ROCm' in get_gpu_type() else float('inf')


def not_exists(val: 'Any') ->bool:
    """Check if a value does not exist.

    :param val: The value to check.
    :return: `True` if the value does not exist, otherwise `False`.
    """
    return val is None


DNA_NUCLEOTIDES = dict(A=dict(resname='DA', smile='OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)CC1O', first_atom_idx=0, last_atom_idx=21, complement='T', distogram_atom_idx=21, token_center_atom_idx=11, three_atom_indices_for_frame=(11, 8, 6)), C=dict(resname='DC', smile='OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1O', first_atom_idx=0, last_atom_idx=19, complement='G', distogram_atom_idx=13, token_center_atom_idx=11, three_atom_indices_for_frame=(11, 8, 6)), G=dict(resname='DG', smile='OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1O', first_atom_idx=0, last_atom_idx=22, complement='C', distogram_atom_idx=22, token_center_atom_idx=11, three_atom_indices_for_frame=(11, 8, 6)), T=dict(resname='DT', smile='OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1O', first_atom_idx=0, last_atom_idx=20, complement='A', distogram_atom_idx=13, token_center_atom_idx=11, three_atom_indices_for_frame=(11, 8, 6)), X=dict(resname='DN', smile='OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)CC1O', first_atom_idx=0, last_atom_idx=21, complement='N', distogram_atom_idx=21, token_center_atom_idx=11, three_atom_indices_for_frame=None))


HUMAN_AMINO_ACIDS = dict(A=dict(resname='ALA', smile='NC(C=O)C', first_atom_idx=0, last_atom_idx=4, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), R=dict(resname='ARG', smile='NC(C=O)CCCNC(N)=N', first_atom_idx=0, last_atom_idx=10, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), N=dict(resname='ASN', smile='NC(C=O)CC(=O)N', first_atom_idx=0, last_atom_idx=7, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), D=dict(resname='ASP', smile='NC(C=O)CC(=O)O', first_atom_idx=0, last_atom_idx=7, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), C=dict(resname='CYS', smile='NC(C=O)CS', first_atom_idx=0, last_atom_idx=5, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), Q=dict(resname='GLN', smile='NC(C=O)CCC(=O)N', first_atom_idx=0, last_atom_idx=8, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), E=dict(resname='GLU', smile='NC(C=O)CCC(=O)O', first_atom_idx=0, last_atom_idx=8, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), G=dict(resname='GLY', smile='NCC=O', first_atom_idx=0, last_atom_idx=3, distogram_atom_idx=1, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), H=dict(resname='HIS', smile='NC(C=O)CC1=CNC=N1', first_atom_idx=0, last_atom_idx=9, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), I=dict(resname='ILE', smile='NC(C=O)C(CC)C', first_atom_idx=0, last_atom_idx=7, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), L=dict(resname='LEU', smile='NC(C=O)CC(C)C', first_atom_idx=0, last_atom_idx=7, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), K=dict(resname='LYS', smile='NC(C=O)CCCCN', first_atom_idx=0, last_atom_idx=8, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), M=dict(resname='MET', smile='NC(C=O)CCSC', first_atom_idx=0, last_atom_idx=7, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), F=dict(resname='PHE', smile='NC(C=O)CC1=CC=CC=C1', first_atom_idx=0, last_atom_idx=10, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), P=dict(resname='PRO', smile='N1C(C=O)CCC1', first_atom_idx=0, last_atom_idx=6, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), S=dict(resname='SER', smile='NC(C=O)CO', first_atom_idx=0, last_atom_idx=5, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), T=dict(resname='THR', smile='NC(C=O)C(O)C', first_atom_idx=0, last_atom_idx=6, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), W=dict(resname='TRP', smile='NC(C=O)CC1=CNC2=C1C=CC=C2', first_atom_idx=0, last_atom_idx=13, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), Y=dict(resname='TYR', smile='NC(C=O)CC1=CC=C(O)C=C1', first_atom_idx=0, last_atom_idx=11, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), V=dict(resname='VAL', smile='NC(C=O)C(C)C', first_atom_idx=0, last_atom_idx=6, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=(0, 1, 2)), X=dict(resname='UNK', smile='NC(C=O)C', first_atom_idx=0, last_atom_idx=4, distogram_atom_idx=4, token_center_atom_idx=1, three_atom_indices_for_frame=None))


RNA_NUCLEOTIDES = dict(A=dict(resname='A', smile='OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)C(O)C1O', first_atom_idx=0, last_atom_idx=22, complement='U', distogram_atom_idx=22, token_center_atom_idx=12, three_atom_indices_for_frame=(12, 8, 6)), C=dict(resname='C', smile='OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)C(O)C1O', first_atom_idx=0, last_atom_idx=20, complement='G', distogram_atom_idx=14, token_center_atom_idx=12, three_atom_indices_for_frame=(12, 8, 6)), G=dict(resname='G', smile='OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)C(O)C1O', first_atom_idx=0, last_atom_idx=23, complement='C', distogram_atom_idx=23, token_center_atom_idx=12, three_atom_indices_for_frame=(12, 8, 6)), U=dict(resname='U', smile='OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C=C2)C(O)C1O', first_atom_idx=0, last_atom_idx=20, complement='A', distogram_atom_idx=14, token_center_atom_idx=12, three_atom_indices_for_frame=(12, 8, 6)), X=dict(resname='N', smile='OP(=O)(O)OCC1OC(N2C=NC3=C2N=CN=C3N)C(O)C1O', first_atom_idx=0, last_atom_idx=22, complement='N', distogram_atom_idx=22, token_center_atom_idx=12, three_atom_indices_for_frame=None))


NUM_MSA_ONE_HOT = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES) + 1


ADDITIONAL_MOLECULE_FEATS = 5


def log(t: 'Tensor', eps=1e-20) ->Tensor:
    """Run a safe log function that clamps the input to be above `eps` to avoid `log(0)`.

    :param t: The input tensor.
    :param eps: The epsilon value.
    :return: Tensor in the log domain.
    """
    return torch.log(t.clamp(min=eps))


def exclusive_cumsum(t: 'Tensor', dim: 'int'=-1) ->Tensor:
    """Perform an exclusive cumulative summation on a Tensor.

    :param t: The Tensor.
    :param dim: The dimension to sum over.
    :return: The exclusive cumulative sum Tensor.
    """
    return t.cumsum(dim=dim) - t


IS_DNA_INDEX = 2


IS_LIGAND_INDEX = -2


IS_METAL_ION_INDEX = -1


IS_PROTEIN_INDEX = 0


IS_RNA_INDEX = 1


def l2norm(t: 'Tensor', eps: 'float'=1e-20, dim: 'int'=-1) ->Tensor:
    """Perform an L2 normalization on a Tensor.

    :param t: The Tensor.
    :param eps: The epsilon value.
    :param dim: The dimension to normalize over.
    :return: The L2 normalized Tensor.
    """
    return F.normalize(t, p=2, eps=eps, dim=dim)


NUM_MOLECULE_IDS = len(HUMAN_AMINO_ACIDS) + len(RNA_NUCLEOTIDES) + len(DNA_NUCLEOTIDES) + 2


IS_MOLECULE_TYPES = 5


IS_PROTEIN, IS_RNA, IS_DNA, IS_LIGAND, IS_METAL_ION = tuple(IS_MOLECULE_TYPES + i if i < 0 else i for i in [IS_PROTEIN_INDEX, IS_RNA_INDEX, IS_DNA_INDEX, IS_LIGAND_INDEX, IS_METAL_ION_INDEX])

