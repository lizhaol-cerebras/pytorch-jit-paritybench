
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


import torch


import time


from collections import defaultdict


import logging


import random


from typing import Dict


from typing import List


from typing import Type


import warnings


from typing import Any


from typing import Optional


from typing import Tuple


from typing import cast


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import matplotlib.pyplot as plt


import torch.nn.functional as F


import torch.utils.data


from scipy.io import savemat


from typing import Union


import matplotlib as mpl


import copy


from enum import Enum


from scipy.sparse import csr_matrix


from scipy.sparse import tril


from torch.autograd import grad


from torch.autograd import gradcheck


import itertools


import math


import scipy.sparse


from functools import reduce


from typing import Sequence


import re


import abc


from itertools import chain


from typing import Callable


import torch.autograd.functional as autogradF


from collections import OrderedDict


from typing import Iterable


from typing import Protocol


from itertools import count


from typing import Set


from scipy import ndimage


import torch.linalg


from scipy.sparse import csc_matrix


from torch.distributions import Normal


from typing import NoReturn


from torch.autograd.function import once_differentiable


from typing import NamedTuple


import numpy.random as npr


from torch.autograd import Function


from torch.autograd import Variable


from torch.nn import Module


import matplotlib.patches as mpatches


import collections


import torch.optim as optim


from scipy.sparse import lil_matrix


from typing import cast as type_cast


from torch.types import Number


from torch.utils._pytree import tree_flatten


from torch.utils._pytree import tree_map_only


from typing import TYPE_CHECKING


class SimpleCNN(nn.Module):

    def __init__(self, D=32):
        super(SimpleCNN, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, D, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(D, 3, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(D)

    def forward(self, img):
        x = self.relu(self.bn1(self.conv1(img)))
        return self.conv2(x)


class SimpleNN(nn.Module):

    def __init__(self, in_size, out_size, hid_size=30, use_offset=False):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, hid_size), nn.ReLU(), nn.Linear(hid_size, hid_size), nn.ReLU(), nn.Linear(hid_size, out_size))

    def forward(self, state_):
        return self.fc(state_)


class BackwardMode(Enum):
    UNROLL = 0
    IMPLICIT = 1
    TRUNCATED = 2
    DLM = 3

    @staticmethod
    def resolve(key: "Union[str, 'BackwardMode']") ->'BackwardMode':
        if isinstance(key, BackwardMode):
            return key
        if not isinstance(key, str):
            raise ValueError('Backward mode must be th.BackwardMode or string.')
        try:
            backward_mode = BackwardMode[key.upper()]
        except KeyError:
            raise ValueError(f'Unrecognized backward mode f{key}.Valid choices are unroll, implicit, truncated, dlm.')
        return backward_mode


DeviceType = Optional[Union[str, torch.device]]


__FROM_THESEUS_LAYER_TOKEN__ = '__FROM_THESEUS_LAYER_TOKEN__'


def _forward(objective: 'Objective', optimizer: 'Optimizer', optimizer_kwargs: 'Dict[str, Any]', input_tensors: 'Dict[str, torch.Tensor]'):
    objective.update(input_tensors)
    optimizer_kwargs[__FROM_THESEUS_LAYER_TOKEN__] = True
    info = optimizer.optimize(**optimizer_kwargs)
    vars = [var.tensor for var in objective.optim_vars.values()]
    return vars, info


class TheseusLayerDLMForward(torch.autograd.Function):
    """
    Functionally the same as the forward method in a TheseusLayer
    but computes the direct loss minimization in the backward pass.
    """
    _DLM_EPSILON_STR = 'dlm_epsilon'
    _GRAD_SUFFIX = '_grad'

    @staticmethod
    def forward(ctx, objective, optimizer, optimizer_kwargs, bwd_objective, bwd_optimizer, epsilon, n, *inputs):
        input_keys = inputs[:n]
        input_vals = inputs[n:2 * n]
        differentiable_tensors = inputs[2 * n:]
        ctx.n = n
        ctx.k = len(differentiable_tensors)
        inputs = dict(zip(input_keys, input_vals))
        ctx.input_keys = input_keys
        optim_tensors, info = _forward(objective, optimizer, optimizer_kwargs, inputs)
        if ctx.k > 0:
            ctx.bwd_objective = bwd_objective
            ctx.bwd_optimizer = bwd_optimizer
            ctx.epsilon = epsilon
            with torch.enable_grad():
                grad_sol = torch.autograd.grad(objective.error_metric().sum(), differentiable_tensors, allow_unused=True)
            ctx.save_for_backward(*input_vals, *grad_sol, *differentiable_tensors, *optim_tensors)
        return *optim_tensors, info

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        n, k = ctx.n, ctx.k
        saved_tensors = ctx.saved_tensors
        input_vals = saved_tensors[:n]
        grad_sol = saved_tensors[n:n + k]
        differentiable_tensors = saved_tensors[n + k:n + k + k]
        optim_tensors = saved_tensors[n + k + k:]
        grad_outputs = grad_outputs[:-1]
        bwd_objective: 'Objective' = ctx.bwd_objective
        bwd_optimizer: 'Optimizer' = ctx.bwd_optimizer
        epsilon = ctx.epsilon
        input_keys = ctx.input_keys
        bwd_data = dict(zip(input_keys, input_vals))
        for k, v in zip(bwd_objective.optim_vars.keys(), optim_tensors):
            bwd_data[k] = v.detach()
        grad_data = {TheseusLayerDLMForward._DLM_EPSILON_STR: torch.tensor(epsilon).reshape(1, 1)}
        for i, name in enumerate(bwd_objective.optim_vars.keys()):
            grad_data[name + TheseusLayerDLMForward._GRAD_SUFFIX] = grad_outputs[i]
        bwd_data.update(grad_data)
        bwd_objective.update(bwd_data)
        with torch.no_grad():
            bwd_optimizer.linear_solver.linearization.linearize()
            delta = bwd_optimizer.linear_solver.solve()
            bwd_optimizer.objective.retract_vars_sequence(delta, bwd_optimizer.linear_solver.linearization.ordering)
        with torch.enable_grad():
            grad_perturbed = torch.autograd.grad(bwd_objective.error_metric().sum(), differentiable_tensors, allow_unused=True)
        nones = [None] * (ctx.n * 2)
        grads = [((gs - gp) / epsilon if gs is not None else None) for gs, gp in zip(grad_sol, grad_perturbed)]
        return None, None, None, None, None, None, None, *nones, *grads


_CostFunctionSchema = Tuple[str, ...]


def _CHECK_DTYPE_SUPPORTED(dtype):
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(f'Unsupported data type {dtype}. Theseus only supports 32- and 64-bit tensors.')


class masked_variables:

    def __init__(self, vars: 'List[Variable]', mask: 'torch.Tensor') ->None:
        assert mask.dtype == torch.bool and mask.ndim == 1
        self._vars = vars
        self._original_tensors = [v.tensor for v in vars]
        self._mask = mask

    def __enter__(self) ->None:
        for v in self._vars:
            assert v.shape[0] == self._mask.shape[0]
            v.update(v.tensor[self._mask])

    def __exit__(self, exc_type: 'Any', exc_value: 'Any', traceback: 'Any') ->None:
        for i, v in enumerate(self._vars):
            v.update(self._original_tensors[i])


def masked_jacobians(cost_fn: "'CostFunction'", mask: 'torch.Tensor') ->Tuple[List[torch.Tensor], torch.Tensor]:
    cost_fn_vars: 'List[Variable]' = cast(List[Variable], list(cost_fn.optim_vars)) + list(cost_fn.aux_vars)
    batch_size = max(v.shape[0] for v in cost_fn_vars)
    aux_tensor = cost_fn_vars[0].tensor
    jacobians = [aux_tensor.new_zeros(batch_size, cost_fn.dim(), v.dof()) for v in cost_fn.optim_vars]
    err = aux_tensor.new_zeros(batch_size, cost_fn.dim())
    with masked_variables(cost_fn_vars, mask):
        masked_jacobians_, err[mask] = cost_fn.jacobians()
        for masked_jac, jac in zip(masked_jacobians_, jacobians):
            jac[mask] = masked_jac
    return jacobians, err


class AutogradMode(Enum):
    DENSE = 0
    LOOP_BATCH = 1
    VMAP = 2

    @staticmethod
    def resolve(key: "Union[str, 'AutogradMode']") ->'AutogradMode':
        if isinstance(key, AutogradMode):
            return key
        if not isinstance(key, str):
            raise ValueError('Autograd mode must be of type th.AutogradMode or string.')
        try:
            mode = AutogradMode[key.upper()]
        except KeyError:
            raise ValueError(f'Invalid autograd mode {key}. Valid options are dense, loop_batch, and vmap.')
        return mode


def as_variable(value: 'Union[float, Sequence[float], torch.Tensor, Variable]', device: 'DeviceType'=None, dtype: 'Optional[torch.dtype]'=None, name: 'Optional[str]'=None) ->Variable:
    if isinstance(value, Variable):
        return value
    if isinstance(device, str):
        device = torch.device(device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if isinstance(value, float):
        tensor = tensor.view(1, 1)
    return Variable(tensor, name=name)


def _get_cost_function_schema(cost_function: 'CostFunction') ->_CostFunctionSchema:

    def _fullname(obj) ->str:
        _name = f'{obj.__module__}.{obj.__class__.__name__}'
        if isinstance(obj, AutoDiffCostFunction):
            _name += f'__{id(obj._err_fn)}'
        if isinstance(obj, RobustCostFunction):
            _name += f'__{_fullname(obj.cost_function)}__{_fullname(obj.loss)}__{obj.flatten_dims}'
        return _name

    def _varinfo(var) ->str:
        return f'{_fullname(var)}{tuple(var.shape[1:])}'
    return (_fullname(cost_function),) + tuple(_varinfo(v) for v in cost_function.optim_vars) + tuple(_varinfo(v) for v in cost_function.aux_vars) + (_fullname(cost_function.weight),) + tuple(_varinfo(v) for v in cost_function.weight.aux_vars)


class _VectorizationMode(Enum):
    ERROR = 0
    WEIGHTED_ERROR = 1
    FULL = 2


class VariableOrdering:

    def __init__(self, objective: 'Objective', default_order: 'bool'=True):
        self.objective = objective
        self._var_order: 'List[Variable]' = []
        self._var_name_to_index: 'Dict[str, int]' = {}
        if default_order:
            self._compute_default_order(objective)

    def _compute_default_order(self, objective: 'Objective'):
        assert not self._var_order and not self._var_name_to_index
        cur_idx = 0
        for variable_name, variable in objective.optim_vars.items():
            if variable_name in self._var_name_to_index:
                continue
            self._var_order.append(variable)
            self._var_name_to_index[variable_name] = cur_idx
            cur_idx += 1

    def index_of(self, key: 'str') ->int:
        return self._var_name_to_index[key]

    def __getitem__(self, index) ->Variable:
        return self._var_order[index]

    def __iter__(self):
        return iter(self._var_order)

    def append(self, var: 'Variable'):
        if var.name in self._var_name_to_index:
            raise ValueError(f'Variable {var.name} has already been added to the order.')
        if var.name not in self.objective.optim_vars:
            raise ValueError(f'Variable {var.name} is not an optimization variable for the objective.')
        self._var_order.append(var)
        self._var_name_to_index[var.name] = len(self._var_order) - 1

    def remove(self, var: 'Variable'):
        self._var_order.remove(var)
        del self._var_name_to_index[var.name]

    def extend(self, variables: 'Sequence[Variable]'):
        for var in variables:
            self.append(var)

    @property
    def complete(self):
        return len(self._var_order) == self.objective.size_variables()


class Linearization(abc.ABC):

    def __init__(self, objective: 'Objective', ordering: 'Optional[VariableOrdering]'=None, **kwargs):
        self.objective = objective
        if ordering is None:
            ordering = VariableOrdering(objective, default_order=True)
        self.ordering = ordering
        if not self.ordering.complete:
            raise ValueError('Given variable ordering is not complete.')
        self.var_dims: 'List[int]' = []
        self.var_start_cols: 'List[int]' = []
        col_counter = 0
        for var in ordering:
            v_dim = var.dof()
            self.var_start_cols.append(col_counter)
            self.var_dims.append(v_dim)
            col_counter += v_dim
        self.num_cols = col_counter
        self.num_rows = self.objective.dim()

    @abc.abstractmethod
    def _linearize_jacobian_impl(self):
        pass

    @abc.abstractmethod
    def _linearize_hessian_impl(self, _detach_hessian: 'bool'=False):
        pass

    def linearize(self, _detach_hessian: 'bool'=False):
        if not self.ordering.complete:
            raise RuntimeError('Attempted to linearize an objective with an incomplete variable order.')
        self._linearize_hessian_impl(_detach_hessian=_detach_hessian)

    def hessian_approx(self):
        raise NotImplementedError(f'hessian_approx is not implemented for {self.__class__.__name__}')

    @abc.abstractmethod
    def _ata_impl(self) ->torch.Tensor:
        pass

    @abc.abstractmethod
    def _atb_impl(self) ->torch.Tensor:
        pass

    @property
    def AtA(self) ->torch.Tensor:
        return self._ata_impl()

    @property
    def Atb(self) ->torch.Tensor:
        return self._atb_impl()

    @abc.abstractmethod
    def Av(self, v: 'torch.Tensor') ->torch.Tensor:
        pass

    @abc.abstractmethod
    def diagonal_scaling(self, v: 'torch.Tensor') ->torch.Tensor:
        pass


class DenseLinearization(Linearization):

    def __init__(self, objective: 'Objective', ordering: 'Optional[VariableOrdering]'=None, **kwargs):
        super().__init__(objective, ordering)
        self.A: 'torch.Tensor' = None
        self.b: 'torch.Tensor' = None
        self._AtA: 'torch.Tensor' = None
        self._Atb: 'torch.Tensor' = None

    def _linearize_jacobian_impl(self):
        err_row_idx = 0
        self.A = torch.zeros((self.objective.batch_size, self.num_rows, self.num_cols), device=self.objective.device, dtype=self.objective.dtype)
        self.b = torch.zeros((self.objective.batch_size, self.num_rows), device=self.objective.device, dtype=self.objective.dtype)
        for cost_function in self.objective._get_jacobians_iter():
            jacobians, error = cost_function.weighted_jacobians_error()
            num_rows = cost_function.dim()
            for var_idx_in_cost_function, var_jacobian in enumerate(jacobians):
                var_idx_in_order = self.ordering.index_of(cost_function.optim_var_at(var_idx_in_cost_function).name)
                var_start_col = self.var_start_cols[var_idx_in_order]
                num_cols = var_jacobian.shape[2]
                row_slice = slice(err_row_idx, err_row_idx + num_rows)
                col_slice = slice(var_start_col, var_start_col + num_cols)
                self.A[:, row_slice, col_slice] = var_jacobian
            self.b[:, row_slice] = -error
            err_row_idx += cost_function.dim()

    def _linearize_hessian_impl(self, _detach_hessian: 'bool'=False):
        self._linearize_jacobian_impl()
        At = self.A.transpose(1, 2)
        self._AtA = At.bmm(self.A).detach() if _detach_hessian else At.bmm(self.A)
        self._Atb = At.bmm(self.b.unsqueeze(2))

    def hessian_approx(self):
        return self._AtA

    def _ata_impl(self) ->torch.Tensor:
        return self._AtA

    def _atb_impl(self) ->torch.Tensor:
        return self._Atb

    def Av(self, v: 'torch.Tensor') ->torch.Tensor:
        return self.A.bmm(v.unsqueeze(2)).squeeze(2)

    def diagonal_scaling(self, v: 'torch.Tensor') ->torch.Tensor:
        return v * self._AtA.diagonal(dim1=1, dim2=2)


class LinearSolver(abc.ABC):

    def __init__(self, objective: 'Objective', linearization_cls: 'Optional[Type[Linearization]]'=None, linearization_kwargs: 'Optional[Dict[str, Any]]'=None, **kwargs):
        linearization_kwargs = linearization_kwargs or {}
        self.linearization: 'Linearization' = linearization_cls(objective, **linearization_kwargs)

    def reset(self, **kwargs):
        pass

    @abc.abstractmethod
    def solve(self, damping: 'Optional[Union[float, torch.Tensor]]'=None, **kwargs) ->torch.Tensor:
        pass


class DenseSolver(LinearSolver, abc.ABC):

    def __init__(self, objective: 'Objective', linearization_cls: 'Optional[Type[Linearization]]'=None, linearization_kwargs: 'Optional[Dict[str, Any]]'=None, check_singular: 'bool'=False):
        linearization_cls = linearization_cls or DenseLinearization
        if linearization_cls != DenseLinearization:
            raise RuntimeError(f'DenseSolver only works with theseus.nonlinear.DenseLinearization, but {linearization_cls} was provided.')
        super().__init__(objective, linearization_cls, linearization_kwargs)
        self.linearization: 'DenseLinearization' = self.linearization
        self._check_singular = check_singular

    @staticmethod
    def _apply_damping(matrix: 'torch.Tensor', damping: 'Union[float, torch.Tensor]', ellipsoidal: 'bool'=True, eps: 'float'=1e-08) ->torch.Tensor:
        if matrix.ndim != 3:
            raise ValueError('Matrix must have a 3 dimensions, the first one being a batch dimension.')
        _, n, m = matrix.shape
        if n != m:
            raise ValueError('Matrix must be square.')
        damping = torch.as_tensor(damping)
        if ellipsoidal:
            damping = damping.view(-1, 1)
            damped_D = torch.diag_embed(damping * matrix.diagonal(dim1=1, dim2=2) + eps)
        else:
            damping = damping.view(-1, 1, 1)
            damped_D = damping * torch.eye(n, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
        return matrix + damped_D

    def _apply_damping_and_solve(self, Atb: 'torch.Tensor', AtA: 'torch.Tensor', damping: 'Optional[Union[float, torch.Tensor]]'=None, ellipsoidal_damping: 'bool'=True, damping_eps: 'float'=1e-08) ->torch.Tensor:
        if damping is not None:
            AtA = DenseSolver._apply_damping(AtA, damping, ellipsoidal=ellipsoidal_damping, eps=damping_eps)
        return self._solve_sytem(Atb, AtA)

    @abc.abstractmethod
    def _solve_sytem(self, Atb: 'torch.Tensor', AtA: 'torch.Tensor') ->torch.Tensor:
        pass

    def solve(self, damping: 'Optional[Union[float, torch.Tensor]]'=None, ellipsoidal_damping: 'bool'=True, damping_eps: 'float'=1e-08, **kwargs) ->torch.Tensor:
        if self._check_singular:
            AtA = self.linearization.AtA
            Atb = self.linearization.Atb
            with torch.no_grad():
                output = torch.zeros(AtA.shape[0], AtA.shape[1])
                _, _, infos = torch.lu(AtA, get_infos=True)
                good_idx = infos.bool().logical_not()
                if not good_idx.all():
                    warnings.warn('Singular matrix found in batch, solution will be set to all 0 for all singular matrices.', RuntimeWarning)
            AtA = AtA[good_idx]
            Atb = Atb[good_idx]
            solution = self._apply_damping_and_solve(Atb, AtA, damping=damping, ellipsoidal_damping=ellipsoidal_damping, damping_eps=damping_eps)
            output[good_idx] = solution
            return output
        else:
            return self._apply_damping_and_solve(self.linearization.Atb, self.linearization.AtA, damping=damping, ellipsoidal_damping=ellipsoidal_damping, damping_eps=damping_eps)


class CholeskyDenseSolver(DenseSolver):

    def __init__(self, objective: 'Objective', linearization_cls: 'Optional[Type[Linearization]]'=DenseLinearization, linearization_kwargs: 'Optional[Dict[str, Any]]'=None, check_singular: 'bool'=False):
        super().__init__(objective, linearization_cls, linearization_kwargs, check_singular=check_singular)

    def _solve_sytem(self, Atb: 'torch.Tensor', AtA: 'torch.Tensor') ->torch.Tensor:
        lower = torch.linalg.cholesky(AtA)
        return torch.cholesky_solve(Atb, lower).squeeze(2)


_LUCudaSolveFunctionBwdReturnType = Tuple[torch.Tensor, torch.Tensor, None, None, None, None, None, None, None]


def compute_A_grad(batch_size: 'int', A_row_ptr: 'np.ndarray', A_col_ind: 'np.ndarray', b: 'torch.Tensor', x: 'torch.Tensor', b_Ax: 'torch.Tensor', H: 'torch.Tensor', AH: 'torch.Tensor', damping_alpha_beta: 'Optional[Tuple[torch.Tensor, torch.Tensor]]', A_val: 'Optional[torch.Tensor]', ctx_A_col_ind: 'Optional[torch.Tensor]', detach_hessian: 'bool'):
    A_grad = torch.empty(size=(batch_size, len(A_col_ind)), device=x.device)
    for r in range(len(A_row_ptr) - 1):
        start, end = A_row_ptr[r], A_row_ptr[r + 1]
        columns = A_col_ind[start:end]
        if detach_hessian:
            A_grad[:, start:end] = b[:, r].unsqueeze(1) * H[:, columns]
        else:
            A_grad[:, start:end] = b_Ax[:, r].unsqueeze(1) * H[:, columns] - AH[:, r].unsqueeze(1) * x[:, columns]
    if damping_alpha_beta is not None and (damping_alpha_beta[0] > 0.0).any():
        assert not detach_hessian
        alpha = damping_alpha_beta[0].view(-1, 1)
        alpha2Hx = alpha * 2.0 * H * x
        A_grad -= A_val * alpha2Hx[:, ctx_A_col_ind.type(torch.long)]
    return A_grad


def _mat_vec_cpu(batch_size: 'int', num_cols: 'int', A_row_ptr: 'torch.Tensor', A_col_ind: 'torch.Tensor', A_val: 'torch.Tensor', v: 'torch.Tensor') ->torch.Tensor:
    assert batch_size == A_val.shape[0]
    num_rows = len(A_row_ptr) - 1
    retv_data = np.array([(csr_matrix((A_val[i].numpy(), A_col_ind, A_row_ptr), (num_rows, num_cols)) * v[i]) for i in range(batch_size)], dtype=np.float64)
    return torch.tensor(retv_data, dtype=torch.float64)


def _tmat_vec_cpu(batch_size: 'int', num_cols: 'int', A_row_ptr: 'torch.Tensor', A_col_ind: 'torch.Tensor', A_val: 'torch.Tensor', v: 'torch.Tensor') ->torch.Tensor:
    assert batch_size == A_val.shape[0]
    num_rows = len(A_row_ptr) - 1
    retv_data = np.array([(csc_matrix((A_val[i].numpy(), A_col_ind, A_row_ptr), (num_cols, num_rows)) * v[i]) for i in range(batch_size)], dtype=np.float64)
    return torch.tensor(retv_data, dtype=torch.float64)


class SparseStructure(abc.ABC):

    def __init__(self, col_ind: 'np.ndarray', row_ptr: 'np.ndarray', num_rows: 'int', num_cols: 'int', dtype: 'np.dtype'=np.float64):
        self.col_ind = col_ind
        self.row_ptr = row_ptr
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.dtype = dtype

    def csr_straight(self, val: 'torch.Tensor') ->csr_matrix:
        return csr_matrix((val, self.col_ind, self.row_ptr), (self.num_rows, self.num_cols), dtype=self.dtype)

    def csc_transpose(self, val: 'torch.Tensor') ->csc_matrix:
        return csc_matrix((val, self.col_ind, self.row_ptr), (self.num_cols, self.num_rows), dtype=self.dtype)

    def mock_csc_transpose(self) ->csc_matrix:
        return csc_matrix((np.ones(len(self.col_ind), dtype=self.dtype), self.col_ind, self.row_ptr), (self.num_cols, self.num_rows), dtype=self.dtype)


def _sparse_mat_vec_bwd_backend(ctx: 'Any', grad_output: 'torch.Tensor', is_tmat: 'bool') ->Tuple[torch.Tensor, torch.Tensor]:
    A_val, A_row_ptr, A_col_ind, v = ctx.saved_tensors
    num_rows = len(A_row_ptr) - 1
    A_grad = torch.zeros_like(A_val)
    v_grad = torch.zeros_like(v)
    for row in range(num_rows):
        start = A_row_ptr[row]
        end = A_row_ptr[row + 1]
        columns = A_col_ind[start:end].long()
        if is_tmat:
            A_grad[:, start:end] = v[:, row].view(-1, 1) * grad_output[:, columns]
            v_grad[:, row] = (grad_output[:, columns] * A_val[:, start:end]).sum(dim=1)
        else:
            A_grad[:, start:end] = v[:, columns] * grad_output[:, row].view(-1, 1)
            v_grad[:, columns] += grad_output[:, row].view(-1, 1) * A_val[:, start:end]
    return A_grad, v_grad


def _sparse_mat_vec_fwd_backend(ctx: 'Any', num_cols: 'int', A_row_ptr: 'torch.Tensor', A_col_ind: 'torch.Tensor', A_val: 'torch.Tensor', v: 'torch.Tensor', op: 'Callable[[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]') ->torch.Tensor:
    assert A_row_ptr.ndim == 1
    assert A_col_ind.ndim == 1
    assert A_val.ndim == 2
    assert v.ndim == 2
    ctx.save_for_backward(A_val, A_row_ptr, A_col_ind, v)
    ctx.num_cols = num_cols
    return op(A_val.shape[0], num_cols, A_row_ptr, A_col_ind, A_val, v)


class _SparseMtvPAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Any', num_cols: 'int', A_row_ptr: 'torch.Tensor', A_col_ind: 'torch.Tensor', A_val: 'torch.Tensor', v: 'torch.Tensor') ->torch.Tensor:
        return _sparse_mat_vec_fwd_backend(ctx, num_cols, A_row_ptr, A_col_ind, A_val, v, tmat_vec)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: 'Any', grad_output: 'torch.Tensor') ->Tuple[None, None, None, torch.Tensor, torch.Tensor]:
        A_grad, v_grad = _sparse_mat_vec_bwd_backend(ctx, grad_output, True)
        return None, None, None, A_grad, v_grad


sparse_mtv = _SparseMtvPAutograd.apply


class _SparseMvPAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Any', num_cols: 'int', A_row_ptr: 'torch.Tensor', A_col_ind: 'torch.Tensor', A_val: 'torch.Tensor', v: 'torch.Tensor') ->torch.Tensor:
        return _sparse_mat_vec_fwd_backend(ctx, num_cols, A_row_ptr, A_col_ind, A_val, v, mat_vec)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: 'Any', grad_output: 'torch.Tensor') ->Tuple[None, None, None, torch.Tensor, torch.Tensor]:
        A_grad, v_grad = _sparse_mat_vec_bwd_backend(ctx, grad_output, False)
        return None, None, None, A_grad, v_grad


sparse_mv = _SparseMvPAutograd.apply


class SparseLinearization(Linearization):

    def __init__(self, objective: 'Objective', ordering: 'Optional[VariableOrdering]'=None, **kwargs):
        super().__init__(objective, ordering)
        A_col_ind: 'List[int]' = []
        A_row_ptr: 'List[int]' = [0]
        cost_function_block_pointers = []
        cost_function_row_block_starts = []
        cost_function_stride = []
        for _, cost_function in enumerate(self.objective._get_jacobians_iter()):
            num_rows = cost_function.dim()
            col_slices_indices = []
            for var_idx_in_cost_function, variable in enumerate(cost_function.optim_vars):
                var_idx_in_order = self.ordering.index_of(cost_function.optim_var_at(var_idx_in_cost_function).name)
                var_start_col = self.var_start_cols[var_idx_in_order]
                num_cols = variable.dof()
                col_slice = slice(var_start_col, var_start_col + num_cols)
                col_slices_indices.append((col_slice, var_idx_in_cost_function))
            col_slices_indices.sort()
            sorted_block_sizes = [(s.stop - s.start) for s, _ in col_slices_indices]
            sorted_block_pointers = np.cumsum([0] + sorted_block_sizes)[:-1]
            sorted_indices = np.array([i for _, i in col_slices_indices])
            block_pointers: 'np.ndarray' = np.ndarray((len(col_slices_indices),), dtype=int)
            block_pointers[sorted_indices] = sorted_block_pointers
            cost_function_block_pointers.append(block_pointers)
            cost_function_row_block_starts.append(len(A_col_ind))
            col_ind = [c for s, _ in col_slices_indices for c in range(s.start, s.stop)]
            cost_function_stride.append(len(col_ind))
            for _ in range(num_rows):
                A_col_ind += col_ind
                A_row_ptr.append(len(A_col_ind))
        self.cost_function_block_pointers = cost_function_block_pointers
        self.cost_function_row_block_starts: 'np.ndarray' = np.array(cost_function_row_block_starts, dtype=int)
        self.cost_function_stride: 'np.ndarray' = np.array(cost_function_stride, dtype=int)
        self.A_row_ptr: 'np.ndarray' = np.array(A_row_ptr, dtype=int)
        self.A_col_ind: 'np.ndarray' = np.array(A_col_ind, dtype=int)
        self.A_val: 'torch.Tensor' = None
        self.b: 'torch.Tensor' = None
        self._Atb: 'torch.Tensor' = None
        self._AtA_diag: 'torch.Tensor' = None
        self.detached_hessian = False

    def _linearize_jacobian_impl(self):
        self._detached_hessian = False
        self._Atb = None
        self._AtA_diag = None
        self.A_val = torch.empty(size=(self.objective.batch_size, len(self.A_col_ind)), device=self.objective.device, dtype=self.objective.dtype)
        self.b = torch.empty(size=(self.objective.batch_size, self.num_rows), device=self.objective.device, dtype=self.objective.dtype)
        err_row_idx = 0
        for f_idx, cost_function in enumerate(self.objective._get_jacobians_iter()):
            jacobians, error = cost_function.weighted_jacobians_error()
            num_rows = cost_function.dim()
            row_slice = slice(err_row_idx, err_row_idx + num_rows)
            block_start = self.cost_function_row_block_starts[f_idx]
            stride = self.cost_function_stride[f_idx]
            block = self.A_val[:, block_start:block_start + stride * num_rows].view(-1, num_rows, stride)
            block_pointers = self.cost_function_block_pointers[f_idx]
            for var_idx_in_cost_function, var_jacobian in enumerate(jacobians):
                num_cols = var_jacobian.shape[2]
                pointer = block_pointers[var_idx_in_cost_function]
                block[:, :, pointer:pointer + num_cols] = var_jacobian
            self.b[:, row_slice] = -error
            err_row_idx += cost_function.dim()

    def structure(self):
        return SparseStructure(self.A_col_ind, self.A_row_ptr, self.num_rows, self.num_cols, dtype=np.float64 if self.objective.dtype == torch.double else np.float32)

    def _linearize_hessian_impl(self, _detach_hessian: 'bool'=False):
        self._linearize_jacobian_impl()
        self.detached_hessian = _detach_hessian

    def _ata_impl(self) ->torch.Tensor:
        raise NotImplementedError('AtA is not yet implemented for SparseLinearization.')

    def _atb_impl(self) ->torch.Tensor:
        if self._Atb is None:
            A_row_ptr = torch.tensor(self.A_row_ptr, dtype=torch.int32)
            A_col_ind = A_row_ptr.new_tensor(self.A_col_ind)
            self._Atb = sparse_mtv(self.num_cols, A_row_ptr, A_col_ind, self.A_val.double(), self.b.double()).unsqueeze(2)
        return self._Atb

    def Av(self, v: 'torch.Tensor') ->torch.Tensor:
        A_row_ptr = torch.tensor(self.A_row_ptr, dtype=torch.int32)
        A_col_ind = A_row_ptr.new_tensor(self.A_col_ind)
        return sparse_mv(self.num_cols, A_row_ptr, A_col_ind, self.A_val.double(), v.double())

    def diagonal_scaling(self, v: 'torch.Tensor') ->torch.Tensor:
        assert v.ndim == 2
        assert v.shape[1] == self.num_cols
        if self._AtA_diag is None:
            A_val = self.A_val
            self._AtA_diag = torch.zeros(A_val.shape[0], self.num_cols)
            for row in range(self.num_rows):
                start = self.A_row_ptr[row]
                end = self.A_row_ptr[row + 1]
                columns = self.A_col_ind[start:end]
                self._AtA_diag[:, columns] += A_val[:, start:end] ** 2
        return self._AtA_diag * v


def convert_to_alpha_beta_damping_tensors(damping: 'Union[float, torch.Tensor]', damping_eps: 'float', ellipsoidal_damping: 'bool', batch_size: 'int', device: 'DeviceType', dtype: 'torch.dtype') ->Tuple[torch.Tensor, torch.Tensor]:
    damping = torch.as_tensor(damping)
    if damping.ndim > 1:
        raise ValueError('Damping must be a float or a 1-D tensor.')
    if damping.ndim == 0 or damping.shape[0] == 1 and batch_size != 1:
        damping = damping.repeat(batch_size)
    return (damping, damping_eps * torch.ones_like(damping)) if ellipsoidal_damping else (torch.zeros_like(damping), damping)


class NonlinearOptimizerStatus(Enum):
    START = 0
    CONVERGED = 1
    MAX_ITERATIONS = 2
    FAIL = -1


def error_squared_norm_fn(error_vector: 'torch.Tensor') ->torch.Tensor:
    return (error_vector ** 2).sum(dim=1) / 2


def _instantiate_dlm_bwd_objective(objective: 'Objective'):
    bwd_objective = objective.copy()
    epsilon_var = Variable(torch.ones(1, 1, dtype=bwd_objective.dtype, device=bwd_objective.device), name=TheseusLayerDLMForward._DLM_EPSILON_STR)
    unit_weight = ScaleCostWeight(1.0)
    unit_weight
    for name, var in bwd_objective.optim_vars.items():
        grad_var = Variable(torch.zeros_like(var.tensor), name=name + TheseusLayerDLMForward._GRAD_SUFFIX)
        bwd_objective.add(_DLMPerturbation(var, epsilon_var, grad_var, unit_weight, name='dlm_perturbation' + name))
    bwd_optimizer = GaussNewton(bwd_objective, max_iterations=1, step_size=1.0)
    return bwd_objective, bwd_optimizer


def _rand_fill_(v: 'th.Variable', batch_size: 'int'):
    if isinstance(v, (th.SE2, th.SO3, th.SE3, th.SO3)):
        v.update(v.rand(batch_size, dtype=v.dtype, device=v.device).tensor)
    else:
        v.update(torch.rand((batch_size,) + v.shape[1:], dtype=v.dtype, device=v.device))


@torch.no_grad()
def check_jacobians(cf: 'th.CostFunction', num_checks: 'int'=1, tol: 'float'=0.001):
    optim_vars: 'List[th.Manifold]' = list(cf.optim_vars)
    aux_vars = list(cf.aux_vars)

    def autograd_fn(*optim_var_tensors):
        for v, t in zip(optim_vars, optim_var_tensors):
            v.update(t)
        return cf.error()
    with _tmp_tensors(optim_vars), _tmp_tensors(aux_vars):
        for _ in range(num_checks):
            for v in (optim_vars + aux_vars):
                _rand_fill_(v, 1)
            autograd_jac = torch.autograd.functional.jacobian(autograd_fn, tuple(v.tensor for v in optim_vars))
            jac, _ = cf.jacobians()
            for idx, v in enumerate(optim_vars):
                j1 = jac[idx][0]
                j2 = autograd_jac[idx]
                j2_sparse = j2[0, :, 0, :]
                j2_sparse_manifold = v.project(j2_sparse, is_sparse=True)
                if (j1 - j2_sparse_manifold).abs().max() > tol:
                    raise RuntimeError(f'Jacobian for variable {v.name} appears incorrect to the given tolerance.')


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze()


def lt_version(v1: 'str', v2: 'str') ->bool:

    def _as_tuple(s: 'str') ->Tuple[int, int, int]:
        pattern = '^[\\d.]+'
        match = re.match(pattern, s)
        try:
            return tuple(int(x) for x in match.group().split('.')[:3])
        except Exception:
            raise ValueError(f'String {s} cannot be converted to (mayor, minor, micro) format.')
    x1, y1, z1 = _as_tuple(v1)
    x2, y2, z2 = _as_tuple(v2)
    return x1 < x2 or x1 == x2 and y1 < y2 or x1 == x2 and y1 == y2 and z1 < z2


old_torch = lt_version(torch.__version__, '0.4.0')


class LML_Function(Function):

    @staticmethod
    def forward(ctx, x, N, eps, n_iter, branch, verbose):
        ctx.N = N
        ctx.eps = eps
        ctx.n_iter = n_iter
        ctx.branch = branch
        ctx.verbose = verbose
        branch = ctx.branch
        if branch is None:
            if not x.is_cuda:
                branch = 10
            else:
                branch = 100
        single = x.ndimension() == 1
        orig_x = x
        if single:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2
        n_batch, nx = x.shape
        if nx <= ctx.N:
            y = (1.0 - 1e-05) * torch.ones(n_batch, nx).type_as(x)
            if single:
                y = y.squeeze(0)
            if old_torch:
                ctx.save_for_backward(orig_x)
                ctx.y = y
                ctx.nu = torch.Tensor()
            else:
                ctx.save_for_backward(orig_x, y, torch.Tensor())
            return y
        x_sorted, _ = torch.sort(x, dim=1, descending=True)
        nu_lower = -x_sorted[:, ctx.N - 1] - 7.0
        nu_upper = -x_sorted[:, ctx.N] + 7.0
        ls = torch.linspace(0, 1, branch).type_as(x)
        for i in range(ctx.n_iter):
            r = nu_upper - nu_lower
            I = r > ctx.eps
            n_update = I.sum()
            if n_update == 0:
                break
            Ix = I.unsqueeze(1).expand_as(x) if old_torch else I
            nus = r[I].unsqueeze(1) * ls + nu_lower[I].unsqueeze(1)
            _xs = x[Ix].view(n_update, 1, nx) + nus.unsqueeze(2)
            fs = torch.sigmoid(_xs).sum(dim=2) - ctx.N
            i_lower = ((fs < 0).sum(dim=1) - 1).long()
            J = i_lower < 0
            if J.sum() > 0:
                None
                i_lower[J] = 0
            i_upper = i_lower + 1
            nu_lower[I] = nus.gather(1, i_lower.unsqueeze(1)).squeeze()
            nu_upper[I] = nus.gather(1, i_upper.unsqueeze(1)).squeeze()
            if J.sum() > 0:
                nu_lower[J] -= 7.0
        if ctx.verbose >= 0 and np.any(I.cpu().numpy()):
            None
        nu = nu_lower + r / 2.0
        y = torch.sigmoid(x + nu.unsqueeze(1))
        if single:
            y = y.squeeze(0)
        if old_torch:
            ctx.save_for_backward(orig_x)
            ctx.y = y
            ctx.nu = nu
        else:
            ctx.save_for_backward(orig_x, y, nu)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if old_torch:
            x, = ctx.saved_tensors
            y = ctx.y
            nu = ctx.nu
        else:
            x, y, nu = ctx.saved_tensors
        single = x.ndimension() == 1
        if single:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)
        assert x.ndimension() == 2
        assert y.ndimension() == 2
        assert grad_output.ndimension() == 2
        n_batch, nx = x.shape
        if nx <= ctx.N:
            dx = torch.zeros_like(x)
            if single:
                dx = dx.squeeze()
            grads = tuple([dx] + [None] * 5)
            return grads
        Hinv = 1.0 / (1.0 / y + 1.0 / (1.0 - y))
        dnu = bdot(Hinv, grad_output) / Hinv.sum(dim=1)
        dx = -Hinv * (-grad_output + dnu.unsqueeze(1))
        if single:
            dx = dx.squeeze()
        grads = tuple([dx] + [None] * 5)
        return grads


class LML(Module):

    def __init__(self, N, eps=0.0001, n_iter=100, branch=None, verbose=0):
        super().__init__()
        self.N = N
        self.eps = eps
        self.n_iter = n_iter
        self.branch = branch
        self.verbose = verbose

    def forward(self, x):
        return LML_Function.apply(x, self.N, self.eps, self.n_iter, self.branch, self.verbose)


class _ScalarModel(nn.Module):

    def __init__(self, hidden_size: 'int'):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self):
        dummy = torch.ones(1, 1)
        return self.layers(dummy)


class _OrderOfMagnitudeModel(nn.Module):

    def __init__(self, hidden_size: 'int', max_order: 'int'):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, max_order), nn.ReLU())
        self.register_buffer('magnitudes', (10 ** torch.arange(max_order)).unsqueeze(0))

    def forward(self):
        dummy = torch.ones(1, 1)
        mag_weights = self.layers(dummy).softmax(dim=1)
        return (mag_weights * self.magnitudes).sum(dim=1, keepdim=True)


class ScalarCollisionWeightModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = _OrderOfMagnitudeModel(10, 5)

    def forward(self, _: 'Dict[str, torch.Tensor]'):
        return {'collision_w': self.model()}


class ScalarCollisionWeightAndCostEpstModel(nn.Module):

    def __init__(self, robot_radius: 'float'):
        super().__init__()
        self.collision_weight_model = _OrderOfMagnitudeModel(200, 5)
        self.safety_dist_model = _ScalarModel(100)
        self.robot_radius = robot_radius

    def forward(self, _: 'Dict[str, torch.Tensor]'):
        collision_w = self.collision_weight_model()
        safety_dist = self.safety_dist_model().sigmoid()
        return {'collision_w': collision_w, 'cost_eps': safety_dist + self.robot_radius}


class InitialTrajectoryModel(nn.Module):

    def __init__(self, planner: 'MotionPlanner', max_num_images: 'int'=1000, hid_size: 'int'=200):
        super().__init__()
        self.aux_motion_planner = planner.copy(collision_weight=0.0)
        self.layers_u = nn.Sequential(nn.Linear(2 * max_num_images, hid_size), nn.ReLU(), nn.Linear(hid_size, hid_size), nn.ReLU(), nn.Linear(hid_size, 4 * (planner.objective.num_time_steps + 1)))
        self.bend_factor = nn.Sequential(nn.Linear(2 * max_num_images, hid_size), nn.ReLU(), nn.Linear(hid_size, 1), nn.Tanh())

        def init_weights(m_):
            if isinstance(m_, nn.Linear):
                torch.nn.init.normal_(m_.weight)
                torch.nn.init.normal_(m_.bias)
        self.bend_factor.apply(init_weights)
        self.dt = planner.objective.total_time / planner.objective.num_time_steps
        self.num_images = max_num_images

    def forward(self, batch: 'Dict[str, Any]'):
        device = self.aux_motion_planner.objective.device
        start = batch['expert_trajectory'][:, :2, 0]
        goal = batch['expert_trajectory'][:, :2, -1]
        one_hot_dummy = torch.zeros(start.shape[0], self.num_images * 2)
        file_ids = batch['file_id']
        for batch_idx, fi in enumerate(file_ids):
            idx = int(fi.split('_')[1]) + int('forest' in fi) * self.num_images
            one_hot_dummy[batch_idx, idx] = 1
        trajectory_len = self.aux_motion_planner.objective.trajectory_len
        dist_vec = goal - start
        pos_incr_per_step = dist_vec / (trajectory_len - 1)
        trajectory = torch.zeros(start.shape[0], 4 * trajectory_len)
        trajectory[:, :2] = start
        for t_step in range(1, trajectory_len):
            idx = 4 * t_step
            trajectory[:, idx:idx + 2] = trajectory[:, idx - 4:idx - 2] + pos_incr_per_step
        bend_factor = self.bend_factor(one_hot_dummy)
        start_goal_dist = dist_vec.norm(dim=1)
        cur_t = torch.zeros_like(start_goal_dist) - start_goal_dist / 2
        c = (start_goal_dist / 2) ** 2
        angle = th.SO2(theta=torch.ones(dist_vec.shape[0], 1) * np.pi / 2)
        normal_vector = angle.rotate(th.Point2(tensor=dist_vec)).tensor
        normal_vector /= normal_vector.norm(dim=1, keepdim=True)
        for t_step in range(1, trajectory_len):
            idx = 4 * t_step
            cur_t += start_goal_dist / (trajectory_len - 1)
            add = 2 * bend_factor * ((cur_t ** 2 - c) / c).view(-1, 1)
            trajectory[:, idx:idx + 2] += normal_vector * add
        for t_step in range(1, trajectory_len):
            idx = 4 * t_step
            trajectory[:, idx + 2:idx + 4] = (trajectory[:, idx:idx + 2] - trajectory[:, idx - 4:idx - 2]) / self.dt
        with torch.no_grad():
            planner_inputs = {'sdf_origin': batch['sdf_origin'], 'start': start, 'goal': goal, 'cell_size': batch['cell_size'], 'sdf_data': batch['sdf_data']}
            self.aux_motion_planner.objective.update(planner_inputs)
            motion_optimizer = cast(th.NonlinearLeastSquares, self.aux_motion_planner.layer.optimizer)
            linearization = cast(th.DenseLinearization, motion_optimizer.linear_solver.linearization)
            for var in linearization.ordering:
                var_type, time_idx = var.name.split('_')
                assert var_type in ['pose', 'vel']
                if var_type == 'pose':
                    traj_idx = int(time_idx) * 4
                if var_type == 'vel':
                    traj_idx = int(time_idx) * 4 + 2
                var.update(trajectory[:, traj_idx:traj_idx + 2])
            linearization.linearize()
            cov_matrix = torch.inverse(linearization.AtA)
            lower_cov = torch.linalg.cholesky(cov_matrix)
        u = self.layers_u(one_hot_dummy).unsqueeze(2)
        initial_trajectory = trajectory.unsqueeze(2) + torch.matmul(lower_cov, u)
        values: 'Dict[str, torch.Tensor]' = {}
        for t_step in range(trajectory_len):
            idx = 4 * t_step
            values[f'pose_{t_step}'] = initial_trajectory[:, idx:idx + 2, 0]
            values[f'vel_{t_step}'] = initial_trajectory[:, idx + 2:idx + 4, 0]
        return values


class TactileMeasModel(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x1: 'torch.Tensor', x2: 'torch.Tensor', k: 'torch.Tensor'):
        x = torch.cat([x1, x2], dim=1)
        k1_ = k.unsqueeze(1)
        x1_ = x.unsqueeze(-1)
        x = torch.mul(x1_, k1_)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


class TactileWeightModel(nn.Module):

    def __init__(self, device: 'th.DeviceType', dim: 'int'=3, wt_init: 'Optional[torch.Tensor]'=None):
        super().__init__()
        wt_init_ = torch.rand(1, dim)
        if wt_init is not None:
            wt_init_ = wt_init
        self.param = nn.Parameter(wt_init_)
        self

    def forward(self):
        return self.param.clone()


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (LML,
     lambda: ([], {'N': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ScalarCollisionWeightAndCostEpstModel,
     lambda: ([], {'robot_radius': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScalarCollisionWeightModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SimpleCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (SimpleNN,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TactileWeightModel,
     lambda: ([], {'device': 0}),
     lambda: ([], {})),
    (_OrderOfMagnitudeModel,
     lambda: ([], {'hidden_size': 4, 'max_order': 4}),
     lambda: ([], {})),
    (_ScalarModel,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([], {})),
]

