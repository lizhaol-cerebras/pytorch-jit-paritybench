import sys
_module = sys.modules[__name__]
del sys
utils = _module
einops = _module
_backends = _module
_torch_specific = _module
array_api = _module
einops = _module
experimental = _module
indexing = _module
layers = _module
_einmix = _module
flax = _module
keras = _module
oneflow = _module
paddle = _module
tensorflow = _module
packing = _module
parsing = _module
tests = _module
run_tests = _module
test_einsum = _module
test_examples = _module
test_layers = _module
test_ops = _module
test_other = _module
test_packing = _module
test_parsing = _module
convert_readme = _module
converter = _module
setup = _module
test_notebooks = _module

from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import warnings


from typing import Dict


from typing import List


from typing import Tuple


import torch


import functools


import itertools


import string


import typing


from collections import OrderedDict


from typing import Set


from typing import Union


from typing import Callable


from typing import Optional


from typing import TypeVar


from typing import cast


from typing import Any


import numpy as np


import numpy


from collections import namedtuple


class EinopsError(RuntimeError):
    """Runtime error thrown by einops"""
    pass


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """

    def __init__(self, elementary_axes_lengths: 'List[int]', axis_name2elementary_axis: 'Dict[str, int]', input_composition_known_unknown: 'List[Tuple[List[int], List[int]]]', axes_permutation: 'List[int]', first_reduced_axis: 'int', added_axes: 'Dict[int, int]', output_composite_axes: 'List[List[int]]'):
        self.elementary_axes_lengths: 'List[int]' = elementary_axes_lengths
        self.axis_name2elementary_axis: 'Dict[str, int]' = axis_name2elementary_axis
        self.input_composition_known_unknown: 'List[Tuple[List[int], List[int]]]' = input_composition_known_unknown
        self.axes_permutation: 'List[int]' = axes_permutation
        self.first_reduced_axis: 'int' = first_reduced_axis
        self.added_axes: 'Dict[int, int]' = added_axes
        self.output_composite_axes: 'List[List[int]]' = output_composite_axes


class AnonymousAxis(object):
    """Important thing: all instances of this class are not equal to each other"""

    def __init__(self, value: 'str'):
        self.value = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise EinopsError('No need to create anonymous axis of length 1. Report this as an issue')
            else:
                raise EinopsError('Anonymous axis should have positive length, not {}'.format(self.value))

    def __repr__(self):
        return '{}-axis'.format(str(self.value))


class ParsedExpression:
    """
    non-mutable structure that contains information about one side of expression (e.g. 'b c (h w)')
    and keeps some information important for downstream
    """

    def __init__(self, expression: 'str', *, allow_underscore: bool=False, allow_duplicates: bool=False):
        self.has_ellipsis: 'bool' = False
        self.has_ellipsis_parenthesized: 'Optional[bool]' = None
        self.identifiers: 'Set[str]' = set()
        self.has_non_unitary_anonymous_axes: 'bool' = False
        self.composition: 'List[Union[List[str], str]]' = []
        if '.' in expression:
            if '...' not in expression:
                raise EinopsError('Expression may contain dots only inside ellipsis (...)')
            if str.count(expression, '...') != 1 or str.count(expression, '.') != 3:
                raise EinopsError('Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ')
            expression = expression.replace('...', _ellipsis)
            self.has_ellipsis = True
        bracket_group: 'Optional[List[str]]' = None

        def add_axis_name(x):
            if x in self.identifiers:
                if not (allow_underscore and x == '_') and not allow_duplicates:
                    raise EinopsError('Indexing expression contains duplicate dimension "{}"'.format(x))
            if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
                if not (is_number or is_axis_name):
                    raise EinopsError('Invalid axis identifier: {}\n{}'.format(x, reason))
                if is_number:
                    x = AnonymousAxis(x)
                self.identifiers.add(x)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([x])
                else:
                    bracket_group.append(x)
        current_identifier = None
        for char in expression:
            if char in '() ':
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == '(':
                    if bracket_group is not None:
                        raise EinopsError('Axis composition is one-level (brackets inside brackets not allowed)')
                    bracket_group = []
                elif char == ')':
                    if bracket_group is None:
                        raise EinopsError('Brackets are not balanced')
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ['_', _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise EinopsError("Unknown character '{}'".format(char))
        if bracket_group is not None:
            raise EinopsError('Imbalanced parentheses in expression: "{}"'.format(expression))
        if current_identifier is not None:
            add_axis_name(current_identifier)

    def flat_axes_order(self) ->List:
        result = []
        for composed_axis in self.composition:
            assert isinstance(composed_axis, list), 'does not work with ellipsis'
            for axis in composed_axis:
                result.append(axis)
        return result

    def has_composed_axes(self) ->bool:
        for axes in self.composition:
            if isinstance(axes, list) and len(axes) > 1:
                return True
        return False

    @staticmethod
    def check_axis_name_return_reason(name: 'str', allow_underscore: 'bool'=False) ->Tuple[bool, str]:
        if not str.isidentifier(name):
            return False, 'not a valid python identifier'
        elif name[0] == '_' or name[-1] == '_':
            if name == '_' and allow_underscore:
                return True, ''
            return False, 'axis name should should not start or end with underscore'
        else:
            if keyword.iskeyword(name):
                warnings.warn('It is discouraged to use axes names that are keywords: {}'.format(name), RuntimeWarning)
            if name in ['axis']:
                warnings.warn("It is discouraged to use 'axis' as an axis name and will raise an error in future", FutureWarning)
            return True, ''

    @staticmethod
    def check_axis_name(name: 'str') ->bool:
        """
        Valid axes names are python identifiers except keywords,
        and additionally should not start or end with underscore
        """
        is_valid, _reason = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid


_expected_axis_length = -99999


_reductions = 'min', 'max', 'sum', 'mean', 'prod', 'any', 'all'


_unknown_axis_length = -999999


@functools.lru_cache(256)
def _prepare_transformation_recipe(pattern: 'str', operation: 'Reduction', axes_names: 'Tuple[str, ...]', ndim: 'int') ->TransformRecipe:
    """Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left_str, rght_str = pattern.split('->')
    left = ParsedExpression(left_str)
    rght = ParsedExpression(rght_str)
    if not left.has_ellipsis and rght.has_ellipsis:
        raise EinopsError('Ellipsis found in right side, but not left side of a pattern {}'.format(pattern))
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise EinopsError('Ellipsis inside parenthesis in the left side is not allowed: {}'.format(pattern))
    if operation == 'rearrange':
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise EinopsError('Non-unitary anonymous axes are not supported in rearrange (exception is length 1)')
        difference = set.symmetric_difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError('Identifiers only on one side of expression (should be on both): {}'.format(difference))
    elif operation == 'repeat':
        difference = set.difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the left side of repeat: {}'.format(difference))
        axes_without_size = set.difference({ax for ax in rght.identifiers if not isinstance(ax, AnonymousAxis)}, {*left.identifiers, *axes_names})
        if len(axes_without_size) > 0:
            raise EinopsError('Specify sizes for new axes in repeat: {}'.format(axes_without_size))
    elif operation in _reductions or callable(operation):
        difference = set.difference(rght.identifiers, left.identifiers)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the right side of reduce {}: {}'.format(operation, difference))
    else:
        raise EinopsError('Unknown reduction {}. Expect one of {}.'.format(operation, _reductions))
    if left.has_ellipsis:
        n_other_dims = len(left.composition) - 1
        if ndim < n_other_dims:
            raise EinopsError(f'Wrong shape: expected >={n_other_dims} dims. Received {ndim}-dim tensor.')
        ellipsis_ndim = ndim - n_other_dims
        ell_axes = [(_ellipsis + str(i)) for i in range(ellipsis_ndim)]
        left_composition = []
        for composite_axis in left.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    left_composition.append([axis])
            else:
                left_composition.append(composite_axis)
        rght_composition = []
        for composite_axis in rght.composition:
            if composite_axis == _ellipsis:
                for axis in ell_axes:
                    rght_composition.append([axis])
            else:
                group = []
                for axis in composite_axis:
                    if axis == _ellipsis:
                        group.extend(ell_axes)
                    else:
                        group.append(axis)
                rght_composition.append(group)
        left.identifiers.update(ell_axes)
        left.identifiers.remove(_ellipsis)
        if rght.has_ellipsis:
            rght.identifiers.update(ell_axes)
            rght.identifiers.remove(_ellipsis)
    else:
        if ndim != len(left.composition):
            raise EinopsError(f'Wrong shape: expected {len(left.composition)} dims. Received {ndim}-dim tensor.')
        left_composition = left.composition
        rght_composition = rght.composition
    axis_name2known_length: 'Dict[Union[str, AnonymousAxis], int]' = OrderedDict()
    for composite_axis in left_composition:
        for axis_name in composite_axis:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
    repeat_axes_names = []
    for axis_name in rght.identifiers:
        if axis_name not in axis_name2known_length:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
            repeat_axes_names.append(axis_name)
    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}
    for elementary_axis in axes_names:
        if not ParsedExpression.check_axis_name(elementary_axis):
            raise EinopsError('Invalid name for an axis', elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise EinopsError('Axis {} is not used in transform'.format(elementary_axis))
        axis_name2known_length[elementary_axis] = _expected_axis_length
    input_axes_known_unknown = []
    for i, composite_axis in enumerate(left_composition):
        known: 'Set[str]' = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
        unknown: 'Set[str]' = {axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length}
        if len(unknown) > 1:
            raise EinopsError('Could not infer sizes for {}'.format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown]))
    axis_position_after_reduction: 'Dict[str, int]' = {}
    for axis_name in itertools.chain(*left_composition):
        if axis_name in rght.identifiers:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)
    result_axes_grouping: 'List[List[int]]' = [[axis_name2position[axis] for axis in composite_axis] for i, composite_axis in enumerate(rght_composition)]
    ordered_axis_left = list(itertools.chain(*left_composition))
    ordered_axis_rght = list(itertools.chain(*rght_composition))
    reduced_axes = [axis for axis in ordered_axis_left if axis not in rght.identifiers]
    order_after_transposition = [axis for axis in ordered_axis_rght if axis in left.identifiers] + reduced_axes
    axes_permutation = [ordered_axis_left.index(axis) for axis in order_after_transposition]
    added_axes = {i: axis_name2position[axis_name] for i, axis_name in enumerate(ordered_axis_rght) if axis_name not in left.identifiers}
    first_reduced_axis = len(order_after_transposition) - len(reduced_axes)
    return TransformRecipe(elementary_axes_lengths=list(axis_name2known_length.values()), axis_name2elementary_axis={axis: axis_name2position[axis] for axis in axes_names}, input_composition_known_unknown=input_axes_known_unknown, axes_permutation=axes_permutation, first_reduced_axis=first_reduced_axis, added_axes=added_axes, output_composite_axes=result_axes_grouping)


def _prepare_recipes_for_all_dims(pattern: 'str', operation: 'Reduction', axes_names: 'Tuple[str, ...]') ->Dict[int, TransformRecipe]:
    """
    Internal function, used in layers.
    Layer makes all recipe creation when it is initialized, thus to keep recipes simple we pre-compute for all dims
    """
    left_str, rght_str = pattern.split('->')
    left = ParsedExpression(left_str)
    dims = [len(left.composition)]
    if left.has_ellipsis:
        dims = [(len(left.composition) - 1 + ellipsis_dims) for ellipsis_dims in range(8)]
    return {ndim: _prepare_transformation_recipe(pattern, operation, axes_names, ndim=ndim) for ndim in dims}


class AbstractBackend:
    """Base backend class, major part of methods are only for debugging purposes."""
    framework_name: 'str'

    def is_appropriate_type(self, tensor):
        """helper method should recognize tensors it can handle"""
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop):
        raise NotImplementedError("framework doesn't implement arange")

    def shape(self, x):
        """shape should return a tuple with integers or "shape symbols" (which will evaluate to actual size)"""
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def reduce(self, x, operation, axes):
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: 'list'):
        raise NotImplementedError()

    def add_axis(self, x, new_position):
        raise NotImplementedError()

    def add_axes(self, x, n_axes, pos2len):
        repeats = [1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return self.tile(x, tuple(repeats))

    def tile(self, x, repeats):
        """repeats - same lengths as x.shape"""
        raise NotImplementedError()

    def concat(self, tensors, axis: 'int'):
        """concatenates tensors along axis.
        Assume identical across tensors: devices, dtypes and shapes except selected axis."""
        raise NotImplementedError()

    def is_float_type(self, x):
        raise NotImplementedError()

    def layers(self):
        raise NotImplementedError('backend does not provide layers')

    def __repr__(self):
        return '<einops backend for {}>'.format(self.framework_name)

    def einsum(self, pattern, *x):
        raise NotImplementedError('backend does not support einsum')


_debug_importing = False


def get_backend(tensor) ->'AbstractBackend':
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
    """
    _type = type(tensor)
    _result = _type2backend.get(_type, None)
    if _result is not None:
        return _result
    for framework_name, backend in list(_loaded_backends.items()):
        if backend.is_appropriate_type(tensor):
            _type2backend[_type] = backend
            return backend
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)
    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            None
        if BackendSubclass.framework_name not in _loaded_backends:
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    None
                backend = BackendSubclass()
                _loaded_backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    _type2backend[_type] = backend
                    return backend
    raise RuntimeError('Tensor type unknown to einops {}'.format(type(tensor)))


class ReduceMixin:
    """
    Reduce layer behaves identically to einops.reduce operation.

    :param pattern: str, rearrangement pattern
    :param reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
    :param axes_lengths: any additional specification of dimensions

    See einops.reduce for source_examples.
    """

    def __init__(self, pattern: 'str', reduction: 'str', **axes_lengths: Any):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths
        self._multirecipe = self.multirecipe()
        self._axes_lengths = tuple(self.axes_lengths.items())

    def __repr__(self):
        params = '{!r}, {!r}'.format(self.pattern, self.reduction)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    def multirecipe(self) ->Dict[int, TransformRecipe]:
        try:
            return _prepare_recipes_for_all_dims(self.pattern, operation=self.reduction, axes_names=tuple(self.axes_lengths))
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        backend = get_backend(x)
        return _apply_recipe(backend=backend, recipe=self._multirecipe[len(x.shape)], tensor=x, reduction_type=self.reduction, axes_lengths=self._axes_lengths)

    def __getstate__(self):
        return {'pattern': self.pattern, 'reduction': self.reduction, 'axes_lengths': self.axes_lengths}

    def __setstate__(self, state):
        self.__init__(pattern=state['pattern'], reduction=state['reduction'], **state['axes_lengths'])


class TorchJitBackend:
    """
    Completely static backend that mimics part of normal backend functionality
    but restricted to be within torchscript.
    """

    @staticmethod
    def reduce(x: 'torch.Tensor', operation: 'str', reduced_axes: 'List[int]'):
        if operation == 'min':
            return x.amin(dim=reduced_axes)
        elif operation == 'max':
            return x.amax(dim=reduced_axes)
        elif operation == 'sum':
            return x.sum(dim=reduced_axes)
        elif operation == 'mean':
            return x.mean(dim=reduced_axes)
        elif operation == 'prod':
            for i in list(sorted(reduced_axes))[::-1]:
                x = x.prod(dim=i)
            return x
        else:
            raise NotImplementedError('Unknown reduction ', operation)

    @staticmethod
    def transpose(x, axes: 'List[int]'):
        return x.permute(axes)

    @staticmethod
    def stack_on_zeroth_dimension(tensors: 'List[torch.Tensor]'):
        return torch.stack(tensors)

    @staticmethod
    def tile(x, repeats: 'List[int]'):
        return x.repeat(repeats)

    @staticmethod
    def add_axes(x, n_axes: 'int', pos2len: 'Dict[int, int]'):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = torch.unsqueeze(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    @staticmethod
    def is_float_type(x):
        return x.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def reshape(x, shape: 'List[int]'):
        return x.reshape(shape)


CookedRecipe = Tuple[Optional[List[int]], Optional[List[int]], List[int], Dict[int, int], Optional[List[int]], int]


def _product(sequence: 'List[int]') ->int:
    """minimalistic product that works both with numbers and symbols. Supports empty lists"""
    result = 1
    for element in sequence:
        result *= element
    return result


def _reconstruct_from_shape_uncached(self: 'TransformRecipe', shape: 'List[int]', axes_dims: 'FakeHashableAxesLengths') ->CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, theano) and UnknownSize (tf, previously mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    need_init_reshape = False
    axes_lengths: 'List[int]' = list(self.elementary_axes_lengths)
    for axis, dim in axes_dims:
        axes_lengths[self.axis_name2elementary_axis[axis]] = dim
    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composition_known_unknown):
        length = shape[input_axis]
        if len(known_axes) == 0 and len(unknown_axes) == 1:
            axes_lengths[unknown_axes[0]] = length
            continue
        known_product = 1
        for axis in known_axes:
            known_product *= axes_lengths[axis]
        if len(unknown_axes) == 0:
            if isinstance(length, int) and isinstance(known_product, int) and length != known_product:
                raise EinopsError(f'Shape mismatch, {length} != {known_product}')
        else:
            if isinstance(length, int) and isinstance(known_product, int) and length % known_product != 0:
                raise EinopsError(f"Shape mismatch, can't divide axis of length {length} in chunks of {known_product}")
            unknown_axis = unknown_axes[0]
            inferred_length: 'int' = length // known_product
            axes_lengths[unknown_axis] = inferred_length
        if len(known_axes) + len(unknown_axes) != 1:
            need_init_reshape = True
    init_shapes: 'Optional[List[int]]' = axes_lengths[:len(self.axes_permutation)] if need_init_reshape else None
    need_final_reshape = False
    final_shapes: 'List[int]' = []
    for grouping in self.output_composite_axes:
        lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
        final_shapes.append(_product(lengths))
        if len(lengths) != 1:
            need_final_reshape = True
    added_axes: 'Dict[int, int]' = {pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()}
    reduced_axes = list(range(self.first_reduced_axis, len(self.axes_permutation)))
    n_axes_after_adding_axes = len(added_axes) + len(self.axes_permutation)
    axes_reordering: 'Optional[List[int]]' = self.axes_permutation
    if self.axes_permutation == list(range(len(self.axes_permutation))):
        axes_reordering = None
    _final_shapes = final_shapes if need_final_reshape else None
    return init_shapes, axes_reordering, reduced_axes, added_axes, _final_shapes, n_axes_after_adding_axes


def apply_for_scriptable_torch(recipe: 'TransformRecipe', tensor: 'torch.Tensor', reduction_type: 'str', axes_dims: 'List[Tuple[str, int]]') ->torch.Tensor:
    backend = TorchJitBackend
    init_shapes, axes_reordering, reduced_axes, added_axes, final_shapes, n_axes_w_added = _reconstruct_from_shape_uncached(recipe, backend.shape(tensor), axes_dims=axes_dims)
    if init_shapes is not None:
        tensor = backend.reshape(tensor, init_shapes)
    if axes_reordering is not None:
        tensor = backend.transpose(tensor, axes_reordering)
    if len(reduced_axes) > 0:
        tensor = backend.reduce(tensor, operation=reduction_type, reduced_axes=reduced_axes)
    if len(added_axes) > 0:
        tensor = backend.add_axes(tensor, n_axes=n_axes_w_added, pos2len=added_axes)
    if final_shapes is not None:
        tensor = backend.reshape(tensor, final_shapes)
    return tensor


class Reduce(ReduceMixin, torch.nn.Module):

    def forward(self, input):
        recipe = self._multirecipe[input.ndim]
        return apply_for_scriptable_torch(recipe, input, reduction_type=self.reduction, axes_dims=self._axes_lengths)

    def _apply_recipe(self, x):
        pass


class RearrangeMixin:
    """
    Rearrange layer behaves identically to einops.rearrange operation.

    :param pattern: str, rearrangement pattern
    :param axes_lengths: any additional specification of dimensions

    See einops.rearrange for source_examples.
    """

    def __init__(self, pattern: 'str', **axes_lengths: Any) ->None:
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        self._multirecipe = self.multirecipe()
        self._axes_lengths = tuple(self.axes_lengths.items())

    def __repr__(self) ->str:
        params = repr(self.pattern)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    def multirecipe(self) ->Dict[int, TransformRecipe]:
        try:
            return _prepare_recipes_for_all_dims(self.pattern, operation='rearrange', axes_names=tuple(self.axes_lengths))
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        backend = get_backend(x)
        return _apply_recipe(backend=backend, recipe=self._multirecipe[len(x.shape)], tensor=x, reduction_type='rearrange', axes_lengths=self._axes_lengths)

    def __getstate__(self):
        return {'pattern': self.pattern, 'axes_lengths': self.axes_lengths}

    def __setstate__(self, state):
        self.__init__(pattern=state['pattern'], **state['axes_lengths'])


class Rearrange(RearrangeMixin, torch.nn.Module):

    def forward(self, input):
        recipe = self._multirecipe[input.ndim]
        return apply_for_scriptable_torch(recipe, input, reduction_type='rearrange', axes_dims=self._axes_lengths)

    def _apply_recipe(self, x):
        pass


def _report_axes(axes: 'set', report_message: 'str'):
    if len(axes) > 0:
        raise EinopsError(report_message.format(axes))


class _EinmixMixin:

    def __init__(self, pattern: 'str', weight_shape: 'str', bias_shape: 'Optional[str]'=None, **axes_lengths: Any):
        """
        EinMix - Einstein summation with automated tensor management and axis packing/unpacking.

        EinMix is an advanced tool, helpful tutorial:
        https://github.com/arogozhnikov/einops/blob/master/docs/3-einmix-layer.ipynb

        Imagine taking einsum with two arguments, one of each input, and one - tensor with weights
        >>> einsum('time batch channel_in, channel_in channel_out -> time batch channel_out', input, weight)

        This layer manages weights for you, syntax highlights separate role of weight matrix
        >>> EinMix('time batch channel_in -> time batch channel_out', weight_shape='channel_in channel_out')
        But otherwise it is the same einsum under the hood.

        Simple linear layer with bias term (you have one like that in your framework)
        >>> EinMix('t b cin -> t b cout', weight_shape='cin cout', bias_shape='cout', cin=10, cout=20)
        There is no restriction to mix the last axis. Let's mix along height
        >>> EinMix('h w c-> hout w c', weight_shape='h hout', bias_shape='hout', h=32, hout=32)
        Channel-wise multiplication (like one used in normalizations)
        >>> EinMix('t b c -> t b c', weight_shape='c', c=128)
        Multi-head linear layer (each head is own linear layer):
        >>> EinMix('t b (head cin) -> t b (head cout)', weight_shape='head cin cout', ...)

        ... and yes, you need to specify all dimensions of weight shape/bias shape in parameters.

        Use cases:
        - when channel dimension is not last, use EinMix, not transposition
        - patch/segment embeddings
        - when need only within-group connections to reduce number of weights and computations
        - perfect as a part of sequential models
        - next-gen MLPs (follow tutorial to learn more!)

        Uniform He initialization is applied to weight tensor. This accounts for number of elements mixed.

        Parameters
        :param pattern: transformation pattern, left side - dimensions of input, right side - dimensions of output
        :param weight_shape: axes of weight. A tensor of this shape is created, stored, and optimized in a layer
        :param bias_shape: axes of bias added to output. Weights of this shape are created and stored. If `None` (the default), no bias is added.
        :param axes_lengths: dimensions of weight tensor
        """
        super().__init__()
        self.pattern = pattern
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.axes_lengths = axes_lengths
        self.initialize_einmix(pattern=pattern, weight_shape=weight_shape, bias_shape=bias_shape, axes_lengths=axes_lengths)

    def initialize_einmix(self, pattern: 'str', weight_shape: 'str', bias_shape: 'Optional[str]', axes_lengths: 'dict'):
        left_pattern, right_pattern = pattern.split('->')
        left = ParsedExpression(left_pattern)
        right = ParsedExpression(right_pattern)
        weight = ParsedExpression(weight_shape)
        _report_axes(set.difference(right.identifiers, {*left.identifiers, *weight.identifiers}), 'Unrecognized identifiers on the right side of EinMix {}')
        if left.has_ellipsis or right.has_ellipsis or weight.has_ellipsis:
            raise EinopsError('Ellipsis is not supported in EinMix (right now)')
        if any(x.has_non_unitary_anonymous_axes for x in [left, right, weight]):
            raise EinopsError('Anonymous axes (numbers) are not allowed in EinMix')
        if '(' in weight_shape or ')' in weight_shape:
            raise EinopsError(f'Parenthesis is not allowed in weight shape: {weight_shape}')
        pre_reshape_pattern = None
        pre_reshape_lengths = None
        post_reshape_pattern = None
        if any(len(group) != 1 for group in left.composition):
            names: 'List[str]' = []
            for group in left.composition:
                names += group
            composition = ' '.join(names)
            pre_reshape_pattern = f'{left_pattern}->{composition}'
            pre_reshape_lengths = {name: length for name, length in axes_lengths.items() if name in names}
        if any(len(group) != 1 for group in right.composition):
            names = []
            for group in right.composition:
                names += group
            composition = ' '.join(names)
            post_reshape_pattern = f'{composition}->{right_pattern}'
        self._create_rearrange_layers(pre_reshape_pattern, pre_reshape_lengths, post_reshape_pattern, {})
        for axis in weight.identifiers:
            if axis not in axes_lengths:
                raise EinopsError('Dimension {} of weight should be specified'.format(axis))
        _report_axes(set.difference(set(axes_lengths), {*left.identifiers, *weight.identifiers}), 'Axes {} are not used in pattern')
        _report_axes(set.difference(weight.identifiers, {*left.identifiers, *right.identifiers}), 'Weight axes {} are redundant')
        if len(weight.identifiers) == 0:
            warnings.warn('EinMix: weight has no dimensions (means multiplication by a number)')
        _weight_shape = [axes_lengths[axis] for axis, in weight.composition]
        _fan_in = _product([axes_lengths[axis] for axis, in weight.composition if axis not in right.identifiers])
        if bias_shape is not None:
            if not isinstance(bias_shape, str):
                raise EinopsError('bias shape should be string specifying which axes bias depends on')
            bias = ParsedExpression(bias_shape)
            _report_axes(set.difference(bias.identifiers, right.identifiers), 'Bias axes {} not present in output')
            _report_axes(set.difference(bias.identifiers, set(axes_lengths)), 'Sizes not provided for bias axes {}')
            _bias_shape = []
            for axes in right.composition:
                for axis in axes:
                    if axis in bias.identifiers:
                        _bias_shape.append(axes_lengths[axis])
                    else:
                        _bias_shape.append(1)
        else:
            _bias_shape = None
        weight_bound = (3 / _fan_in) ** 0.5
        bias_bound = (1 / _fan_in) ** 0.5
        self._create_parameters(_weight_shape, weight_bound, _bias_shape, bias_bound)
        mapped_identifiers = {*left.identifiers, *right.identifiers, *weight.identifiers}
        mapping2letters = {k: letter for letter, k in zip(string.ascii_lowercase, mapped_identifiers)}

        def write_flat(axes: 'list'):
            return ''.join(mapping2letters[axis] for axis in axes)
        self.einsum_pattern: 'str' = '{},{}->{}'.format(write_flat(left.flat_axes_order()), write_flat(weight.flat_axes_order()), write_flat(right.flat_axes_order()))

    def _create_rearrange_layers(self, pre_reshape_pattern: 'Optional[str]', pre_reshape_lengths: 'Optional[Dict]', post_reshape_pattern: 'Optional[str]', post_reshape_lengths: 'Optional[Dict]'):
        raise NotImplementedError('Should be defined in framework implementations')

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        """Shape and implementations"""
        raise NotImplementedError('Should be defined in framework implementations')

    def __repr__(self):
        params = repr(self.pattern)
        params += f", '{self.weight_shape}'"
        if self.bias_shape is not None:
            params += f", '{self.bias_shape}'"
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)


class EinMix(_EinmixMixin, torch.nn.Module):

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = torch.nn.Parameter(torch.zeros(weight_shape).uniform_(-weight_bound, weight_bound), requires_grad=True)
        if bias_shape is not None:
            self.bias = torch.nn.Parameter(torch.zeros(bias_shape).uniform_(-bias_bound, bias_bound), requires_grad=True)
        else:
            self.bias = None

    def _create_rearrange_layers(self, pre_reshape_pattern: 'Optional[str]', pre_reshape_lengths: 'Optional[Dict]', post_reshape_pattern: 'Optional[str]', post_reshape_lengths: 'Optional[Dict]'):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **cast(dict, pre_reshape_lengths))
        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **cast(dict, post_reshape_lengths))

    def forward(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = torch.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result

