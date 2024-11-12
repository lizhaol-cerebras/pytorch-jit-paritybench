
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


from typing import List


from typing import Dict


import torch


import torch.distributed as dist


import torch.optim as optim


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torch.distributed.pipelining import pipeline


from torch.distributed.pipelining import PipelineStage


from torch.distributed.pipelining import ScheduleGPipe


from torch.distributed.pipelining import SplitPoint


import logging


from torch.profiler import profile


from torch.profiler import ProfilerActivity


from torch.distributed._tensor import DeviceMesh


from torch.distributed.tensor.parallel import parallelize_module


from collections import deque


from typing import Any


from typing import Deque


from typing import Optional


from typing import Tuple


from typing import Union


import torch.nn as nn


from typing import Callable


import torch.fx as fx


from abc import ABC


from abc import abstractmethod


from collections import defaultdict


from torch.profiler import record_function


import copy


from enum import Enum


from inspect import Parameter


from inspect import signature


from inspect import Signature


from types import MethodType


from torch.export import ExportedProgram


from torch.fx.node import map_aggregate


from torch.fx.passes.split_module import split_module


from torch._subclasses.fake_tensor import FakeTensor


from torch.distributed._composable.fsdp.fully_shard import FSDPModule


from torch.nn.parallel import DistributedDataParallel


from typing import cast


import torch.fx._pytree as fx_pytree


import torch.utils._pytree as pytree


from torch.export.exported_program import ConstantArgument


from torch.export.exported_program import ModuleCallSignature


from torch.export.exported_program import SymIntArgument


from torch.export.exported_program import TensorArgument


from torch.export.unflatten import InterpreterModule


from torch import fx


import time


from torch import ones


from torch import zeros


from torch.utils._pytree import tree_flatten


from torch.utils._pytree import tree_unflatten


from itertools import chain


from torch import nn


import inspect


import warnings


from torch.distributed._composable.fsdp.fully_shard import fully_shard


from torch.distributed._composable.fsdp.fully_shard import MixedPrecisionPolicy


from torch.distributed._tensor import DTensor


from torch.distributed.device_mesh import DeviceMesh


from torch.distributed.device_mesh import init_device_mesh


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.testing._internal.common_distributed import MultiProcessTestCase


from torch.testing._internal.common_utils import FILE_SCHEMA


from torch.testing._internal.common_utils import instantiate_parametrized_tests


from torch.testing._internal.common_utils import parametrize


from typing import NamedTuple


import random


from torch.testing._internal.common_distributed import skip_if_lt_x_gpu


import torch.multiprocessing as mp


from torch.distributed._tensor.device_mesh import init_device_mesh


class MyNetworkBlock(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


in_dim = 512


layer_dims = [512, 1024, 256]


out_dim = 10


class MyNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.num_layers = len(layer_dims)
        prev_dim = in_dim
        for i, dim in enumerate(layer_dims):
            super().add_module(f'layer{i}', MyNetworkBlock(prev_dim, dim))
            prev_dim = dim
        self.output_proj = torch.nn.Linear(layer_dims[-1], out_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            layer = getattr(self, f'layer{i}')
            x = layer(x)
        return self.output_proj(x)


class ModelChunk0(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer0 = MyNetworkBlock(in_dim, layer_dims[0])

    def forward(self, x):
        return self.layer0(x)


class ModelChunk1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = MyNetworkBlock(layer_dims[0], layer_dims[1])

    def forward(self, x):
        return self.layer1(x)


class ModelChunk2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer2 = MyNetworkBlock(layer_dims[1], layer_dims[2])
        self.output_proj = torch.nn.Linear(layer_dims[2], out_dim)

    def forward(self, x):
        x = self.layer2(x)
        return self.output_proj(x)


batch_size = 4


d_hid = 16


def pipe_split():
    return torch.ops.pippy._pipe_split()


class ExampleCode(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y=torch.zeros(batch_size, d_hid)):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = torch.relu(x)
        skip_conn = x
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        pipe_split()
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = x + skip_conn
        x = self.lin2(x)
        x = torch.relu(x)
        return x


class ExpertLayer(torch.nn.Module):

    def __init__(self, d_hid) ->None:
        super(ExpertLayer, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x) ->torch.Tensor:
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class MoE(torch.nn.Module):

    def __init__(self, n_experts: 'int') ->None:
        super().__init__()
        self.pre_proc = torch.nn.Linear(d_hid, d_hid)
        self.experts = torch.nn.ModuleList([ExpertLayer(d_hid) for _ in range(n_experts)])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.pre_proc(x)
        outputs = []
        for expert in self.experts:
            outputs.append(expert(x))
        return torch.cat(outputs, dim=1)


class MLPModule(torch.nn.Module):

    def __init__(self, d_hid):
        super(MLPModule, self).__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class IterationBlock(torch.nn.Module):

    def __init__(self, d_hid):
        super().__init__()
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.lin(x)
        x = torch.relu(x)
        return x


class IterativeNetwork(torch.nn.Module):

    def __init__(self, d_hid, num_iters):
        super().__init__()
        self.num_iters = num_iters
        self.iter_block = IterationBlock(d_hid)
        self.output_proj = torch.nn.Linear(d_hid, 10)

    def forward(self, x):
        for i in range(self.num_iters):
            x = self.iter_block(x)
        return self.output_proj(x)


class PipeSequential(torch.nn.Sequential):

    @staticmethod
    def from_sequential(sequential_instance: 'torch.nn.Sequential'):
        return PipeSequential(*[copy.copy(m) for m in sequential_instance])

    def forward(self, input):
        for i, module in enumerate(self):
            input = module(input)
            if i != len(self) - 1:
                pipe_split()
        return input


class LossWrapper(torch.nn.Module):
    """
    LossWrapper is a convenient abstract class that allows you to wrap up both
    your model as well as its loss function and specify the connectivity between
    the inputs, model, loss function, and output value. Example::

        class MyModelWrapper(LossWrapper):
            def forward(self, x, targets):
                model_out = self.module(x)
                loss_value = self.loss_fn(model_out, targets)
                return loss_value

    The above example defines a connectivity where we expect the forward/loss/backward
    training procedure to take two arguments (x and targets), pass x into the module
    to get the output of the feedforward computation, pass the model output and the
    targets value into the loss function, and get and return the loss value, which will
    be backpropagated by PiPPy. The above class would then be instantiated like::

        model = ... # instantiate the model
        loss_fn = torch.nn.MSELoss() # for the sake of demonstration

        wrapper = MyModelWrapper(model, loss_fn)
        pipe = Pipe.from_tracing(wrapper, ...)

    """

    def __init__(self, module, loss_fn):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        raise NotImplementedError('This instance of LossWrapper does not have an overriddenforward(). Please implement forward() to specify the arguments, connection between the module and loss, and loss output value.')


class TrivialLossWrapper(LossWrapper):

    def forward(self, x, targets):
        model_out = self.module(x)
        return self.loss_fn(model_out, targets)
    loss_spec = True


def friendly_debug_info(v):
    """
    Helper function to print out debug info in a friendly way.
    """
    if isinstance(v, torch.Tensor):
        return f'Tensor({v.shape}, grad={v.requires_grad})'
    else:
        return str(v)


def map_debug_info(a):
    """
    Helper function to apply `friendly_debug_info` to items in `a`.
    `a` may be a list, tuple, or dict.
    """
    return torch.fx.node.map_aggregate(a, friendly_debug_info)


def stage_backward(stage_output, output_grads, input_values, outputs_with_grads_idxs: 'Optional[List[int]]'=None):
    """
    This is a helper function to:
    1. compute the gradients for the stage inputs, and
    2. accumulate gradients for the stage module's parameters.

    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """
    if outputs_with_grads_idxs is not None:
        stage_output = [stage_output[i] for i in outputs_with_grads_idxs]
        output_grads = [output_grads[i] for i in outputs_with_grads_idxs]
    try:
        stage_output_tensors = []
        output_grad_tensors = []

        def extract_tensors_with_grads(output_val, grad_val):
            if isinstance(output_val, torch.Tensor):
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                assert isinstance(grad_val, (torch.Tensor, type(None))), f'Expected Tensor or None gradient but got {type(grad_val)}'
                stage_output_tensors.append(output_val)
                output_grad_tensors.append(grad_val)
            elif isinstance(output_val, (tuple, list)):
                if grad_val is None:
                    return
                assert isinstance(grad_val, (tuple, list)), f'grad_value expected to have type {type(output_val)} but got {type(grad_val)}'
                assert len(output_val) == len(grad_val)
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(ov, gv)
            elif isinstance(output_val, dict):
                if grad_val is None:
                    return
                assert isinstance(grad_val, dict)
                assert set(output_val.keys()) == set(grad_val.keys())
                for k in output_val.keys():
                    extract_tensors_with_grads(output_val[k], grad_val[k])
            else:
                pass
        extract_tensors_with_grads(stage_output, output_grads)
        torch.autograd.backward(stage_output_tensors, grad_tensors=output_grad_tensors)
        grad_inputs = []
        for val in input_values:
            if isinstance(val, torch.Tensor):
                grad_inputs.append(val.grad)
            else:
                grad_inputs.append(None)
        """
        inputs_with_grad = []
        for val in input_values:
            if isinstance(val, torch.Tensor) and val.requires_grad:
                inputs_with_grad.append(val)

        grad_inputs = torch.autograd.grad(
            stage_output_tensors, inputs_with_grad, output_grad_tensors,  # type: ignore[arg-type]
        )
        """
    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward:
        Stage output: {map_debug_info(stage_output)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e
    return grad_inputs


class DetachExecutor(fx.Interpreter):
    """
    Special interpreter to run the split_gm in testing that detaches all inputs to
    a module invocation. This is needed so that the values at the boundary are
    leaf modules in autograd execution.
    """

    def __init__(self, module, garbage_collect_values=True):
        garbage_collect_values = False
        super().__init__(module, garbage_collect_values)
        self.value_remap = {}

    def run(self, *args, initial_env=None):
        self.value_remap = {}
        return super().run(*args, initial_env=initial_env)

    def call_module(self, target, args, kwargs):

        def detach_tensors(a):
            if isinstance(a, torch.Tensor) and a.requires_grad:
                if a not in self.value_remap:
                    new_val = a.detach().requires_grad_(True)
                    self.value_remap[a] = new_val
                return self.value_remap[a]
            else:
                return a
        """
        def dont_traverse_size(a):
            return type(a) != torch.Size
        """
        args = map_aggregate(args, detach_tensors)
        kwargs = map_aggregate(kwargs, detach_tensors)
        return super().call_module(target, args, kwargs)

    def call_function(self, target, args, kwargs):
        if target == stage_backward:
            kwargs = dict(kwargs)
            kwargs['input_values'] = [self.value_remap.get(v, v) for v in kwargs['input_values']]
        return super().call_function(target, args, kwargs)


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2


class QualnameMapMixin:
    """
    A mixin class that helps a `Pipe` object to remap its qualnames back to
    original qualnames.
    """

    def __init__(self, splitter_qualname_map: 'Dict[str, str]'=None, tracer_qualname_map: 'Dict[str, str]'=None):
        self.new_to_old_qualname_mapping: 'Dict[str, str]' = splitter_qualname_map or {}
        self.tracer_qualname_map = tracer_qualname_map

    def remap_qualname(self, qualname: 'str'):
        if qualname.startswith('split_gm.'):
            qualname = qualname[len('split_gm.'):]
        name_before_split = None
        if qualname in self.new_to_old_qualname_mapping:
            name_before_split = self.new_to_old_qualname_mapping[qualname]
        else:
            split_names = qualname.rsplit('.', 1)
            leaf = split_names[-1]
            while len(split_names) > 1:
                prefix = split_names[0]
                if prefix in self.new_to_old_qualname_mapping:
                    old_prefix = self.new_to_old_qualname_mapping[prefix]
                    name_before_split = '.'.join([old_prefix, leaf])
                    break
                split_names = prefix.rsplit('.', 1)
                leaf = '.'.join([split_names[-1], leaf])
        if name_before_split is None:
            raise RuntimeError(f'Could not find mapping for {qualname}')
        if self.tracer_qualname_map is not None:
            return self.tracer_qualname_map[name_before_split]
        else:
            return name_before_split


class _AttrKind(Enum):
    PARAMETER = 'parameter'
    BUFFER = 'buffer'
    CONSTANT = 'constant'


def _assign_attr(from_obj: 'Union[torch.Tensor, torch.ScriptObject]', to_module: 'torch.nn.Module', target: 'str', attr_kind: '_AttrKind', persistent: 'bool'=True):
    *prefix, field = target.split('.')
    for item in prefix:
        t = getattr(to_module, item, None)
        if t is None:
            t = torch.nn.Module()
            setattr(to_module, item, t)
        to_module = t
    if attr_kind == _AttrKind.PARAMETER:
        assert isinstance(from_obj, torch.nn.Parameter)
        to_module.register_parameter(field, from_obj)
    elif attr_kind == _AttrKind.BUFFER:
        assert isinstance(from_obj, torch.Tensor)
        to_module.register_buffer(field, from_obj, persistent=persistent)
    elif attr_kind == _AttrKind.CONSTANT:
        assert isinstance(from_obj, (torch.Tensor, torch.ScriptObject))
        setattr(to_module, field, from_obj)


class _NodeReference:

    def __init__(self, name):
        self.name = name
    name: 'str'


class _LinearNodeList:

    def __init__(self, node_list):
        self.serialize_node_list = []
        for node in node_list:
            node_args = fx.node.map_arg(node.args, lambda n: _NodeReference(n.name))
            node_kwargs = fx.node.map_arg(node.kwargs, lambda n: _NodeReference(n.name))
            serialize_node = fx.Node(graph=None, name=node.name, op=node.op, target=node.target, args=node_args, kwargs=node_kwargs, return_type=node.type)
            serialize_node.meta = copy.copy(node.meta)
            self.serialize_node_list.append(serialize_node)

    def to_graph(self):
        graph = fx.Graph()
        ref_str_to_node: 'Dict[str, fx.Node]' = {}

        def ref_to_node(arg):
            if isinstance(arg, _NodeReference):
                return ref_str_to_node[arg.name]
            else:
                return arg
        for node in self.serialize_node_list:
            node_args = map_aggregate(node.args, ref_to_node)
            node_kwargs = map_aggregate(node.kwargs, ref_to_node)
            deser_node = graph.create_node(op=node.op, target=node.target, args=node_args, kwargs=node_kwargs, name=node.name, type_expr=node.type)
            ref_str_to_node[node.name] = deser_node
        return graph


def _direct_serialization_deserialize(body, nodes):
    """
    Custom `__reduce__` method for serialization.
    DO AS I SAY -- NOT AS I DO. This violates the principle that
    GraphModules serialize via code export & re-tracing. We allow
    for this here because **PIPE STAGES SHOULD NOT BE PERSISTED
    TO DISK -- THIS IS ONLY FOR TRANSMISSION VIA RPC**. Persisting
    these instances to disk will expose internal implementation
    details of `fx.Graph` and related data structures and is
    NOT advised.
    """


    class DummyModule(torch.nn.Module):

        def __init__(self, body):
            super().__init__()
            self.__dict__.update(body)
    dummy = DummyModule(body)
    return fx.GraphModule(dummy, nodes.to_graph())


def _direct_serialization_reduce(self):
    serialization_dict = dict(self.__dict__)
    serialization_dict.pop('_graph')
    return _direct_serialization_deserialize, (serialization_dict, _LinearNodeList(self.graph.nodes))


def _find_loss_from_output_and_spec(output_val, spec_val):
    if spec_val is False:
        return None
    if spec_val is True:
        if not isinstance(output_val, fx.Node):
            raise RuntimeError(f'Loss spec must specify a dynamic value but got {output_val}')
        return output_val
    if isinstance(spec_val, (tuple, list)):
        if not isinstance(output_val, (tuple, list)):
            raise RuntimeError(f'Output value {output_val} must match type of loss specification {spec_val}')
        if len(output_val) != len(spec_val):
            raise RuntimeError(f'Output value {output_val} must match length of loss specification {spec_val}')
        for out, spec in zip(output_val, spec_val):
            loss_val = _find_loss_from_output_and_spec(out, spec)
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f'Did not find loss value in specification {spec_val}')
    if isinstance(spec_val, dict):
        if not isinstance(output_val, dict):
            raise RuntimeError(f'Output value {output_val} must match type of loss specification {spec_val}')
        if set(output_val.keys()) != set(spec_val.keys()):
            raise RuntimeError(f'Output value {output_val} must match keys of loss specification {spec_val}')
        for k in spec_val:
            loss_val = _find_loss_from_output_and_spec(output_val[k], spec_val[k])
            if loss_val is not None:
                return loss_val
        raise RuntimeError(f'Did not find loss value in specification {spec_val}')
    raise RuntimeError(f'Unsupported type {type(spec_val)} in loss specification')


def _find_loss_output(mod: 'torch.nn.Module', g: 'fx.Graph', output_loss_value_spec):
    output_nodes = [n for n in g.nodes if n.op == 'output']
    assert len(output_nodes) == 1
    output_node = output_nodes[0]
    output_val = output_node.args[0]
    generated_spec: 'Any' = None
    if isinstance(mod, TrivialLossWrapper):
        assert len(output_node.args) == 1
        loss_node = output_val
        generated_spec = TrivialLossWrapper.loss_spec
    elif output_loss_value_spec is None:
        if isinstance(output_val, dict) and 'loss' in output_val.keys():
            loss_node = output_val['loss']
            generated_spec = {k: (k == 'loss') for k in output_val}
        else:
            loss_node = None
            generated_spec = None
    else:
        loss_node = _find_loss_from_output_and_spec(output_val, output_loss_value_spec)
        generated_spec = output_loss_value_spec
    return loss_node, output_node, generated_spec


def _null_coalesce_accumulate(lhs, rhs):
    """
    Coalesce two values, even if one of them is null, returning the non-null
    value.
    """
    if lhs is None:
        return rhs
    elif rhs is None:
        return lhs
    else:
        return torch.add(lhs, rhs)


def _insert_stage_symbolic_backward(g: 'fx.Graph', loss_node: 'fx.Node', output_node: 'fx.Node'):
    tuples: 'Dict[fx.Node, Tuple]' = {}
    for node in reversed(g.nodes):
        if node.op == 'call_function':
            assert node.target == operator.getitem, 'Found non-getitem call in forward pass. Please report a bug to PiPPy'
            assert len(node.args) == 2, 'Found malformed getitem call. Please report a bug to PiPPy'
            indexed_value, node_idx = tuple(node.args)
            existing_list_size = len(tuples[indexed_value]) if indexed_value in tuples else -1
            new_list_size = max(node_idx + 1, existing_list_size)
            reconstructed_list = [None for _ in range(new_list_size)]
            if indexed_value in tuples:
                for i, val in enumerate(tuples[indexed_value]):
                    reconstructed_list[i] = val
            reconstructed_list[node_idx] = node
            tuples[indexed_value] = tuple(reconstructed_list)
    live_nodes = {loss_node: None}
    val_to_grad: 'Dict[fx.Node, Optional[fx.Node]]' = {loss_node: None}

    def assign_or_accumulate_grad(forward_node, grad_value):
        if forward_node in val_to_grad and forward_node.op != 'placeholder':
            grad_value = g.call_function(_null_coalesce_accumulate, (val_to_grad[forward_node], grad_value))
        val_to_grad[forward_node] = grad_value
    with g.inserting_before(output_node):
        for node in reversed(g.nodes):
            if node not in live_nodes:
                continue

            def add_to_live_nodes(n):
                live_nodes.setdefault(n, None)
            fx.node.map_arg(node.args, add_to_live_nodes)
            fx.node.map_arg(node.kwargs, add_to_live_nodes)
            if node.op == 'call_module':
                output_grads: 'Union[Tuple[Optional[fx.Node], ...], Optional[fx.Node]]'
                if node in tuples:
                    stage_output = tuples[node]
                    output_grads = tuple(val_to_grad.get(n, None) for n in tuples[node])
                    outputs_with_grads_idxs = [i for i, n in enumerate(tuples[node]) if n in live_nodes]
                else:
                    stage_output = node,
                    output_grads = val_to_grad[node]
                    outputs_with_grads_idxs = [0]
                output_grads = (output_grads,) if not isinstance(output_grads, tuple) else output_grads
                grad_call = g.call_function(stage_backward, kwargs={'stage_output': stage_output, 'output_grads': output_grads, 'input_values': list(node.all_input_nodes), 'outputs_with_grads_idxs': outputs_with_grads_idxs})
                kwargs_copy = dict(grad_call.kwargs)
                grad_call.kwargs = kwargs_copy
                grad_call_proxy = fx.Proxy(grad_call)
                grads = grad_call_proxy.node
                input_nodes = list(node.all_input_nodes)
                grads_proxy = fx.Proxy(grads)
                for i, input_node in enumerate(input_nodes):
                    assign_or_accumulate_grad(input_node, grads_proxy[i].node)
    return g


def _add_submodule(mod: 'torch.nn.Module', target: 'str', module_to_add: 'torch.nn.Module'):
    *prefix, field = target.split('.')
    for item in prefix:
        submod = getattr(mod, item, None)
        if submod is None:
            submod = torch.nn.Module()
            setattr(mod, item, submod)
        if not isinstance(submod, torch.nn.Module):
            return False
        mod = submod
    mod.add_module(field, module_to_add)


def _compute_accessor(parent_fqn: 'str', child_fqn: 'str') ->str:
    if parent_fqn == '':
        return child_fqn
    parent_split = parent_fqn.split('.')
    child_split = child_fqn.split('.')
    assert child_split[:len(parent_split)] == parent_split, f"Child module '{child_fqn}' is not a descendant of parent module '{parent_fqn}'"
    return '.'.join(child_split[len(parent_split):])


def _add_spec(gm: 'torch.nn.Module', spec) ->str:
    i = 0
    while hasattr(gm, f'_spec_{i}'):
        i += 1
    name = f'_spec_{i}'
    setattr(gm, name, spec)
    return name


def _generate_flatten(gm: 'torch.nn.Module', node, spec) ->torch.fx.Node:
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(fx_pytree.tree_flatten_spec, (node, spec_node))


def _generate_unflatten(gm: 'torch.nn.Module', nodes, spec) ->torch.fx.Node:
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(pytree.tree_unflatten, (nodes, spec_node))


def _is_prefix(candidate, target):
    """Check whether `candidate` is a prefix of `target`."""
    return len(candidate) < len(target) and target[:len(candidate)] == candidate


def _verify_graph_equivalence(x: 'torch.nn.Module', y: 'torch.nn.Module'):

    def graph_dump(graph: 'torch.fx.Graph') ->str:
        ret = []
        nodes_idx: 'Dict[int, int]' = {}

        def arg_dump(arg) ->str:
            if isinstance(arg, torch.fx.Node):
                return '%' + str(nodes_idx[id(arg)])
            return str(arg)
        for i, node in enumerate(graph.nodes):
            args_dump = [str(arg) for arg in pytree.tree_map(arg_dump, node.args)]
            args_dump += [f'{key}={value}' for key, value in pytree.tree_map(arg_dump, node.kwargs).items()]
            target = node.target if node.op == 'call_function' else ''
            ret.append(f"{i}: {node.op}[{target}]({', '.join(args_dump)})")
            nodes_idx[id(node)] = i
        return '\n'.join(ret)
    assert graph_dump(x.graph) == graph_dump(y.graph)


class _ModuleFrame:

    def __init__(self, flat_graph, nodes, seen_nodes, seen_modules, parent, module_stack, module_id, module_call_graph: 'Optional[Dict[str, ModuleCallSignature]]'=None, module: 'Optional[torch.nn.Module]'=None):
        self.flat_graph = flat_graph
        self.nodes = nodes
        self.seen_nodes = seen_nodes
        self.seen_modules = seen_modules
        self.parent = parent
        self.module_stack = module_stack
        self.module_id = module_id
        self.module_call_graph = module_call_graph
        self.verbose = False
        self.fqn = self.module_stack[-1]
        if module is not None:
            self.module = module
        else:
            self.module = InterpreterModule(torch.fx.Graph())
        if self.module_id in self.seen_modules:
            self.cached_graph_module = self.seen_modules[self.module_id]
        else:
            self.cached_graph_module = None
            self.seen_modules[self.module_id] = self.module
        self.graph = self.module.graph
        self.node_map: 'Dict[torch.fx.Node, torch.fx.Node]' = {}
        self.node_to_placeholder = {}
        self.parent_call_module: 'Optional[torch.fx.Node]' = None
        if parent is not None:
            accessor = _compute_accessor(parent.fqn, self.fqn)
            _add_submodule(parent.module, accessor, self.module if self.cached_graph_module is None else self.cached_graph_module)
            self.parent_call_module = parent.graph.call_module(accessor)
        signature = self.get_signature()
        if signature is not None and self.parent is not None:
            assert signature.in_spec.num_children == 2
            args_spec = signature.in_spec.children_specs[0]
            kwargs_spec = signature.in_spec.children_specs[1]
            assert args_spec.context is None
            assert kwargs_spec.context is not None
            with self.graph.inserting_after(None):
                arg_nodes = []
                for idx in range(args_spec.num_children):
                    arg_nodes.append(self.graph.placeholder(f'_positional_arg_{idx}'))
                kwarg_nodes = {}
                for name in kwargs_spec.context:
                    kwarg_nodes[name] = self.graph.placeholder(name)
                flat_args = _generate_flatten(self.module, (tuple(arg_nodes), kwarg_nodes), signature.in_spec)
                for idx, arg in enumerate(signature.inputs):
                    flat_arg_node = self.graph.create_node(op='call_function', target=operator.getitem, args=(flat_args, idx), name=arg.name if not isinstance(arg, ConstantArgument) else f'_constant_{idx}')
                    if isinstance(arg, ConstantArgument):
                        continue
                    flat_arg_node.meta = copy.copy(self.seen_nodes[arg.name].meta)
                    self.node_to_placeholder[self.seen_nodes[arg.name]] = flat_arg_node
            with self.parent.graph.inserting_before(self.parent_call_module):
                input_nodes: 'List[Optional[torch.fx.Node]]' = []
                for input in signature.inputs:
                    if isinstance(input, ConstantArgument) and input.value is None:
                        input_nodes.append(None)
                    else:
                        assert isinstance(input, (TensorArgument, SymIntArgument))
                        input_nodes.append(self.parent.remap_input(self.seen_nodes[input.name]))
                inputs_node = _generate_unflatten(self.parent.module, input_nodes, signature.in_spec)
                args_node = self.parent.graph.call_function(operator.getitem, (inputs_node, 0))
                kwargs_node = self.parent.graph.call_function(operator.getitem, (inputs_node, 1))
                arg_nodes = [self.parent.graph.call_function(operator.getitem, (args_node, i)) for i in range(args_spec.num_children)]
                kwarg_nodes = {k: self.parent.graph.call_function(operator.getitem, (kwargs_node, k)) for k in kwargs_spec.context}
            assert self.parent_call_module is not None
            self.parent_call_module.args = tuple(arg_nodes)
            self.parent_call_module.kwargs = kwarg_nodes

    def add_placeholder(self, x):
        assert x.graph is self.flat_graph
        with self.graph.inserting_before(None):
            placeholder_node = self.graph.placeholder(x.name, type_expr=x.type)
        placeholder_node.meta = copy.copy(x.meta)
        self.node_to_placeholder[x] = placeholder_node

    def remap_input(self, x):
        assert x.graph is self.flat_graph
        if x in self.node_map:
            return self.node_map[x]
        if x not in self.node_to_placeholder:
            self.add_placeholder(x)
            if self.parent_call_module is not None:
                self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
        return self.node_to_placeholder[x]

    def get_signature(self):
        if self.module_call_graph is not None:
            return self.module_call_graph.get(self.fqn)
        return None

    def finalize_outputs(self):
        orig_outputs = []
        signature = self.get_signature()
        if signature is not None and self.parent is not None:
            for output in signature.outputs:
                if isinstance(output, (TensorArgument, SymIntArgument)):
                    orig_outputs.append(self.seen_nodes[output.name])
                else:
                    raise RuntimeError(f'Unsupported data type for output node: {output}')
            tree_out_node = _generate_unflatten(self.module, tuple(self.node_map[self.seen_nodes[output.name]] for output in orig_outputs), signature.out_spec)
            parent_out: 'Optional[torch.fx.Node]' = _generate_flatten(self.parent.module, self.parent_call_module, signature.out_spec)
            graph_outputs: 'Union[torch.fx.Node, List[torch.fx.Node]]' = tree_out_node
        else:
            graph_outputs = []
            for orig_node in self.node_map.keys():
                for user_node in orig_node.users:
                    if user_node.name not in self.seen_nodes:
                        orig_outputs.append(orig_node)
                        graph_outputs.append(self.node_map[orig_node])
                        break
            parent_out = self.parent_call_module
            if len(graph_outputs) == 1:
                graph_outputs = graph_outputs[0]
        assert isinstance(graph_outputs, (list, torch.fx.Node))
        self.graph.output(graph_outputs)
        if parent_out is None:
            return
        parent_out.meta['val'] = graph_outputs.meta.get('val') if isinstance(graph_outputs, torch.fx.Node) else [o.meta.get('val') for o in graph_outputs]
        if len(orig_outputs) == 1 and signature is None:
            self.parent.node_map[orig_outputs[0]] = parent_out
        else:
            for i, orig_output in enumerate(orig_outputs):
                proxy_out = torch.fx.Proxy(parent_out)[i].node
                proxy_out.meta['val'] = orig_output.meta.get('val')
                self.parent.node_map[orig_output] = proxy_out
        if self.cached_graph_module is not None:
            _verify_graph_equivalence(self.cached_graph_module, self.module)

    def copy_node(self, node):
        self.print('copying', node.format_node())
        self.node_map[node] = self.graph.node_copy(node, self.remap_input)
        self.seen_nodes[node.name] = node

    def run_outer(self):
        i = 0
        for node in self.flat_graph.nodes:
            self.print(i, node.meta.get('nn_module_stack'), node.format_node())
            i += 1
        node_idx: 'int' = 0
        node = self.nodes[node_idx]
        while node.op == 'placeholder':
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]
        self.run_from(node_idx)
        for node in self.flat_graph.nodes:
            if node.op == 'output':
                self.copy_node(node)

    def print(self, *args, **kwargs):
        if self.verbose:
            None

    def run_from(self, node_idx):
        module_idx = 0
        while node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            assert node.op != 'placeholder'
            self.print()
            self.print('STEP', node_idx, node.format_node())
            self.print(self.module_stack)
            if node.op == 'output':
                if len(self.module_stack) == 1:
                    return node_idx
                self.finalize_outputs()
                return node_idx
            node_module_stack = [path for path, ty in node.meta['nn_module_stack'].values()] if 'nn_module_stack' in node.meta else self.module_stack
            if node_module_stack[:len(self.module_stack)] != self.module_stack:
                self.finalize_outputs()
                self.print('outlining', self.fqn)
                self.print(self.graph)
                return node_idx
            assert node_module_stack is not None
            if _is_prefix(self.module_stack, node_module_stack):
                next_module = node_module_stack[len(self.module_stack)]
                self.print('Creating new stack frame for', next_module)
                node_idx = _ModuleFrame(self.flat_graph, self.nodes, self.seen_nodes, self.seen_modules, self, self.module_stack + [next_module], list(node.meta['nn_module_stack'].keys())[len(self.module_stack)], self.module_call_graph).run_from(node_idx)
                module_idx += 1
                continue
            assert node_module_stack == self.module_stack
            self.copy_node(node)
            node_idx += 1


def _outline_submodules(orig_graph: 'torch.fx.Graph'):
    new_module = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
    seen_nodes: 'Dict[str, torch.fx.Node]' = {}
    seen_modules: 'Dict[int, torch.nn.Module]' = {}
    _ModuleFrame(orig_graph, tuple(orig_graph.nodes), seen_nodes, seen_modules, None, [''], '', module=new_module).run_outer()
    new_module.graph.lint()
    new_module.recompile()
    return new_module


def _recursive_getattr(obj, attr_path):
    for attr in attr_path:
        obj = getattr(obj, attr)
    return obj


def _sink_params(module: 'torch.nn.Module', inputs_to_state: 'Dict[str, str]', scope: 'List[str]'):
    """Sink params, buffers, and constants from graph inputs into get_attr nodes.

    Exported modules are purely functional, so they pass their parameters and
    buffers in as inputs to the graph.

    To replicate eager's semantics, we need to get them from the module state
    via get_attr instead.

    module: GraphModule, potentially containining nested submodules.
    inputs_to_state: mapping graph input names to the corresponding key in the state_dict.
    scope: tracks where we are in the module hierarchy, so that we can emit the
        right `getattr(self, "foo.bar")` calls, etc.
    """
    for name, submodule in module._modules.items():
        _sink_params(cast(torch.nn.Module, submodule), inputs_to_state, scope + [name])
    if not hasattr(module, 'graph'):
        return
    graph = module.graph
    inputs = list(filter(lambda n: n.op == 'placeholder', graph.nodes))
    the_last_input = inputs[-1]
    call_module_nodes = filter(lambda n: n.op == 'call_module', graph.nodes)
    for node in call_module_nodes:
        node.args = tuple(filter(lambda n: n.name not in inputs_to_state, node.args))
    for node in inputs:
        if node.name not in inputs_to_state:
            continue
        if len(node.users) > 0:
            state_name = inputs_to_state[node.name].split('.')
            if state_name[:len(scope)] != scope:
                continue
            attr_path = state_name[len(scope):]
            state_attr = _recursive_getattr(module, attr_path)
            assert isinstance(state_attr, (torch.Tensor, torch.ScriptObject))
            with graph.inserting_after(the_last_input):
                new_node = graph.create_node('get_attr', '.'.join(attr_path))
            node.replace_all_uses_with(new_node, propagate_meta=True)
        graph.erase_node(node)
    if isinstance(module, InterpreterModule):
        module.finalize()


aten_pipe_split_alias = torch.ops.pippy._pipe_split.default


logger = logging.getLogger(__name__)


DEFAULT_CHUNK_DIM = 0


class TensorChunkSpec:

    def __init__(self, split_dim):
        self.split_dim = split_dim
    split_dim: 'int'

    def __repr__(self):
        return f'{self.__class__.__module__}.{self.__class__.__name__}({self.split_dim})'

    def __str__(self):
        return f'TensorChunkSpec({self.split_dim})'


class _Replicate:
    pass


_debug_mask_minibatches = False


def _shard_dict_of_args(args_dict, args_chunk_spec, num_chunks):
    """
    Given a dictionary of args, and a dictionary of chunking specs, shard the
    args according to the chunking specs.

    Args:
        args_dict: Dictionary of args
        args_chunk_spec: Dictionary of chunking specs
        num_chunks: Number of chunks to shard the args into

    Returns:
        args_split: List of sharded args
    """
    args_sharded_replicated = {}
    arg_specs = []
    real_num_chunks = num_chunks
    first_tensor = True
    assert len(args_dict) == len(args_chunk_spec), f'args_dict.keys() = {list(args_dict.keys())} args_chunk_spec.keys() = {list(args_chunk_spec.keys())}'
    for arg_key, arg in args_dict.items():
        flat, spec = tree_flatten(arg)
        arg_specs.append(spec)
        chunk_spec = args_chunk_spec[arg_key]
        assert chunk_spec is not None
        chunk_spec_flat, _ = tree_flatten(chunk_spec)
        if len(flat) != len(chunk_spec_flat):
            raise ValueError(f'Argument value {arg} did not have the same number of values as as chunk spec {chunk_spec}')
        sharded_arg_flat = []
        for v, chunk_v in zip(flat, chunk_spec_flat):
            if chunk_v is _Replicate or not isinstance(v, torch.Tensor):
                sharded_arg_flat.append([v] * real_num_chunks)
            elif isinstance(chunk_v, TensorChunkSpec):
                assert isinstance(v, torch.Tensor), f'{v} is not a tensor'
                v_split_dim_size = v.size(chunk_v.split_dim)
                if v_split_dim_size < real_num_chunks:
                    if first_tensor:
                        logger.warning(f'Tensor size on chunking dimension is {v_split_dim_size}, downsizing the number of chunks from {num_chunks} to {v_split_dim_size}.')
                        real_num_chunks = v_split_dim_size
                    else:
                        raise RuntimeError(f'Arg {arg_key} on chunking dimension has a size of {v_split_dim_size}, smaller than the number of chunks {num_chunks}. PiPPy cannot reduce the number of chunks because other arguments have bigger chunk-dimension sizes. Please adjust your num_chunks setting.')
                chunk_tensors = torch.tensor_split(v, real_num_chunks, chunk_v.split_dim)
                if _debug_mask_minibatches:
                    expanded_chunks = []
                    split_dim_idx = 0
                    for chunk_tensor in chunk_tensors:
                        new_val = torch.zeros_like(v)
                        upper_idx = split_dim_idx + chunk_tensor.size(chunk_v.split_dim)
                        slice_indices = [slice(None, None, None)] * new_val.ndim
                        slice_indices[chunk_v.split_dim] = slice(split_dim_idx, upper_idx)
                        new_val[slice_indices] = chunk_tensor
                        expanded_chunks.append(new_val)
                        split_dim_idx += chunk_tensor.size(chunk_v.split_dim)
                    sharded_arg_flat.append(expanded_chunks)
                else:
                    sharded_arg_flat.append(chunk_tensors)
                first_tensor = False
            else:
                raise TypeError(f'Unrecognized chunk spec: {chunk_v}')
        args_sharded_replicated[arg_key] = sharded_arg_flat
    chunks_flat = []
    for chunk_idx in range(real_num_chunks):
        chunk_args = {}
        for key, arg in args_sharded_replicated.items():
            arg_single_chunk = []
            for v_flat in arg:
                arg_single_chunk.append(v_flat[chunk_idx])
            chunk_args[key] = arg_single_chunk
        chunks_flat.append(chunk_args)
    args_split = []
    for chunk in chunks_flat:
        per_chunk_args = {}
        assert len(arg_specs) == len(chunk)
        for (key, arg), arg_spec in zip(chunk.items(), arg_specs):
            per_chunk_args[key] = tree_unflatten(arg, arg_spec)
        args_split.append(per_chunk_args)
    return args_split


def split_args_kwargs_into_chunks(args: 'Tuple[Any, ...]', kwargs: 'Optional[Dict[str, Any]]', chunks: 'int', args_chunk_spec: 'Optional[Tuple[TensorChunkSpec, ...]]'=None, kwargs_chunk_spec: 'Optional[Dict[str, TensorChunkSpec]]'=None) ->Tuple[List[Tuple], List[Dict]]:
    """
    Given a sequence of args and kwargs, split them into a number of chunks
    according to  their respective chunking specs.

    Args:
        args: Tuple of args
        kwargs: Dict of kwargs
        chunks: Number of chunks to split the args and kwargs into
        args_chunk_spec: chunking specs for args, in same shape as args
        kwargs_chunk_spec: chunking specs for kwargs, in same shape as kwargs

    Returns:
        args_split: List of sharded args
        kwargs_split: List of sharded kwargs
    """
    if kwargs is None:
        kwargs = {}
    if args_chunk_spec is None:
        args_chunk_spec = (TensorChunkSpec(DEFAULT_CHUNK_DIM),) * len(args)
    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = dict.fromkeys(kwargs, TensorChunkSpec(DEFAULT_CHUNK_DIM))
    args_split_dict = _shard_dict_of_args(dict(enumerate(args)), dict(enumerate(args_chunk_spec)), chunks)
    real_num_chunks = len(args_split_dict)
    kwargs_split = _shard_dict_of_args(kwargs, kwargs_chunk_spec, real_num_chunks)
    if len(kwargs_split) < real_num_chunks:
        real_num_chunks = len(kwargs_split)
        args_split_dict = _shard_dict_of_args(dict(enumerate(args)), dict(enumerate(args_chunk_spec)), real_num_chunks)
    if len(args_split_dict) != len(kwargs_split):
        raise RuntimeError(f'args and kwargs are split into different number of chunks: {len(args_split_dict)}, {len(kwargs_split)}')
    args_split = []
    for chunk_args in args_split_dict:
        args_split.append(tuple(chunk_args[i] for i in range(len(chunk_args))))
    return args_split, kwargs_split


class DDPAROnce(torch.nn.Module):

    def __init__(self, module: 'torch.nn.Module', group: 'dist.ProcessGroup', dtype=None):
        super().__init__()
        self.module = module
        self.group = group
        self.dtype = dtype
        global_rank = dist.get_global_rank(self.group, self.group.rank())
        for param in self.module.parameters():
            dist.broadcast(param.data, src=global_rank, group=self.group)
        self.buffer = torch.zeros(sum([p.numel() for p in module.parameters()])) if self.dtype is None else torch.zeros(sum([p.numel() for p in module.parameters()])).to(self.dtype)

    def zero_grad(self):
        self.buffer.zero_()
        offset = 0
        for p in self.module.parameters():
            p.grad = self.buffer[offset:offset + p.numel()].view(p.shape)
            offset = offset + p.numel()

    def all_reduce_async(self, norm_factor: 'int'):
        self.buffer.div_(norm_factor * self.group.size())
        work = dist.all_reduce(self.buffer, async_op=True, group=self.group)
        return work

    def all_reduce(self, norm_factor: 'int'):
        self.buffer.div_(norm_factor * self.group.size())
        work = dist.all_reduce(self.buffer, async_op=True, group=self.group)
        work.wait()

    def forward(self, *args, **kwargs):
        return self.module.forward(*args, **kwargs)


class MultiMLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)
        self.mse_loss = torch.nn.MSELoss(reduction='sum')

    def forward(self, x, y):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        loss = self.mse_loss(x, y)
        return x, loss


n_layers = 8


class TransformerLike(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.layers = torch.nn.Sequential(*[MLPModule(d_hid) for _ in range(n_layers)])

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.layers(x)


class MLP(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int', out_dim: 'int'):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.w1(x)
        x = self.w2(x)
        x = self.relu(x)
        return x


class MultiInputArgMLP(nn.Module):

    def __init__(self, dim1: 'int', dim2: 'int', out_dim: 'int'):
        super().__init__()
        self.w1 = nn.Linear(dim1, out_dim, bias=False)
        self.w2 = nn.Linear(dim2, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.w1(x)
        y = self.w2(y)
        z = x + y
        z = self.relu(z)
        return z


class MultiOutputArgMLP(nn.Module):

    def __init__(self, dim: 'int', out_dim: 'int'):
        super().__init__()
        self.w1 = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x):
        x = self.w1(x)
        y = torch.cat([x, x], dim=0)
        return x, y


class InvalidOutputModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {}


class ModelWithSleep(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int', out_dim: 'int', rank: 'int'):
        super().__init__()
        self.in_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.middle = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU())
        self.out_layer = nn.Linear(hidden_dim, out_dim, bias=False)
        self.relu = nn.ReLU()
        self.rank = rank

    def forward(self, x):
        x = self.in_layer(x)
        x = self.middle(x)
        if self.rank == 0 or self.rank == 1:
            time.sleep(random.uniform(0, 0.5))
        x = self.out_layer(x)
        x = self.relu(x)
        return x


class Block(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.lin0 = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(256, 256)

    def forward(self, x: 'torch.Tensor', constant=None) ->torch.Tensor:
        x = self.conv(x)
        x = self.lin0(x)
        pipe_split()
        x.add_(constant)
        x = self.lin1(x)
        return self.relu(x)


class M(torch.nn.Module):

    def __init__(self) ->None:
        super().__init__()
        self.block0 = Block()
        self.block1 = Block()

    def forward(self, x: 'torch.Tensor', constant=None) ->torch.Tensor:
        x = self.block0(x, constant=constant)
        pipe_split()
        x = self.block1(x, constant=constant)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ExpertLayer,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (InvalidOutputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IterationBlock,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (IterativeNetwork,
     lambda: ([], {'d_hid': 4, 'num_iters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLPModule,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ModelWithSleep,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'out_dim': 4, 'rank': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiInputArgMLP,
     lambda: ([], {'dim1': 4, 'dim2': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MultiOutputArgMLP,
     lambda: ([], {'dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MyNetworkBlock,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PipeSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TrivialLossWrapper,
     lambda: ([], {'module': torch.nn.ReLU(), 'loss_fn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

