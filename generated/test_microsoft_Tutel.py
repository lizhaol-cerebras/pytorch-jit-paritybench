
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


from typing import Tuple


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CppExtension


import itertools


import math


import torch


import re


import warnings


import time


import torch.optim as optim


import torch.nn.functional as F


from torch import nn


import torch.distributed as dist


from torch.cuda.amp import autocast


import torch.nn as nn


from torchvision import datasets


from torchvision import transforms


from torch.optim.lr_scheduler import StepLR


import logging


from typing import TYPE_CHECKING


from typing import Any


from typing import Optional


from typing import Union


from typing import cast


from torch import Tensor


from torch.distributions.normal import Normal


import copy


import collections


from torch.nn import ModuleList


class ExampleModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self._moe_layer = tutel_moe.moe_layer(gate_type={'type': 'top', 'k': top_value, 'fp32_gate': args.fp32_gate}, experts={'type': 'ffn', 'count_per_node': num_local_experts, 'hidden_size_per_expert': hidden_size, 'activation_fn': lambda x: F.relu(x)}, model_dim=model_dim, scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True), seeds=(1, dist_rank + 1, 1), a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, use_2dh=args.use_2dh)
        local_count = sum([torch.numel(param) for name, param in self._moe_layer.get_parameter_iterator(param_type='local_experts')])
        shared_count = sum([torch.numel(param) for name, param in self._moe_layer.get_parameter_iterator(param_type='gate')])
        dist_print('[Statistics] param count for MoE local_experts = %s, param count for MoE gate = %s.\n' % (local_count, shared_count))
        self.r_index = -1

    def forward(self, input):
        r, o = self._moe_layer.valid_rs[self.r_index // 8 % len(self._moe_layer.valid_rs)], self.r_index % 8 + 1
        self.r_index += 1
        result = self._moe_layer(input, capacity_factor=args.cap_factor, adaptive_r=r, a2a_ffn_overlap_degree=o)
        result = F.log_softmax(torch.sum(result, dim=2), dim=1)
        return result


class CustomExpertDemo(torch.nn.Module):

    def _create_sharded_param(self, *full_shape, **kwargs):
        full_shape = torch.Size(full_shape)
        sharded_shape = (full_shape.numel() + self.sharded_count - 1) // self.sharded_count
        return torch.nn.Parameter(torch.empty(sharded_shape, **kwargs)), full_shape

    def _get_gathered_param(self, param, full_shape):
        sharded_group = net.create_groups_from_world(group_count=-self.sharded_count).model_group
        return net.zero_gather(param, group=sharded_group).view(-1).narrow(0, 0, full_shape.numel()).view(full_shape)

    def __init__(self, model_dim, local_experts, sharded_count, my_config):
        super().__init__()
        self.sharded_count = sharded_count
        self.W, self.W_full_shape = self._create_sharded_param(local_experts, model_dim, model_dim)
        self.my_activation = torch.nn.functional.relu if my_config == 'relu' else None
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.W.normal_(0, 0.001)

    def forward(self, x, ctx):
        W_full = self._get_gathered_param(self.W, self.W_full_shape)
        y = torch.matmul(x, W_full)
        if self.my_activation is not None:
            y = self.my_activation(y)
        return y


class CustomGate(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.register_parameter(name='wg', param=torch.nn.Parameter(torch.randn([model_dim, num_global_experts]) * 0.001))

    def forward(self, x):
        return torch.matmul(x, self.wg)


class CustomExpert(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(dist.global_rank + 1)
        self.register_parameter(name='batched_fc1_w', param=torch.nn.Parameter(torch.randn([num_local_experts, model_dim, hidden_size]) * 0.001))
        self.register_parameter(name='batched_fc2_w', param=torch.nn.Parameter(torch.randn([num_local_experts, hidden_size, model_dim]) * 0.001))
        self.register_parameter(name='batched_fc1_bias', param=torch.nn.Parameter(torch.zeros([num_local_experts, 1, hidden_size])))
        self.register_parameter(name='batched_fc2_bias', param=torch.nn.Parameter(torch.zeros([num_local_experts, 1, model_dim])))
        for x in self.parameters():
            setattr(x, 'skip_allreduce', True)

    def forward(self, x):
        y = torch.add(torch.matmul(x, self.batched_fc1_w), self.batched_fc1_bias)
        y = F.relu(y)
        y = torch.add(torch.matmul(y, self.batched_fc2_w), self.batched_fc2_bias)
        return y


class CustomMoE(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.gate = CustomGate()
        self.expert = CustomExpert()

    def forward(self, x, k=2):
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)
        crit, l_aux = moe.top_k_routing(scores, top_k=k)
        y = moe.fast_encode(x, crit)
        y = net.all_to_all(y, 1, 0)
        y = self.expert(y)
        y = net.all_to_all(y, 0, 1)
        output = moe.fast_decode(y, crit)
        return output, l_aux


class Net(nn.Module):
    DATASET_TARGET = datasets.MNIST

    def __init__(self, use_moe):
        super(Net, self).__init__()
        self.use_moe = use_moe
        if self.use_moe:
            self.moe_ffn = moe.moe_layer(gate_type={'type': 'top', 'k': 1, 'capacity_factor': 0, 'gate_noise': 1.0}, experts={'type': 'ffn', 'count_per_node': 1, 'hidden_size_per_expert': 128, 'output_dim': 10, 'activation_fn': lambda x: self.dropout2(F.relu(x))}, model_dim=9216, seeds=(1, penv.global_rank + 1), scan_expert_func=lambda name, param: setattr(param, 'skip_allreduce', True))
        else:
            torch.manual_seed(1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)
        torch.manual_seed(1)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x, top_k=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        if self.use_moe:
            x = self.moe_ffn(x, top_k=top_k)
        else:
            x = self.fc1(x)
            x = self.dropout2(F.relu(x))
            x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class FusedExpertsNetwork(torch.nn.Module):

    def __init__(self, model_dim, hidden_size_per_expert, local_experts, sharded_count, activation_fn=None, activation_fn_with_self=None, output_dim=None, has_fc1_bias=True, has_fc2_bias=True):
        super().__init__()
        self.skip_expert = int(torch.os.environ.get('SKIP_EXPERT', '0')) != 0
        assert hidden_size_per_expert % sharded_count == 0, f"Can't evenly divide hidden_size_per_expert ({hidden_size_per_expert}) to {sharded_count} slices."
        self.model_dim = model_dim
        self.hidden_size_per_expert = hidden_size_per_expert
        self.local_experts = local_experts
        self.sharded_count = sharded_count
        self.hidden_size = hidden_size_per_expert // sharded_count
        self.output_dim = output_dim or model_dim
        if activation_fn_with_self is not None:
            assert activation_fn is None, 'Option `activation_fn_with_self` has been specified, please keep exactly one of them.'
            activation_fn = lambda x: activation_fn_with_self(x, self)
        if activation_fn is None:
            activation_fn = lambda x: F.relu(x)
        self.activation_fn = activation_fn
        self.batched_fc1_w = torch.nn.Parameter(torch.empty(local_experts, self.hidden_size, model_dim))
        self.batched_fc2_w = torch.nn.Parameter(torch.empty(local_experts, self.hidden_size, self.output_dim))
        if has_fc1_bias:
            self.batched_fc1_bias = torch.nn.Parameter(torch.empty(local_experts, self.hidden_size))
        else:
            self.register_parameter('batched_fc1_bias', None)
        if has_fc2_bias:
            self.batched_fc2_bias = torch.nn.Parameter(torch.empty(local_experts, (self.output_dim + sharded_count - 1) // sharded_count))
        else:
            self.register_parameter('batched_fc2_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for i in range(self.batched_fc1_w.size(0)):
                fc1 = torch.nn.Linear(self.model_dim, self.hidden_size, bias=self.batched_fc1_bias is not None)
                fc2 = torch.nn.Linear(self.hidden_size, self.output_dim, bias=self.batched_fc2_bias is not None)
                self.batched_fc1_w[i] = fc1.weight
                if self.batched_fc1_bias is not None:
                    self.batched_fc1_bias[i] = fc1.bias
                self.batched_fc2_w[i] = fc2.weight.t()
                if self.batched_fc2_bias is not None:
                    self.batched_fc2_bias[i] = fc2.bias[:self.batched_fc2_bias.size(-1)]

    def extra_repr(self):
        return 'model_dim=%d, hidden_size=%d, output_dim=%d, local_experts=%d. has_fc1_bias=%s, has_fc2_bias=%s.' % (self.batched_fc1_w.size(2), self.batched_fc1_w.size(1), self.batched_fc2_w.size(2), self.batched_fc1_w.size(0), self.batched_fc1_bias is not None, self.batched_fc2_bias is not None)

    def forward(self, x, ctx):
        if self.skip_expert:
            return x
        batched_fc1_w = self.batched_fc1_w
        batched_fc2_w = self.batched_fc2_w
        if self.batched_fc1_bias is not None:
            batched_fc1_bias = self.batched_fc1_bias.unsqueeze(1)
        if self.batched_fc2_bias is not None:
            batched_fc2_bias = self.batched_fc2_bias.unsqueeze(1)
        if ctx.megablocks_size > 0:
            sparse_size = ctx.megablocks_size
            sparse_groups = torch.div(ctx.dispatch_count + (sparse_size - 1), sparse_size, rounding_mode='floor')
            sparse_groups = torch.minimum(sparse_groups, torch.tensor(x.size(1) // sparse_size, dtype=torch.int32, device=x.device))
            y = torch.ops.tutel_ops.sparse_bmm_infer(x, batched_fc1_w, sparse_groups, True, sparse_size)
            if self.batched_fc1_bias is not None:
                y = torch.add(y, batched_fc1_bias)
            y = self.activation_fn(y)
            y = torch.ops.tutel_ops.sparse_bmm_infer(y, batched_fc2_w, sparse_groups, False, sparse_size)
            if self.batched_fc2_bias is not None:
                y = torch.add(y, batched_fc2_bias)
            return y
        if ctx.adaptive_degree == 0:
            batched_fc1_w = net.zero_gather(batched_fc1_w, group=ctx.group).view(ctx.num_global_experts, -1, batched_fc1_w.size(2))
            batched_fc2_w = net.zero_gather(batched_fc2_w, group=ctx.group).view(ctx.num_global_experts, -1, batched_fc2_w.size(2))
            if self.batched_fc1_bias is not None:
                batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ctx.group).view(ctx.num_global_experts, 1, -1)
            if self.batched_fc2_bias is not None:
                batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=ctx.group).view(ctx.num_global_experts, 1, -1)
        elif ctx.sharded_count > 1:
            group_size = ctx.sharded_count // ctx.adaptive_degree
            if group_size > 1:
                ffn_zero_group = net.create_groups_from_world(group_count=-group_size).model_group
                batched_fc1_w = net.zero_gather(batched_fc1_w, group=ffn_zero_group).view(1, -1, ctx.model_dim)
                batched_fc2_w = net.zero_gather(batched_fc2_w, group=ffn_zero_group).view(1, -1, self.output_dim)
                if self.batched_fc1_bias is not None:
                    batched_fc1_bias = net.zero_gather(batched_fc1_bias, group=ffn_zero_group).view(1, 1, -1)
            if self.batched_fc2_bias is not None:
                batched_fc2_bias = net.zero_gather(batched_fc2_bias, group=net.create_groups_from_world(group_count=ctx.num_global_experts).model_group)
                batched_fc2_bias = batched_fc2_bias.view(1, 1, -1)
                if ctx.adaptive_degree > 1:
                    batched_fc2_bias = torch.mul(batched_fc2_bias, 1.0 / ctx.adaptive_degree)
        if self.batched_fc2_bias is not None and batched_fc2_bias.size(-1) != self.output_dim:
            batched_fc2_bias = batched_fc2_bias[:, :, :self.output_dim]
        y = torch.matmul(x, batched_fc1_w.permute(0, 2, 1))
        if self.batched_fc1_bias is not None:
            y = torch.add(y, batched_fc1_bias)
        y = self.activation_fn(y)
        y = torch.matmul(y, batched_fc2_w)
        if self.batched_fc2_bias is not None:
            y = torch.add(y, batched_fc2_bias)
        return y


class LlamaFFNNetwork(torch.nn.Module):

    def _create_sharded_param(self, *full_shape, **kwargs):
        full_shape = torch.Size(full_shape)
        sharded_shape = (full_shape.numel() + self.sharded_count - 1) // self.sharded_count
        return torch.nn.Parameter(torch.empty(sharded_shape, **kwargs)), full_shape

    def _get_gathered_param(self, param, full_shape):
        sharded_group = net.create_groups_from_world(group_count=-self.sharded_count).model_group
        return net.zero_gather(param, group=sharded_group).view(-1).narrow(0, 0, full_shape.numel()).view(full_shape)

    def __init__(self, model_dim, hidden_size_per_expert, local_experts, sharded_count, activation_fn=torch.nn.functional.silu):
        super().__init__()
        self.sharded_count = sharded_count
        self.W_fc1, self.W_fc1_full_shape = self._create_sharded_param(local_experts, model_dim, hidden_size_per_expert)
        self.W_fc2, self.W_fc2_full_shape = self._create_sharded_param(local_experts, model_dim, hidden_size_per_expert)
        self.W_fc3, self.W_fc3_full_shape = self._create_sharded_param(local_experts, hidden_size_per_expert, model_dim)
        self.activation_fn = activation_fn
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.W_fc1.normal_(0, 0.01)
            self.W_fc2.normal_(0, 0.01)
            self.W_fc3.normal_(0, 0.01)

    def forward(self, x, ctx):
        W_fc1_full = self._get_gathered_param(self.W_fc1, self.W_fc1_full_shape)
        W_fc2_full = self._get_gathered_param(self.W_fc2, self.W_fc2_full_shape)
        W_fc3_full = self._get_gathered_param(self.W_fc3, self.W_fc3_full_shape)
        y1 = torch.matmul(x, W_fc1_full)
        y2 = torch.matmul(x, W_fc2_full)
        y = self.activation_fn(y1) * y2
        y = torch.matmul(y, W_fc3_full)
        return y


class CosineTopKGate(torch.nn.Module):

    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, proj_dim=256, init_t=0.5, **options):
        super(CosineTopKGate, self).__init__()
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1.0 / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)
        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
            cosine_projector = self.cosine_projector.float()
            sim_matrix = self.sim_matrix.float()
        else:
            cosine_projector = self.cosine_projector
            sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1), F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits


class LinearTopKGate(torch.nn.Module):

    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, **options):
        super().__init__()
        try:
            self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False, dtype=torch.float32 if fp32_gate else None)
        except:
            self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False)
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        wg = self.wg.float() if self.fp32_gate else self.wg
        return wg(x)


def a2a_ffn_overlap_forward(input, expert_fn, a2a_ffn_overlap_degree, use_2dh, group):
    split_dim = 1
    assert a2a_ffn_overlap_degree <= C.AllToAllStatus.max_num_split, 'Excepting a2a_ffn_overlap_degree (%d) <= AllToAllStatus.max_num_split (%d).' % (a2a_ffn_overlap_degree, C.AllToAllStatus.max_num_split)
    assert input.shape[split_dim] % a2a_ffn_overlap_degree == 0, 'Excepting input.shape[%d] (%d) be multiple of a2a_ffn_overlap_degree (%d).' % (split_dim, input.shape[split_dim], a2a_ffn_overlap_degree)
    C.AllToAllStatus.init(group, a2a_ffn_overlap_degree, split_dim)
    if use_2dh:
        split_size = input.shape[split_dim] // a2a_ffn_overlap_degree
        input_split = input.split(split_size, dim=split_dim)
        input_scattered_after_a2a = [C.NcclStreamRelease.apply(C.AllToAll2DAsync.apply(C.NcclStreamAcquire.apply(C.CurrentStreamRelease.apply(x, i), i)), i) for i, x in enumerate(input_split)]
    else:
        input_ready = C.CurrentStreamRelease.apply(input, 0)
        input_scattered_after_a2a = C.AllToAllScatterAsync.apply(input_ready)
    expert_output_scattered = [C.CurrentStreamRelease.apply(C.post_expert_permute(expert_fn(C.pre_expert_permute(C.CurrentStreamAcquire.apply(x, i), group=group)), group=group), i) for i, x in enumerate(input_scattered_after_a2a)]
    if use_2dh:
        expert_output_gathered_after_a2a = [C.CurrentStreamAcquire.apply(C.NcclStreamRelease.apply(C.AllToAll2DAsync.apply(C.NcclStreamAcquire.apply(x, i)), i), i) for i, x in enumerate(expert_output_scattered)]
        input = torch.cat(expert_output_gathered_after_a2a, dim=split_dim)
    else:
        expert_output_gathered_after_a2a = C.AllToAllGatherAsync.apply(*expert_output_scattered)
        input = C.CurrentStreamAcquire.apply(expert_output_gathered_after_a2a, 0)
    return input


def cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        elif tensor.device.type == 'xpu':
            dtype = torch.xpu.get_autocast_xpu_dtype()
        elif tensor.device.type == 'hpu':
            dtype = torch.hpu.get_autocast_hpu_dtype()
        else:
            raise RuntimeError("User specified autocast device_type must be 'cuda' or 'cpu'")
        return tensor


has_extension = hasattr(torch.ops.tutel_ops, 'cumsum')


def torch_cumsum_sub_one(mask1):
    locations1 = torch.cumsum(mask1, dim=0) - 1
    return locations1


def fast_cumsum_sub_one(data, dim=0):
    if data.dim() != 2 or dim != 0:
        raise Exception('Unimplemented fast_cumsum_sub_one() of data = %s and dim = %s' % (data.size(), dim))
    if not data.is_cuda or not use_fast_cumsum or not has_extension:
        return torch_cumsum_sub_one(data)
    return torch.ops.tutel_ops.cumsum(data)


def compute_sorted_location(x, importance_scores):
    sorted_x = x[importance_scores.argsort(dim=0)]
    sorted_cumsum = fast_cumsum_sub_one(sorted_x) * sorted_x
    return sorted_cumsum[importance_scores.argsort(dim=0).argsort(dim=0)]


def get_world_rank(group=None):
    try:
        return dist.get_rank(group)
    except:
        return 0


def get_world_size(group=None):
    try:
        return dist.get_world_size(group)
    except:
        return 1


def simple_all_reduce(input, group=None, op=torch.distributed.ReduceOp.SUM, inplace=False):
    world_size = get_world_size(group)
    if world_size == 1:
        return input
    output = input if inplace else torch.clone(input, memory_format=torch.contiguous_format)
    dist.all_reduce(output, op=op, group=group)
    return output


class GatingDecoder(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Any', config: 'Any', expert_output: 'Tensor', *gates_):
        ctx.config = config
        if gates_:
            ctx.gates_h2 = [(x.view(-1, 1).repeat(1, 2) if x.dtype == torch.float16 else x) for x in gates_]
        else:
            ctx.gates_h2 = [ctx.config.ones_helper] * len(ctx.config.indices_)
        ctx.save_for_backward(expert_output)
        last_result = None
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
            single_output = torch.empty([config.sample_size, config.model_dim], dtype=expert_output.dtype, device=expert_output.device)
            config.func_bwd_data(g, i, l, single_output, expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
            last_result = single_output if last_result is None else last_result + single_output
        return last_result

    @staticmethod
    def backward(ctx: 'Any', combined_output: 'Tensor'):
        combined_output = combined_output.contiguous()
        expert_output = ctx.saved_tensors[0]
        grad_expert_output = torch.zeros(expert_output.shape, dtype=combined_output.dtype, device=combined_output.device)
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
            ctx.config.func_fwd(g, i, l, combined_output, grad_expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
        grad_gates = []
        if id(ctx.gates_h2[0]) != id(ctx.config.ones_helper):
            for i, l in zip(ctx.config.indices_, ctx.config.locations_):
                grad_gates1_s = torch.empty([ctx.config.sample_size], dtype=combined_output.dtype, device=combined_output.device)
                ctx.config.func_bwd_gate(grad_gates1_s, i, l, combined_output, expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
                grad_gates.append(grad_gates1_s)
        return None, grad_expert_output, *grad_gates


class GatingEncoder(torch.autograd.Function):

    @staticmethod
    def forward(ctx: 'Any', config: 'Any', reshaped_input: 'Tensor', *gates_):
        ctx.config = config
        if gates_:
            ctx.gates_h2 = [(x.view(-1, 1).repeat(1, 2) if x.dtype == torch.float16 else x) for x in gates_]
        else:
            ctx.gates_h2 = [ctx.config.ones_helper] * len(ctx.config.indices_)
        ctx.save_for_backward(reshaped_input)
        dispatched_input = torch.zeros([ctx.config.num_global_experts * ctx.config.capacity, ctx.config.model_dim], dtype=reshaped_input.dtype, device=reshaped_input.device)
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
            ctx.config.func_fwd(g, i, l, reshaped_input, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
        return dispatched_input

    @staticmethod
    def backward(ctx: 'Any', dispatched_input: 'Tensor'):
        dispatched_input = dispatched_input.contiguous()
        last_result = None
        reshaped_input = ctx.saved_tensors[0]
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
            grad_data = torch.empty(reshaped_input.shape, dtype=dispatched_input.dtype, device=dispatched_input.device)
            ctx.config.func_bwd_data(g, i, l, grad_data, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
            last_result = grad_data if last_result is None else last_result + grad_data
        grad_gates = []
        if id(ctx.gates_h2[0]) != id(ctx.config.ones_helper):
            for i, l in zip(ctx.config.indices_, ctx.config.locations_):
                grad_gates1_s = torch.empty([ctx.config.sample_size], dtype=dispatched_input.dtype, device=dispatched_input.device)
                ctx.config.func_bwd_gate(grad_gates1_s, i, l, reshaped_input, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
                grad_gates.append(grad_gates1_s)
        return None, last_result, *grad_gates


class TutelMoeFastDispatcher:
    kernel_pool = dict()
    ones_helper = None

    def __init__(self, num_global_experts, capacity, model_dim, dispatch_dtype):
        self.num_global_experts = int(num_global_experts)
        self.capacity = int(capacity)
        self.model_dim = int(model_dim)
        self.dtype = dispatch_dtype
        if IS_HIP_EXTENSION or dispatch_dtype != torch.float16:
            self.dtype = torch.float32
        self.original_dtype = dispatch_dtype
        self.aligned_dim = model_dim // (2 if self.dtype == torch.float16 else 1)
        self.is_cuda = None

    def update(self, indices_, locations_, gates_, capacity=None, is_postscore=True):
        self.indices_ = [x.view(-1) for x in indices_]
        self.locations_ = [x for x in locations_]
        self.gates_ = [x for x in gates_]
        self.is_postscore = is_postscore
        self.sample_size, self.capacity = int(self.indices_[0].size(0)), int(capacity) or self.capacity
        if self.is_cuda != indices_[0].is_cuda:
            self.is_cuda = indices_[0].is_cuda
            if self.is_cuda not in TutelMoeFastDispatcher.kernel_pool:
                self.func_fwd = jit_kernel.create_forward(self.dtype, indices_[0].is_cuda)
                self.func_bwd_data = jit_kernel.create_backward_data(self.dtype, indices_[0].is_cuda)
                self.func_bwd_gate = jit_kernel.create_backward_gate(self.dtype, indices_[0].is_cuda)
                TutelMoeFastDispatcher.kernel_pool[self.is_cuda] = self.func_fwd, self.func_bwd_data, self.func_bwd_gate
            else:
                self.func_fwd, self.func_bwd_data, self.func_bwd_gate = TutelMoeFastDispatcher.kernel_pool[self.is_cuda]
        if TutelMoeFastDispatcher.ones_helper is None or TutelMoeFastDispatcher.ones_helper.size(0) < self.sample_size:
            TutelMoeFastDispatcher.ones_helper = torch.ones([self.sample_size, 2], dtype=self.dtype, device=self.indices_[0].device)
        if TutelMoeFastDispatcher.ones_helper.is_cuda != self.indices_[0].is_cuda:
            TutelMoeFastDispatcher.ones_helper = torch.ones([TutelMoeFastDispatcher.ones_helper.size(0), 2], dtype=self.dtype, device=self.indices_[0].device)
        self.ones_helper = TutelMoeFastDispatcher.ones_helper

    def encode(self, data):
        if self.is_postscore:
            return GatingEncoder.apply(self, data.to(self.dtype))
        else:
            return GatingEncoder.apply(self, data.to(self.dtype), *self.gates_)

    def decode(self, data):
        if self.is_postscore:
            return GatingDecoder.apply(self, data.to(self.dtype), *self.gates_)
        else:
            return GatingDecoder.apply(self, data.to(self.dtype))


def fast_decode(data, critial_data, is_postscore=True):
    assert data.is_contiguous(), 'Input tensor for encode/decode should be in contiguous memory format.'
    num_global_experts = critial_data[0]
    dispatcher = TutelMoeFastDispatcher(num_global_experts, 0, data.size(-1), data.dtype)
    dispatcher.update(*critial_data[1:-1], is_postscore=is_postscore)
    return dispatcher.decode(data).view(-1, data.size(-1))


def fast_encode(data, critial_data, is_postscore=True):
    assert data.is_contiguous(), 'Input tensor for encode/decode should be in contiguous memory format.'
    num_global_experts = critial_data[0]
    dispatcher = TutelMoeFastDispatcher(num_global_experts, 0, data.size(-1), data.dtype)
    dispatcher.update(*critial_data[1:-1], is_postscore=is_postscore)
    return dispatcher.encode(data).view(num_global_experts, -1, data.size(-1))


def get_dispatch_count(critial_data):
    return critial_data[-1]


class MOELayer(torch.nn.Module):
    """Tutel optimized MOELayer
    """

    @staticmethod
    def global_expert_count(num_local_experts, group=None):
        if not isinstance(num_local_experts, int):
            num_local_experts = -int(1 / (num_local_experts + 1e-05))
        world_size = C.get_world_size(group)
        if num_local_experts == 0:
            raise Exception('Invalid value of num_local_experts: %d' % num_local_experts)
        if num_local_experts > 0:
            return num_local_experts * world_size
        assert world_size % -num_local_experts == 0, f'Excepting {-num_local_experts} devices to share an expert param, while global device count is {world_size}.'
        return world_size // -num_local_experts

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buff_name = prefix + '_num_global_experts'
        if buff_name not in state_dict:
            logging.warning(f"\x1b[31mYou are loading a legacy format of checkpoint with at least one Tutel MoE layer inside, which wouldn't support new Tutel feature allowing the number of experts per checkpoint file to mutate.\x1b[0m")
            logging.warning(f'\x1b[31m  The next time you overwrite it with new checkpoint, the recording format will be updated automatically.\x1b[0m')
            logging.warning(f"\x1b[31m  However, the new format won't be compatible with early Tutel versions, unless you force loading it with `model.load_state_dict(.., strict=False)`.\x1b[0m")
            state_dict[buff_name] = self._num_global_experts
        else:
            state_experts, expect_experts = int(state_dict[buff_name]), self.num_global_experts
            assert state_experts == expect_experts, 'Failed to load state from checkpoint: the number of global experts mismatch (%s <- %s)' % (expect_experts, state_experts)
        for name, param in self.experts.named_parameters():
            buff_name = prefix + 'experts.' + name
            if buff_name not in state_dict:
                logging.warning('Could not find parameter `%s` in state_dict, zero values will be filled into this parameter.' % buff_name)
                state_dict[buff_name] = torch.zeros_like(param)
            if state_dict[buff_name].numel() == param.numel():
                state_dict[buff_name] = state_dict[buff_name].view(param.shape)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    @property
    def num_global_experts(self):
        return int(self._num_global_experts)

    def __init__(self, gate_type, model_dim: 'int', experts=None, scan_expert_func=None, result_func=None, group=None, seeds=None, a2a_ffn_overlap_degree=1, is_postscore=True, batch_prioritized_routing=False, normalize_gate=True, is_gshard_loss=True, parallel_type='adaptive:1', use_2dh=False, **kwargs):
        super().__init__()
        assert model_dim % 2 == 0, 'Model_dim (%s) must be even value, while this Model_dim mod 2 > 0.' % model_dim
        group = group or dist.group.WORLD
        if 'pad_samples' in kwargs:
            logging.warning(f'`pad_samples` option in Tutel Moe-layer has been deprecated, as Tutel always assumes `pad_samples=False` for better efficiency.')
            kwargs.pop('pad_samples')
        for k in kwargs:
            raise Exception('Unrecognized argument provided to Tutel Moe-layer: %s' % k)
        self.group = group
        self.result_func = result_func
        self.skip_moe = int(os.environ.get('SKIP_MOE', '0')) != 0
        self.num_local_experts = experts.pop('count_per_node', 1)
        if self.num_local_experts == -1:
            self.num_local_experts = 1
        self.register_buffer('_num_global_experts', torch.tensor(MOELayer.global_expert_count(self.num_local_experts, self.group)))
        self.world_size = C.get_world_size(self.group)
        if self.num_global_experts < self.world_size:
            self.sharded_count = self.world_size // self.num_global_experts
            self.num_local_experts = 1
        else:
            self.sharded_count = 1
        self.auto_parallel, self.adaptive_degree, self.use_model_parallel = False, self.sharded_count, True
        self.valid_rs = [0] + [i for i in range(1, self.sharded_count + 1) if self.sharded_count % i == 0]
        if parallel_type.startswith('adaptive:'):
            self.adaptive_degree = int(parallel_type[parallel_type.index(':') + 1:])
            self.adaptive_degree = min(max(self.adaptive_degree, 0), self.sharded_count)
            if self.adaptive_degree not in self.valid_rs:
                raise Exception('Unexpected value of adaptive_degree: %d, expecting a candidate within %s.' % (self.adaptive_degree, self.valid_rs))
        elif self.sharded_count == 1:
            pass
        elif parallel_type in ('data', 'model'):
            self.adaptive_degree = 1 if parallel_type == 'data' else self.sharded_count
        elif parallel_type == 'auto':
            self.adaptive_degree = 1
        else:
            raise Exception('Unrecognized parallel type specified: %s' % parallel_type)
        self.model_dim = model_dim
        self.is_postscore = is_postscore
        self.batch_prioritized_routing = batch_prioritized_routing
        if int(os.environ.get('BATCH_PRIO', 0)) != 0:
            self.batch_prioritized_routing = True
        self.normalize_gate = normalize_gate
        self.is_gshard_loss = is_gshard_loss
        self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        self.use_2dh = use_2dh
        if seeds is not None and seeds[1] is not None:
            torch.manual_seed(seeds[1])
        experts_type = experts.pop('type')
        if experts_type == 'custom':
            expert_module = experts.pop('module')
            experts['model_dim'] = self.model_dim
            experts['local_experts'] = self.num_local_experts
            experts['sharded_count'] = self.sharded_count
            self.experts = cast(ModuleList, expert_module(**experts))
        else:
            assert re.match('[a-zA-Z0-9\\_]+', experts_type), 'Expert type must only include digits, letters and underline characters.'
            try:
                fused_experts = importlib.import_module(f'...experts.{experts_type}', __name__)
            except ModuleNotFoundError:
                raise Exception('Builtin expert type is not recognized: %s' % experts_type)
            if experts_type == 'ffn':
                assert 'fused_custom_fn' not in experts, '`fused_custom_fn` option for Tutel Moe-layer has been deprecated, please follows helloworld_from_scratch.py for custom construction instead.'
                assert 'implicit_dropout_p' not in experts, '`implicit_dropout_p` option for Tutel Moe-layer has been deprecated, please use torch.nn.Dropout(p=implicit_dropout_p) on custom activation_fn (for fc1_dropout) and after Tutel Moe-layer (for fc2_dropout) instead.'
            experts['model_dim'] = self.model_dim
            experts['local_experts'] = self.num_local_experts
            experts['sharded_count'] = self.sharded_count
            self.experts = fused_experts.ExpertModule(**experts)
        if scan_expert_func is not None:
            for n, p in self.experts.named_parameters():
                scan_expert_func(n, p)
        for n, p in self.experts.named_parameters():
            setattr(p, '_tutel_expert', True)
        if isinstance(gate_type, str):
            assert re.match('^Top[0-9]+Gate$', gate_type), 'Unrecognized gate_type: %s' % gate_type
            top_k = int(gate_type[3:-4])
            logging.warning(f"gate_type value `{gate_type}` in Tutel Moe-layer has been deprecated, please use gate_type = {{'type': 'top', 'k': {top_k}}} instead.")
            gate_type = {'type': 'top', 'k': top_k}
        if not isinstance(gate_type, list):
            gate_type = [gate_type]
        self.gates = []
        for gi, single_gate_type in enumerate(gate_type):
            gate_type = single_gate_type['type']
            single_gate_type.pop('type')
            assert re.match('[a-zA-Z0-9\\_]+', gate_type), 'Gate type must only include digits, letters and underline characters.'
            if seeds is not None and seeds[0] is not None:
                torch.manual_seed(seeds[0] + gi)
            try:
                single_gate = importlib.import_module(f'...gates.{gate_type}', __name__)
            except ModuleNotFoundError:
                raise Exception('Unrecognized gate_type: %s' % gate_type)
            gate_module = single_gate.Gate(model_dim=self.model_dim, num_global_experts=self.num_global_experts, **single_gate_type)
            if not hasattr(gate_module, 'gate_noise'):
                gate_module.gate_noise = single_gate_type.get('gate_noise', 0.0)
            if not hasattr(gate_module, 'capacity_factor'):
                gate_module.capacity_factor = single_gate_type.get('capacity_factor', float(os.environ.get('CAP_FACTOR', 1.0)))
            self.gates += [gate_module]
        self.gates = ModuleList(self.gates)
        if seeds is not None and len(seeds) > 2 and seeds[2] is not None:
            torch.manual_seed(seeds[2])

    def extra_repr(self):
        return 'Top-K(s) = %s, Total-Experts = %d [managed by %d device(s)],' % ([f'k={x.top_k}, noise={x.gate_noise}' for x in self.gates], self.num_global_experts, self.world_size)

    def get_parameter_iterator(self, param_type):
        if param_type == 'gate':
            return self.gates.named_parameters()
        elif param_type == 'local_experts':
            return self.experts.named_parameters()
        else:
            raise Exception('Specified parameter type is not recognized: %s. Valid `param_type` includes: gate, local_experts.' % param_type)

    def expert_local(self, x, reserve_shape):
        y = self.experts(x.view(x.size(0), x.size(1), *reserve_shape), self)
        self.protected_shape = y.shape
        return y.reshape(y.size(0), y.size(1), -1)

    def forward(self, input: 'Tensor', gate_index=0, capacity_factor=None, top_k=None, a2a_ffn_overlap_degree=None, reserve_dims=1, inequivalent_tokens=False, adaptive_r=None, megablocks_size=0):
        if self.skip_moe:
            result_output = input
            result_output.l_aux = None
            return self.result_func(result_output) if self.result_func is not None else result_output
        original_shape, original_dtype = input.shape, input.dtype
        assert len(original_shape) >= 2, 'Input data must be at least 2D tensor: (s)amples, .., (m)odel_dim'
        x = input.reshape(-1, original_shape[-reserve_dims:].numel())
        if torch.is_autocast_enabled():
            x = cast_if_autocast_enabled(x)
        else:
            for p in self.experts.parameters():
                x = x
                break
        gctx = self.gates[gate_index]
        if a2a_ffn_overlap_degree is not None:
            self.a2a_ffn_overlap_degree = a2a_ffn_overlap_degree
        a2a_ffn_overlap_degree = self.a2a_ffn_overlap_degree
        top_k = top_k or gctx.top_k
        if megablocks_size > 0:
            if self.num_local_experts <= 1 or torch.is_grad_enabled() or self.world_size > 1:
                megablocks_size = 0

        def routing():
            logits = gctx(x)
            if self.training and gctx.gate_noise > 0:
                logits_w_noise = logits + gctx.gate_noise * torch.randn_like(logits) / self.num_global_experts
            else:
                logits_w_noise = logits
            scores = F.softmax(logits_w_noise, dim=1)
            if self.is_gshard_loss:
                _loss_fn = lambda gates, topk_ids: losses.gshard_loss(gates, topk_ids)
            else:
                _loss_fn = lambda gates, topk_ids: losses.load_importance_loss(F.softmax(logits, dim=1), logits_w_noise.gather(index=topk_ids, dim=1), self.num_global_experts, gctx.gate_noise)
            mega_up = max(megablocks_size, 1)
            return logits.dtype, extract_critical(scores, top_k=top_k, loss_fn=_loss_fn, capacity_factor=capacity_factor or gctx.capacity_factor, batch_prioritized_routing=self.batch_prioritized_routing, normalize_gate=self.normalize_gate, group=self.group, alignment=(self.sharded_count * a2a_ffn_overlap_degree + mega_up - 1) // mega_up * mega_up, inequivalent_tokens=inequivalent_tokens)
        if x.is_cuda:
            with torch.amp.autocast(enabled=False):
                logits_dtype, (crit, l_aux) = routing()
        else:
            logits_dtype, (crit, l_aux) = routing()
        self.megablocks_size = megablocks_size
        self.dispatch_count = get_dispatch_count(crit)
        y = fast_encode(x.to(logits_dtype), crit, self.is_postscore)
        if adaptive_r is not None:
            self.adaptive_degree = adaptive_r
        if self.adaptive_degree == 0:
            y = self.expert_local(y, original_shape[-reserve_dims:])
        else:
            if self.auto_parallel:
                self.use_model_parallel = y.numel() * (self.sharded_count - 1) * 2 < sum([x.numel() for x in self.experts.parameters()])
            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = y.repeat(1, self.adaptive_degree, 1).view(self.world_size, -1, y.size(2))
                else:
                    y = y.view(self.world_size, -1, y.size(2))
            if a2a_ffn_overlap_degree > 1 and y.is_cuda:

                def expert_fn(expert_input):
                    return self.expert_local(expert_input, original_shape[-reserve_dims:])
                y = a2a_ffn_overlap_forward(y, expert_fn=expert_fn, a2a_ffn_overlap_degree=a2a_ffn_overlap_degree, use_2dh=self.use_2dh, group=self.group)
            else:
                y = C.all_to_all(y, 1, 0, use_2dh=self.use_2dh, group=self.group)
                y = self.expert_local(y, original_shape[-reserve_dims:])
                y = C.all_to_all(y, 0, 1, use_2dh=self.use_2dh, group=self.group)
            if self.num_global_experts < self.world_size:
                if self.use_model_parallel:
                    y = torch.sum(y.view(self.num_global_experts, self.adaptive_degree, -1, y.size(2)), dim=1)
                else:
                    y = y.view(self.num_global_experts, -1, y.size(2))
        y = fast_decode(y, crit, self.is_postscore)
        y = y.view(list(original_shape[:-reserve_dims]) + list(self.protected_shape[-reserve_dims:]))
        self.l_aux = y.l_aux = l_aux
        return self.result_func(y) if self.result_func is not None else y


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CosineTopKGate,
     lambda: ([], {'model_dim': 4, 'num_global_experts': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LinearTopKGate,
     lambda: ([], {'model_dim': 4, 'num_global_experts': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

