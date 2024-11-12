
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


import torch


import torch.nn.functional as F


import numpy as np


import copy


from collections import defaultdict


from torch import nn


from torch import optim


import pandas as pd


from torch.utils.data import DataLoader


import matplotlib


import matplotlib.pyplot as plt


from matplotlib.gridspec import GridSpec


from sklearn.model_selection import StratifiedKFold


import math


from scipy import optimize


from torch.nn.modules.loss import _Loss


from sklearn.base import BaseEstimator


from sklearn.base import TransformerMixin


from sklearn.utils.validation import check_is_fitted


from sklearn.model_selection import GridSearchCV


from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import make_pipeline


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.data.dataloader import default_collate


import warnings


import scipy.sparse as sp


from scipy.sparse.linalg import expm


EPS = 0.0001


def exp(x, alpha):
    return torch.exp(alpha * (x - 1.0))


def d_exp(x, alpha):
    return alpha * exp(x, alpha)


def linear(x, alpha):
    return x


d_kernels = {'exp': d_exp, 'linear': linear}


kernels = {'exp': exp, 'linear': linear}


MAXRAM = int(5000000000.0)


class PathConv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, path_indices, features):
        if features.is_cuda:
            output = gckn_fast_cuda.path_conv_forward(path_indices, features)
        else:
            output = gckn_fast_cpu.path_conv_forward(path_indices, features)
        ctx.save_for_backward(path_indices)
        ctx.size = features.size()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(ctx.size)
        if grad_output.is_cuda:
            gckn_fast_cuda.path_conv_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables)
        else:
            gckn_fast_cpu.path_conv_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables)
        return None, grad_input


def dpooling_backward(grad_input, grad_output, indices, pooling='sum'):
    if pooling == 'max':
        if grad_output.is_cuda:
            pooling_cuda.max_backward(grad_input, grad_output, indices)
        else:
            pooling_cpu.max_backward(grad_input, grad_output, indices)
    elif pooling == 'sum':
        if grad_output.is_cuda:
            pooling_cuda.sum_backward(grad_input, grad_output, indices, False)
        else:
            pooling_cpu.sum_backward(grad_input, grad_output, indices, False)
    elif pooling == 'mean':
        if grad_output.is_cuda:
            pooling_cuda.sum_backward(grad_input, grad_output, indices, True)
        else:
            pooling_cpu.sum_backward(grad_input, grad_output, indices, True)


def dpooling_forward(input, kernel_size, pooling='sum'):
    kernel_size = kernel_size.cumsum(0)
    active_indices = kernel_size
    if pooling == 'max':
        if input.is_cuda:
            output, active_indices = pooling_cuda.max_forward(input, kernel_size)
        else:
            output, active_indices = pooling_cpu.max_forward(input, kernel_size)
    elif pooling == 'sum':
        if input.is_cuda:
            output = pooling_cuda.sum_forward(input, kernel_size, False)
        else:
            output = pooling_cpu.sum_forward(input, kernel_size, False)
    elif pooling == 'mean':
        if input.is_cuda:
            output = pooling_cuda.sum_forward(input, kernel_size, True)
        else:
            output = pooling_cpu.sum_forward(input, kernel_size, True)
    return output, active_indices


def get_batch_indices(array, batch_size):
    indices = [0]
    s = 0
    for i, v in enumerate(array):
        s += v.item()
        if s > batch_size:
            indices.append(i)
            s = v.item()
    indices.append(len(array))
    return indices


def path_conv_backward(grad_input, grad_output, path_indices):
    if grad_output.is_cuda:
        gckn_fast_cuda.path_conv_backward(grad_input, grad_output, path_indices)
    else:
        gckn_fast_cpu.path_conv_backward(grad_input, grad_output, path_indices)


def path_conv_forward(path_indices, features):
    if features.is_cuda:
        output = gckn_fast_cuda.path_conv_forward(path_indices, features)
    else:
        output = gckn_fast_cpu.path_conv_forward(path_indices, features)
    return output


class PathConvAggregation(torch.autograd.Function):
    """Path extraction + convolution + aggregation
    features: n_nodes x path_size x hidden_size
    path_indices: n_paths x path_size
    kernel_size: n_nodes (sum = n_paths)
    pooling: {sum, mean, max}
    """

    @staticmethod
    def forward(ctx, features, path_indices, kernel_size, pooling='sum', kappa=torch.exp, d_kappa=torch.exp):
        batch_size = MAXRAM // (features.shape[-1] * features.element_size())
        indices = get_batch_indices(kernel_size, batch_size)
        batch_index = 0
        output = []
        active_indices = []
        n_paths_list = []
        for i in range(len(indices) - 1):
            batch_kernel_size = kernel_size[indices[i]:indices[i + 1]]
            size = batch_kernel_size.sum().item()
            batch_path_indices = path_indices[batch_index:batch_index + size]
            embeded = path_conv_forward(batch_path_indices, features)
            embeded = kappa(embeded)
            embeded, active_index = dpooling_forward(embeded, batch_kernel_size, pooling=pooling)
            output.append(embeded)
            active_indices.append(active_index)
            n_paths_list.append(size)
            batch_index += size
        output = torch.cat(output)
        active_indices = torch.cat(active_indices)
        ctx.save_for_backward(path_indices, active_indices, features)
        ctx.indices = indices
        ctx.size = features.size()
        ctx.d_kappa = d_kappa
        ctx.pooling = pooling
        ctx.n_paths_list = n_paths_list
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_zeros(ctx.size)
        indices = ctx.indices
        path_indices, active_indices, features = ctx.saved_variables
        batch_index = 0
        grad_embed = grad_output.new_zeros(max(ctx.n_paths_list), ctx.size[-1])
        for i in range(len(indices) - 1):
            n_paths = ctx.n_paths_list[i]
            batch_path_indices = path_indices[batch_index:batch_index + n_paths]
            batch_index += n_paths
            grad_embed.zero_()
            dpooling_backward(grad_embed, grad_output[indices[i]:indices[i + 1]], active_indices[indices[i]:indices[i + 1]], ctx.pooling)
            embeded = path_conv_forward(batch_path_indices, features)
            embeded = ctx.d_kappa(embeded)
            grad_embed[:embeded.shape[0]].mul_(embeded)
            path_conv_backward(grad_input, grad_embed, batch_path_indices)
        return grad_input, None, None, None, None, None


class DPoolingMax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, kernel_size):
        kernel_size = kernel_size.cumsum(0)
        if input.is_cuda:
            output, active_indices = pooling_cuda.max_forward(input, kernel_size)
        else:
            output, active_indices = pooling_cpu.max_forward(input, kernel_size)
        ctx.save_for_backward(active_indices)
        ctx.size = input.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(ctx.size)
        if grad_output.is_cuda:
            pooling_cuda.max_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables)
        else:
            pooling_cpu.max_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables)
        return grad_input, None


class DPoolingSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, kernel_size, mean=False):
        kernel_size = kernel_size.cumsum(0)
        if input.is_cuda:
            output = pooling_cuda.sum_forward(input, kernel_size, mean)
        else:
            output = pooling_cpu.sum_forward(input, kernel_size, mean)
        ctx.save_for_backward(kernel_size)
        ctx.mean = mean
        ctx.size = input.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(ctx.size)
        if grad_output.is_cuda:
            pooling_cuda.sum_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables, ctx.mean)
        else:
            pooling_cpu.sum_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables, ctx.mean)
        return grad_input, None, None


def dpooling(input, kernel_size, pooling='sum'):
    if pooling == 'sum':
        return DPoolingSum.apply(input, kernel_size, False)
    elif pooling == 'mean':
        return DPoolingSum.apply(input, kernel_size, True)
    elif pooling == 'max':
        return DPoolingMax.apply(input, kernel_size)
    else:
        raise ValueError('Not implemented!')


def path_conv_agg(features, path_indices, kernel_size, pooling='sum', kappa=torch.exp, d_kappa=torch.exp, mask=None):
    ram_saving = MAXRAM <= 2 * path_indices.shape[0] * features.shape[-1] * features.element_size()
    if ram_saving and mask is None:
        return PathConvAggregation.apply(features, path_indices, kernel_size, pooling, kappa, d_kappa)
    embeded = PathConv.apply(path_indices, features)
    embeded = kappa(embeded)
    if mask is not None:
        embeded = embeded * mask.view(-1, 1)
    embeded = dpooling(embeded, kernel_size, pooling)
    return embeded


def init_kmeans(x, n_clusters, norm=1.0, n_local_trials=None, use_cuda=False):
    n_samples, n_features = x.size()
    clusters = torch.Tensor(n_clusters, n_features)
    if use_cuda:
        clusters = clusters
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))
    clusters[0] = x[np.random.randint(n_samples)]
    closest_dist_sq = 2 * (norm - clusters[[0]].mm(x.t()))
    closest_dist_sq = closest_dist_sq.view(-1)
    current_pot = closest_dist_sq.sum().item()
    for c in range(1, n_clusters):
        rand_vals = np.random.random_sample(n_local_trials).astype('float32') * current_pot
        rand_vals = np.minimum(rand_vals, current_pot * (1.0 - EPS))
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(-1).cpu(), rand_vals)
        distance_to_candidates = 2 * (norm - x[candidate_ids].mm(x.t()))
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            new_dist_sq = torch.min(closest_dist_sq, distance_to_candidates[trial])
            new_pot = new_dist_sq.sum().item()
            if best_candidate is None or new_pot < best_pot:
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq
        clusters[c] = x[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq
    return clusters


def spherical_kmeans(x, n_clusters, max_iters=100, verbose=True, init=None, eps=0.0001):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x kmer_size x n_features): data points
        n_clusters (int): number of clusters
    """
    use_cuda = x.is_cuda
    if x.ndim == 3:
        n_samples, kmer_size, n_features = x.size()
    else:
        n_samples, n_features = x.size()
    if init == 'kmeans++':
        None
        if x.ndim == 3:
            clusters = init_kmeans(x.view(n_samples, -1), n_clusters, norm=kmer_size, use_cuda=use_cuda)
            clusters = clusters.view(n_clusters, kmer_size, n_features)
        else:
            clusters = init_kmeans(x, n_clusters, use_cuda=use_cuda)
    else:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices
        clusters = x[indices]
    prev_sim = np.inf
    for n_iter in range(max_iters):
        cos_sim = x.view(n_samples, -1).mm(clusters.view(n_clusters, -1).t())
        tmp, assign = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            None
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm(dim=-1, keepdim=True).clamp(min=EPS)
        if torch.abs(prev_sim - sim) / (torch.abs(sim) + 1e-20) < 1e-06:
            break
        prev_sim = sim
    return clusters


class PathLayer(nn.Module):

    def __init__(self, input_size, hidden_size, path_size=1, kernel_func='exp', kernel_args=[0.5], pooling='mean', aggregation=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.path_size = path_size
        self.pooling = pooling
        self.aggregation = aggregation and path_size > 1
        self.kernel_func = kernel_func
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == 'exp':
            kernel_args = [(1.0 / kernel_arg ** 2) for kernel_arg in kernel_args]
        self.kernel_args = kernel_args
        self.kernel_func = kernels[kernel_func]
        self.kappa = lambda x: self.kernel_func(x, *self.kernel_args)
        d_kernel_func = d_kernels[kernel_func]
        self.d_kappa = lambda x: d_kernel_func(x, *self.kernel_args)
        self._need_lintrans_computed = True
        self.weight = nn.Parameter(torch.Tensor(path_size, hidden_size, input_size))
        if self.aggregation:
            self.register_buffer('lintrans', torch.Tensor(path_size, hidden_size, hidden_size))
            self.register_buffer('divider', torch.arange(1.0, path_size + 1).view(-1, 1, 1))
        else:
            self.register_buffer('lintrans', torch.Tensor(hidden_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            if w.dim() > 1:
                w.data.uniform_(-stdv, stdv)
        self.normalize_()

    def normalize_(self):
        normalize_(self.weight.data, dim=-1)

    def train(self, mode=True):
        super().train(mode)
        self._need_lintrans_computed = True

    def _compute_lintrans(self):
        if not self._need_lintrans_computed:
            return self.lintrans
        lintrans = torch.bmm(self.weight, self.weight.permute(0, 2, 1))
        if self.aggregation:
            lintrans = lintrans.cumsum(dim=0) / self.divider
        else:
            lintrans = lintrans.mean(dim=0)
        lintrans = self.kappa(lintrans)
        lintrans = ops.matrix_inverse_sqrt(lintrans)
        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data.copy_(lintrans.data)
        return lintrans

    def forward(self, features, paths_indices, other_info):
        """
        features: n_nodes x (input_path_size) x input_size
        paths_indices: n_paths x path_size (values < n_nodes)
        output: n_nodes x ((input_path_size) x path_size) x input_size
        """
        self.normalize_()
        norms = features.norm(dim=-1, keepdim=True)
        output = torch.tensordot(features, self.weight, dims=[[-1], [-1]])
        output = output / norms.clamp(min=EPS).unsqueeze(2)
        n_nodes = output.shape[0]
        if output.ndim == 4:
            output = output.permute(0, 2, 1, 3).contiguous()
        mask = None
        if self.aggregation:
            mask = [None for _ in range(self.path_size)]
        if 'mask' in other_info and self.path_size > 1:
            mask = other_info['mask']
        output = output.view(n_nodes, self.path_size, -1)
        if self.aggregation:
            outputs = []
            for i in range(self.path_size):
                embeded = path_conv_agg(output, paths_indices[i], other_info['n_paths'][i], self.pooling, self.kappa, self.d_kappa, mask[i])
                outputs.append(embeded)
            output = torch.stack(outputs, dim=0)
            output = output.view(self.path_size, -1, self.hidden_size)
            output = norms.view(1, -1, 1) * output
        else:
            output = path_conv_agg(output, paths_indices[self.path_size - 1], other_info['n_paths'][self.path_size - 1], self.pooling, self.kappa, self.d_kappa, mask)
            output = output.view(n_nodes, -1, self.hidden_size)
            output = norms.view(n_nodes, -1, 1) * output
        lintrans = self._compute_lintrans()
        if self.aggregation:
            output = output.bmm(lintrans)
            output = output.permute(1, 0, 2)
            output = output.reshape(n_nodes, -1, self.hidden_size)
            output = output.contiguous()
        else:
            output = torch.tensordot(output, lintrans, dims=[[-1], [-1]])
        return output

    def sample_paths(self, features, paths_indices, n_sampling_paths=1000):
        """Sample paths for a given of features and paths
        features: n_nodes x (input_path_size) x input_size
        paths_indices: n_paths x path_size
        output: n_sampling_paths x path_size x input_size
        """
        paths_indices = paths_indices[self.path_size - 1]
        if self.path_size == 1:
            features = features.permute(1, 0, 2).reshape(-1, self.input_size)
            n_all_paths = features.shape[0]
            n_sampling_paths = min(n_all_paths, n_sampling_paths)
            indices = torch.randperm(n_all_paths)[:n_sampling_paths]
            paths = features[indices]
            return paths.view(n_sampling_paths, 1, self.input_size).contiguous()
        n_all_paths = paths_indices.shape[0]
        indices = torch.randperm(n_all_paths)[:min(n_all_paths, n_sampling_paths)]
        paths = F.embedding(paths_indices[indices], features)
        if paths.ndim == 4:
            paths = paths.permute(0, 2, 1, 3)
            paths = paths.reshape(-1, self.path_size, self.input_size)
            paths = paths[:min(paths.shape[0], n_sampling_paths)]
        return paths

    def unsup_train(self, paths, init=None):
        """Unsupervised training for path layer
        paths: n x path_size x input_size
        self.weight: path_size x hidden_size x input_size
        """
        None
        normalize_(paths, dim=-1)
        weight = spherical_kmeans(paths, self.hidden_size, init='kmeans++')
        weight = weight.permute(1, 0, 2)
        self.weight.data.copy_(weight)
        self.normalize_()
        self._need_lintrans_computed = True


class NodePooling(nn.Module):

    def __init__(self, pooling='mean'):
        super().__init__()
        self.pooling = pooling

    def reset_parameters(self):
        pass

    def forward(self, features, other_info):
        """
        features: n_nodes x (input_path_size) x input_size
        output: n_graphs x input_size
        """
        features = features.permute(0, 2, 1).contiguous()
        n_nodes = features.shape[0]
        output = dpooling(features.view(n_nodes, -1), other_info['n_nodes'], self.pooling)
        return output


class HingeLoss(_Loss):

    def __init__(self, nclass=10, weight=None, size_average=None, reduce=None, reduction='elementwise_mean', pos_weight=None, squared=True):
        super(HingeLoss, self).__init__(size_average, reduce, reduction)
        self.nclass = nclass
        self.squared = squared
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        if not target.size(0) == input.size(0):
            raise ValueError('Target size ({}) must be the same as input size ({})'.format(target.size(), input.size()))
        if self.pos_weight is not None:
            pos_weight = 1 + (self.pos_weight - 1) * target
        target = 2 * F.one_hot(target, num_classes=self.nclass) - 1
        target = target.float()
        loss = F.relu(1.0 - target * input)
        if self.squared:
            loss = 0.5 * loss ** 2
        if self.weight is not None:
            loss = loss * self.weight
        if self.pos_weight is not None:
            loss = loss * pos_weight
        loss = loss.sum(dim=-1)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'elementwise_mean':
            return loss.mean()
        else:
            return loss.sum()


class PathSequential(nn.Module):

    def __init__(self, input_size, hidden_sizes, path_sizes, kernel_funcs=None, kernel_args_list=None, pooling='mean', aggregation=False, **kwargs):
        super(PathSequential, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.path_sizes = path_sizes
        self.n_layers = len(hidden_sizes)
        self.aggregation = aggregation
        layers = []
        output_size = hidden_sizes[-1]
        for i in range(self.n_layers):
            if kernel_funcs is None:
                kernel_func = 'exp'
            else:
                kernel_func = kernel_funcs[i]
            if kernel_args_list is None:
                kernel_args = 0.5
            else:
                kernel_args = kernel_args_list[i]
            layer = PathLayer(input_size, hidden_sizes[i], path_sizes[i], kernel_func, kernel_args, pooling, aggregation, **kwargs)
            layers.append(layer)
            input_size = hidden_sizes[i]
            if aggregation:
                output_size *= path_sizes[i]
        self.output_size = output_size
        self.layers = nn.ModuleList(layers)

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return self.n_layers

    def __iter__(self):
        return iter(self.layers._modules.values())

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features, paths_indices, other_info):
        output = features
        for layer in self.layers:
            output = layer(output, paths_indices, other_info)
        return output

    def representation(self, features, paths_indices, other_info, n=-1):
        if n == -1:
            n = self.n_layers
        for i in range(n):
            features = self.layers[i](features, paths_indices, other_info)
        return features

    def normalize_(self):
        for module in self.layers:
            module.normalize_()

    def unsup_train(self, data_loader, n_sampling_paths=100000, init=None, use_cuda=False):
        self.train(False)
        for i, layer in enumerate(self.layers):
            None
            n_sampled_paths = 0
            try:
                n_paths_per_batch = (n_sampling_paths + len(data_loader) - 1) // len(data_loader)
            except Exception:
                n_paths_per_batch = 1000
            paths = torch.Tensor(n_sampling_paths, layer.path_size, layer.input_size)
            if use_cuda:
                paths = paths
            for data in data_loader.make_batch():
                if n_sampled_paths >= n_sampling_paths:
                    continue
                features = data['features']
                paths_indices = data['paths']
                n_paths = data['n_paths']
                n_nodes = data['n_nodes']
                if use_cuda:
                    features = features
                    if isinstance(n_paths, list):
                        paths_indices = [p for p in paths_indices]
                        n_paths = [p for p in n_paths]
                    else:
                        paths_indices = paths_indices
                        n_paths = n_paths
                    n_nodes = n_nodes
                with torch.no_grad():
                    features = self.representation(features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes}, i)
                    paths_batch = layer.sample_paths(features, paths_indices, n_paths_per_batch)
                    size = paths_batch.shape[0]
                    size = min(size, n_sampling_paths - n_sampled_paths)
                    paths[n_sampled_paths:n_sampled_paths + size] = paths_batch[:size]
                    n_sampled_paths += size
            None
            paths = paths[:n_sampled_paths]
            layer.unsup_train(paths, init=init)
        return

    def encode(self, data_loader, use_cuda=False):
        if use_cuda:
            self
        self.eval()
        output = []
        for data in data_loader.make_batch(shuffle=False):
            features = data['features']
            paths_indices = data['paths']
            n_paths = data['n_paths']
            n_nodes = data['n_nodes']
            size = len(n_nodes)
            if use_cuda:
                features = features
                if isinstance(n_paths, list):
                    paths_indices = [p for p in paths_indices]
                    n_paths = [p for p in n_paths]
                else:
                    paths_indices = paths_indices
                    n_paths = n_paths
                n_nodes = n_nodes
            with torch.no_grad():
                batch_out = self(features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes}).cpu()
                batch_out = batch_out.reshape(features.shape[0], -1)
                batch_out = torch.split(batch_out, n_nodes.numpy().tolist())
            output.extend(batch_out)
        return output


class GCKNetFeature(nn.Module):

    def __init__(self, input_size, hidden_sizes, path_sizes, kernel_funcs=None, kernel_args_list=None, pooling='mean', global_pooling='sum', heads=1, out_size=3, max_iter=100, eps=0.1, aggregation=False, **kwargs):
        super().__init__()
        self.path_layers = PathSequential(input_size, hidden_sizes, path_sizes, kernel_funcs, kernel_args_list, pooling, aggregation, **kwargs)
        self.aggregation = aggregation
        self.global_pooling = global_pooling
        self.path_sizes = path_sizes
        self.hidden_sizes = hidden_sizes
        self.node_pooling = NodePooling(global_pooling)
        self.output_size = self.path_layers.output_size

    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()

    def forward(self, input, paths_indices, other_info):
        output = self.path_layers(input, paths_indices, other_info)
        return self.node_pooling(output, other_info)

    def unsup_train(self, data_loader, n_sampling_paths=100000, n_nodes_max=100000, init=None, use_cuda=False):
        self.path_layers.unsup_train(data_loader, n_sampling_paths, init, use_cuda)

    def predict(self, data_loader, use_cuda=False):
        if use_cuda:
            self
        self.eval()
        output = torch.Tensor(data_loader.n, self.output_size)
        batch_start = 0
        for data in data_loader.make_batch(shuffle=False):
            features = data['features']
            paths_indices = data['paths']
            n_paths = data['n_paths']
            n_nodes = data['n_nodes']
            size = len(n_nodes)
            if use_cuda:
                features = features
                if isinstance(n_paths, list):
                    paths_indices = [p for p in paths_indices]
                    n_paths = [p for p in n_paths]
                else:
                    paths_indices = paths_indices
                    n_paths = n_paths
                n_nodes = n_nodes
            with torch.no_grad():
                batch_out = self(features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes}).cpu()
            output[batch_start:batch_start + size] = batch_out
            batch_start += size
        return output, data_loader.labels


class GCKNet(nn.Module):

    def __init__(self, nclass, input_size, hidden_sizes, path_sizes, kernel_funcs=None, kernel_args_list=None, pooling='mean', global_pooling='sum', heads=1, out_size=3, max_iter=100, eps=0.1, aggregation=False, weight_decay=0.0, batch_norm=False, **kwargs):
        super().__init__()
        self.features = GCKNetFeature(input_size, hidden_sizes, path_sizes, kernel_funcs, kernel_args_list, pooling, global_pooling, heads, out_size, max_iter, eps, aggregation, **kwargs)
        self.output_size = self.features.output_size
        self.nclass = nclass
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(self.output_size)
        self.classifier = Linear(self.output_size, nclass, weight_decay)

    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()

    def representation(self, input, paths_indices, other_info):
        return self.features(input, paths_indices, other_info)

    def forward(self, input, paths_indices, other_info):
        features = self.representation(input, paths_indices, other_info)
        if self.batch_norm:
            features = self.bn_layer(features)
        return self.classifier(features)

    def unsup_train(self, data_loader, n_sampling_paths=100000, init=None, use_cuda=False):
        self.features.unsup_train(data_loader=data_loader, n_sampling_paths=n_sampling_paths, init=init, use_cuda=use_cuda)

    def unsup_train_classifier(self, data_loader, criterion, use_cuda=False):
        encoded_data, labels = self.features.predict(data_loader, use_cuda)
        None
        self.classifier.fit(encoded_data, labels, criterion)


def diff_multi_head_attention_forward(query, key, value, pe, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None):
    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by             num_heads'
    scaling = float(head_dim) ** -0.5
    if use_separate_proj_weight is not True:
        if qkv_same:
            q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        elif kv_same:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)
        if in_proj_bias is not None:
            q = nn.functional.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = nn.functional.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:embed_dim * 2])
            v = nn.functional.linear(value, v_proj_weight_non_opt, in_proj_bias[embed_dim * 2:])
        else:
            q = nn.functional.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = nn.functional.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = nn.functional.linear(value, v_proj_weight_non_opt, in_proj_bias)
    k = q
    q = q * scaling
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert bias_k is None
        assert bias_v is None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    src_len = k.size(1)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat([key_padding_mask, torch.zeros((key_padding_mask.size(0), 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device)], dim=1)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
    pe = torch.repeat_interleave(pe, repeats=num_heads, dim=0)
    max_val = attn_output_weights.max(dim=-1, keepdim=True)[0]
    attn_output_weights = torch.exp(attn_output_weights - max_val)
    attn_output_weights = attn_output_weights * pe
    attn_output_weights = attn_output_weights / attn_output_weights.sum(dim=-1, keepdim=True).clamp(min=1e-06)
    attn_output_weights = nn.functional.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class DiffMultiheadAttention(nn.modules.activation.MultiheadAttention):

    def forward(self, query, key, value, pe, key_padding_mask=None, need_weights=True, attn_mask=None):
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return diff_multi_head_attention_forward(query, key, value, pe, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=True, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttentio, module has benn implemented.                         Please re-train your model with the new module', UserWarning)
            return diff_multi_head_attention_forward(query, key, value, pe, self.embed_dim, self.num_heads, self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.add_zero_attn, self.dropout, self.out_proj.weight, self.out_proj.bias, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask)


class DiffTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_norm=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn = DiffMultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        self.scaling = None

    def forward(self, src, pe, degree=None, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, pe, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        if degree is not None:
            src2 = degree.transpose(0, 1).contiguous().unsqueeze(-1) * src2
        else:
            if self.scaling is None:
                self.scaling = 1.0 / pe.diagonal(dim1=1, dim2=2).max().item()
            src2 = (self.scaling * pe.diagonal(dim1=1, dim2=2)).transpose(0, 1).contiguous().unsqueeze(-1) * src2
        src = src + self.dropout1(src2)
        if self.batch_norm:
            bsz = src.shape[1]
            src = src.view(-1, src.shape[-1])
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.batch_norm:
            src = src.view(-1, bsz, src.shape[-1])
        return src


class GlobalAvg1D(nn.Module):

    def __init__(self):
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = (~mask).float().unsqueeze(-1)
        x = x * mask
        return x.sum(dim=1) / mask.sum(dim=1)


class GraphTransformer(nn.Module):

    def __init__(self, in_size, nb_class, d_model, nb_heads, dim_feedforward=2048, dropout=0.1, nb_layers=4, lap_pos_enc=False, lap_pos_enc_dim=0):
        super(GraphTransformer, self).__init__()
        self.embedding = nn.Linear(in_features=in_size, out_features=d_model, bias=False)
        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nb_heads, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalAvg1D()
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, nb_class))

    def forward(self, x, masks, x_pe, x_lap_pos_enc=None, degree=None):
        x = x.permute(1, 0, 2)
        output = self.embedding(x)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        output = self.pooling(output, masks)
        return self.classifier(output)


class DiffTransformerEncoder(nn.TransformerEncoder):

    def forward(self, src, pe, degree=None, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, pe=pe, degree=degree, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class DiffGraphTransformer(nn.Module):

    def __init__(self, in_size, nb_class, d_model, nb_heads, dim_feedforward=2048, dropout=0.1, nb_layers=4, batch_norm=False, lap_pos_enc=False, lap_pos_enc_dim=0):
        super(DiffGraphTransformer, self).__init__()
        self.lap_pos_enc = lap_pos_enc
        self.lap_pos_enc_dim = lap_pos_enc_dim
        if lap_pos_enc and lap_pos_enc_dim > 0:
            self.embedding_lap_pos_enc = nn.Linear(lap_pos_enc_dim, d_model)
        self.embedding = nn.Linear(in_features=in_size, out_features=d_model, bias=False)
        encoder_layer = DiffTransformerEncoderLayer(d_model, nb_heads, dim_feedforward, dropout, batch_norm=batch_norm)
        self.encoder = DiffTransformerEncoder(encoder_layer, nb_layers)
        self.pooling = GlobalAvg1D()
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, nb_class))

    def forward(self, x, masks, pe, x_lap_pos_enc=None, degree=None):
        x = x.permute(1, 0, 2)
        output = self.embedding(x)
        if self.lap_pos_enc and x_lap_pos_enc is not None:
            x_lap_pos_enc = x_lap_pos_enc.transpose(0, 1)
            x_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc)
            output = output + x_lap_pos_enc
        output = self.encoder(output, pe, degree=degree, src_key_padding_mask=masks)
        output = output.permute(1, 0, 2)
        output = self.pooling(output, masks)
        return self.classifier(output)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DiffMultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (DiffTransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (GlobalAvg1D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

