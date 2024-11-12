
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


import pandas as pd


import torch


from torch.utils.data import Dataset


from collections import defaultdict


from sklearn.model_selection import train_test_split


import math


from torch import nn


import torch.nn.functional as F


from scipy import optimize


import copy


from torch.optim.lr_scheduler import ReduceLROnPlateau


from sklearn.model_selection import cross_val_score


from sklearn.linear_model import LogisticRegression


from sklearn.svm import LinearSVC


from torch.autograd import Variable


import random


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torch.optim.lr_scheduler import StepLR


from torch.optim.lr_scheduler import MultiStepLR


import torch.optim as optim


from sklearn.metrics import roc_auc_score


from sklearn.metrics import log_loss


from sklearn.metrics import accuracy_score


from sklearn.metrics import precision_recall_curve


from sklearn.metrics import average_precision_score


from torch.utils.data import TensorDataset


import scipy.io as sio


EPS = 1e-06


def gaussian_filter_1d(size, sigma=None):
    """Create 1D Gaussian filter
    """
    if size == 1:
        return torch.ones(1)
    if sigma is None:
        sigma = (size - 1.0) / (2.0 * math.sqrt(2))
    m = float((size - 1) // 2)
    filt = torch.arange(-m, m + 1)
    filt = torch.exp(-filt.pow(2) / (2.0 * sigma * sigma))
    return filt / torch.sum(filt)


def exp(x, alpha):
    """Element wise non-linearity
    kernel_exp is defined as k(x)=exp(alpha * (x-1))
    return:
        same shape tensor as x
    """
    return torch.exp(alpha * (x - 1.0))


def add_exp(x, alpha):
    return 0.5 * (exp(x, alpha) + x)


kernels = {'exp': exp, 'add_exp': add_exp}


def spherical_kmeans(x, n_clusters, max_iters=100, block_size=None, verbose=True, init=None, eps=0.0001):
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
    if init is None:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices
        clusters = x[indices]
    prev_sim = np.inf
    tmp = x.new_empty(n_samples)
    assign = x.new_empty(n_samples, dtype=torch.long)
    if block_size is None or block_size == 0:
        block_size = x.shape[0]
    for n_iter in range(max_iters):
        for i in range(0, n_samples, block_size):
            end_i = min(i + block_size, n_samples)
            cos_sim = x[i:end_i].view(end_i - i, -1).mm(clusters.view(n_clusters, -1).t())
            tmp[i:end_i], assign[i:end_i] = cos_sim.max(dim=-1)
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


class CKNLayer(nn.Conv1d):

    def __init__(self, in_channels, out_channels, filter_size, padding=0, dilation=1, groups=1, subsampling=1, kernel_func='exp', kernel_args=[0.5], kernel_args_trainable=False):
        if padding == 'SAME':
            padding = (filter_size - 1) // 2
        else:
            padding = 0
        super(CKNLayer, self).__init__(in_channels, out_channels, filter_size, stride=1, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.subsampling = subsampling
        self.filter_size = filter_size
        self.patch_dim = self.in_channels * self.filter_size
        self._need_lintrans_computed = True
        self.kernel_args_trainable = kernel_args_trainable
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == 'exp' or kernel_func == 'add_exp':
            kernel_args = [(1.0 / kernel_arg ** 2) for kernel_arg in kernel_args]
        self.kernel_args = kernel_args
        if kernel_args_trainable:
            self.kernel_args = nn.ParameterList([nn.Parameter(torch.Tensor([kernel_arg])) for kernel_arg in kernel_args])
        kernel_func = kernels[kernel_func]
        self.kappa = lambda x: kernel_func(x, *self.kernel_args)
        ones = torch.ones(1, self.in_channels // self.groups, self.filter_size)
        self.register_buffer('ones', ones)
        self.init_pooling_filter()
        self.register_buffer('lintrans', torch.Tensor(out_channels, out_channels))

    def init_pooling_filter(self):
        if self.subsampling <= 1:
            return
        size = 2 * self.subsampling - 1
        pooling_filter = gaussian_filter_1d(size)
        pooling_filter = pooling_filter.expand(self.out_channels, 1, size)
        self.register_buffer('pooling_filter', pooling_filter)

    def train(self, mode=True):
        super(CKNLayer, self).train(mode)
        if self.training is True:
            self._need_lintrans_computed = True

    def _compute_lintrans(self):
        """Compute the linear transformation factor kappa(ZtZ)^(-1/2)
        Returns:
            lintrans: out_channels x out_channels
        """
        if not self._need_lintrans_computed:
            return self.lintrans
        lintrans = self.weight.view(self.out_channels, -1)
        lintrans = lintrans.mm(lintrans.t())
        lintrans = self.kappa(lintrans)
        lintrans = ops.matrix_inverse_sqrt(lintrans)
        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data = lintrans.data
        return lintrans

    def _conv_layer(self, x_in):
        """Convolution layer
        Compute x_out = ||x_in|| x kappa(Zt x_in/||x_in||)
        Args:
            x_in: batch_size x in_channels x H
            self.filters: out_channels x in_channels x filter_size
            x_out: batch_size x out_channels x (H - filter_size + 1)
        """
        patch_norm = torch.sqrt(F.conv1d(x_in.pow(2), self.ones, padding=self.padding, dilation=self.dilation, groups=self.groups).clamp(min=EPS))
        x_out = super(CKNLayer, self).forward(x_in)
        x_out = x_out / patch_norm.clamp(min=EPS)
        x_out = self.kappa(x_out)
        x_out = patch_norm * x_out
        return x_out

    def _mult_layer(self, x_in, lintrans):
        """Multiplication layer
        Compute x_out = kappa(ZtZ)^(-1/2) x x_in
        Args:
            x_in: batch_size x out_channels x H
            lintrans: out_channels x out_channels
            x_out: batch_size x out_channels x H
        """
        batch_size, out_c, _ = x_in.size()
        if x_in.dim() == 2:
            return torch.mm(x_in, lintrans)
        return torch.bmm(lintrans.expand(batch_size, out_c, out_c), x_in)

    def _pool_layer(self, x_in):
        """Pooling layer
        Compute I(z) = \\sum_{z'} phi(z') x exp(-eta_1 ||z'-z||_2^2)
        """
        if self.subsampling <= 1:
            return x_in
        x_out = F.conv1d(x_in, self.pooling_filter, stride=self.subsampling, padding=self.subsampling - 1, groups=self.out_channels)
        return x_out

    def forward(self, x_in):
        """Encode function for a CKN layer
        Args:
            x_in: batch_size x in_channels x H x W
        """
        x_out = self._conv_layer(x_in)
        x_out = self._pool_layer(x_out)
        lintrans = self._compute_lintrans()
        x_out = self._mult_layer(x_out, lintrans)
        return x_out

    def compute_mask(self, mask=None):
        if mask is None:
            return mask
        mask = mask.float().unsqueeze(1)
        mask = F.avg_pool1d(mask, kernel_size=self.filter_size, stride=self.subsampling)
        mask = mask.squeeze(1) != 0
        return mask

    def extract_1d_patches(self, input, mask=None):
        output = input.unfold(-1, self.filter_size, 1).transpose(1, 2)
        output = output.contiguous().view(-1, self.patch_dim)
        if mask is not None:
            mask = mask.float().unsqueeze(1)
            mask = F.avg_pool1d(mask, kernel_size=self.filter_size, stride=1)
            mask = mask.view(-1) != 0
            output = output[mask]
        return output

    def sample_patches(self, x_in, mask=None, n_sampling_patches=1000):
        """Sample patches from the given Tensor
        Args:
            x_in (Tensor batch_size x in_channels x H)
            n_sampling_patches (int): number of patches to sample
        Returns:
            patches: (batch_size x (H - filter_size + 1)) x (in_channels x filter_size)
        """
        patches = self.extract_1d_patches(x_in, mask)
        n_sampling_patches = min(patches.size(0), n_sampling_patches)
        indices = torch.randperm(patches.size(0))[:n_sampling_patches]
        patches = patches[indices]
        normalize_(patches)
        return patches

    def unsup_train(self, patches, init=None):
        """Unsupervised training for a CKN layer
        Args:
            patches: n x in_channels x H
        Updates:
            filters: out_channels x in_channels x filter_size
        """
        None
        weight = spherical_kmeans(patches, self.out_channels, init=init)
        weight = weight.view_as(self.weight)
        self.weight.data = weight.data
        self._need_lintrans_computed = True

    def normalize_(self):
        norm = self.weight.data.view(self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1)
        norm.clamp_(min=EPS)
        self.weight.data.div_(norm)


def flip(x, dim=-1):
    """Reverse a tensor along given axis
    can be removed later when Pytorch updated
    """
    reverse_indices = torch.arange(x.size(dim) - 1, -1, -1)
    reverse_indices = reverse_indices.type_as(x.data).long()
    return x.index_select(dim=dim, index=reverse_indices)


class BioEmbedding(nn.Module):

    def __init__(self, num_embeddings, reverse_complement=False, mask_zeros=False, no_embed=False, encoding='one_hot'):
        """Embedding layer for biosequences
        Args:
            num_embeddings (int): number of letters in alphabet
            reverse_complement (boolean): reverse complement embedding or not
        """
        super(BioEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.reverse_complement = reverse_complement
        self.mask_zeros = mask_zeros
        self.no_embed = no_embed
        if no_embed:
            return
        self.embedding = lambda x, weight: F.embedding(x, weight)
        if encoding == 'blosum62':
            weight = torch.from_numpy(BLOSUM62.astype(np.float32))
        else:
            weight = self._make_weight(False)
        self.register_buffer('weight', weight)
        if reverse_complement:
            weight_rc = self._make_weight(True)
            self.register_buffer('weight_rc', weight_rc)

    def _make_weight(self, reverse_complement=False):
        if reverse_complement:
            weight = np.zeros((self.num_embeddings + 1, self.num_embeddings), dtype=np.float32)
            weight[0] = 1.0 / self.num_embeddings
            weight[1:] = np.fliplr(np.diag(np.ones(self.num_embeddings)))
            weight = torch.from_numpy(weight)
        else:
            weight = torch.zeros(self.num_embeddings + 1, self.num_embeddings)
            weight[0] = 1.0 / self.num_embeddings
            weight[1:] = torch.diag(torch.ones(self.num_embeddings))
        return weight

    def compute_mask(self, x):
        """Compute the mask for the given Tensor
        """
        if self.no_embed:
            if self.mask_zeros:
                s = x.norm(dim=1)
                mask = s != 0
            else:
                mask = None
            return mask
        if self.mask_zeros:
            mask = x != 0
            if self.reverse_complement:
                mask_rc = flip(mask, dim=-1)
                mask = torch.cat((mask, mask_rc))
            return mask
        return None

    def forward(self, x):
        """
        Args:
            x: LongTensor of indices
        """
        if self.no_embed:
            return x
        x_out = self.embedding(x, self.weight)
        if self.reverse_complement:
            x = flip(x, dim=-1)
            x_out_rc = self.embedding(x, self.weight_rc)
            x_out = torch.cat((x_out, x_out_rc), dim=0)
        return x_out.transpose(1, 2).contiguous()


class GlobalAvg1D(nn.Module):

    def __init__(self):
        super(GlobalAvg1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=-1)
        mask = mask.float().unsqueeze(1)
        x = x * mask
        return x.sum(dim=-1) / mask.sum(dim=-1)


class GlobalMax1D(nn.Module):

    def __init__(self):
        super(GlobalMax1D, self).__init__()

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(x)
            mask = mask.data
            x[~mask] = -float('inf')
        return x.max(dim=-1)[0]


class GMP(nn.Module):

    def __init__(self, alpha=0.001):
        super(GMP, self).__init__()
        self.alpha = alpha

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.float().unsqueeze(1)
            x = x * mask
        xxt = torch.bmm(x, x.transpose(1, 2))
        xxt.diagonal(dim1=1, dim2=2)[:] += self.alpha
        eye = xxt.new_ones(xxt.size(-1)).diag().expand_as(xxt)
        xxt, _ = torch.gesv(eye, xxt)
        x = torch.bmm(xxt, x)
        return x.mean(dim=-1)


class Preprocessor(nn.Module):

    def __init__(self):
        super(Preprocessor, self).__init__()
        self.fitted = True

    def forward(self, input):
        out = input - input.mean(dim=1, keepdim=True)
        return out / out.norm(dim=1, keepdim=True)

    def fit(self, input):
        pass

    def fit_transform(self, input):
        self.fit(input)
        return self(input)


class RowPreprocessor(nn.Module):

    def __init__(self):
        super(RowPreprocessor, self).__init__()
        self.register_buffer('mean', None)
        self.register_buffer('var', None)
        self.register_buffer('scale', None)
        self.count = 0
        self.fitted = False

    def reset(self):
        self.mean = None
        self.var = None
        self.scale = None
        self.count = 0.0
        self.fitted = False

    def forward(self, input):
        if not self.fitted:
            return input
        input -= self.mean
        input /= self.scale
        return input

    def fit(self, input):
        self.mean = input.mean(dim=0)
        self.var = input.var(dim=0, unbiased=False)
        self.scale = self.var.sqrt()

    def fit_transform(self, input):
        self.fit(input)
        return self(input)

    def partial_fit(self, input):
        if self.count == 0.0:
            self.mean = input.mean(0)
            self.var = input.var(0, unbiased=False)
            self.scale = self.var.sqrt()
            self.count += input.shape[0]
        else:
            last_sum = self.count * self.mean
            new_sum = input.sum(0)
            updated_count = self.count + input.shape[0]
            self.mean = (last_sum + new_sum) / updated_count
            new_unnorm_var = input.var(0, unbiased=False) * input.shape[0]
            last_unnorm_var = self.var * self.count
            last_over_new_count = self.count / input.shape[0]
            self.var = (new_unnorm_var + last_unnorm_var + last_over_new_count / updated_count * (last_sum / last_over_new_count - new_sum) ** 2) / updated_count
            self.count = updated_count
            self.scale = self.var.sqrt()

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.fitted = True
        for k, v in self._buffers.items():
            key = prefix + k
            setattr(self, k, state_dict[key])
        super(RowPreprocessor, self)._load_from_state_dict(state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)


class CKNSequential(nn.Module):

    def __init__(self, in_channels, out_channels_list, filter_sizes, subsamplings, kernel_funcs=None, kernel_args_list=None, kernel_args_trainable=False, **kwargs):
        assert len(out_channels_list) == len(filter_sizes) == len(subsamplings), 'incompatible dimensions'
        super(CKNSequential, self).__init__()
        self.n_layers = len(out_channels_list)
        self.in_channels = in_channels
        self.out_channels = out_channels_list[-1]
        self.filter_sizes = filter_sizes
        self.subsamplings = subsamplings
        ckn_layers = []
        for i in range(self.n_layers):
            if kernel_funcs is None:
                kernel_func = 'exp'
            else:
                kernel_func = kernel_funcs[i]
            if kernel_args_list is None:
                kernel_args = 0.3
            else:
                kernel_args = kernel_args_list[i]
            ckn_layer = CKNLayer(in_channels, out_channels_list[i], filter_sizes[i], subsampling=subsamplings[i], kernel_func=kernel_func, kernel_args=kernel_args, kernel_args_trainable=kernel_args_trainable, **kwargs)
            ckn_layers.append(ckn_layer)
            in_channels = out_channels_list[i]
        self.ckn_layers = nn.Sequential(*ckn_layers)

    def __getitem__(self, idx):
        return self.ckn_layers[idx]

    def __len__(self):
        return len(self.ckn_layers)

    def __iter__(self):
        return iter(self.ckn_layers._modules.values())

    def forward_at(self, x, i=0):
        assert x.size(1) == self.ckn_layers[i].in_channels, 'bad dimension'
        return self.ckn_layers[i](x)

    def forward(self, x):
        return self.ckn_layers(x)

    def representation(self, x, n=0):
        if n == -1:
            n = self.n_layers
        for i in range(n):
            x = self.forward_at(x, i)
        return x

    def compute_mask(self, mask=None, n=-1):
        if mask is None:
            return mask
        if n > self.n_layers:
            raise ValueError('Index larger than number of layers')
        if n == -1:
            n = self.n_layers
        for i in range(n):
            mask = self.ckn_layers[i].compute_mask(mask)
        return mask

    def normalize_(self):
        for module in self.ckn_layers:
            module.normalize_()

    @property
    def len_motif(self):
        l = self.filter_sizes[self.n_layers - 1]
        for i in reversed(range(1, self.n_layers)):
            l = self.subsamplings[i - 1] * l + self.filter_sizes[i - 1] - 2
        return l


POOLINGS = {'mean': GlobalAvg1D, 'max': GlobalMax1D, 'gmp': GMP}


class CKN(nn.Module):

    def __init__(self, in_channels, out_channels_list, filter_sizes, subsamplings, kernel_funcs=None, kernel_args_list=None, kernel_args_trainable=False, alpha=0.0, fit_bias=True, reverse_complement=False, global_pool='mean', penalty='l2', scaler='standard_row', no_embed=False, encoding='one_hot', global_pool_arg=0.001, n_class=1, mask_zeros=True, **kwargs):
        super(CKN, self).__init__()
        self.reverse_complement = reverse_complement
        self.embed_layer = BioEmbedding(in_channels, reverse_complement, mask_zeros=mask_zeros, no_embed=no_embed, encoding=encoding)
        self.ckn_model = CKNSequential(in_channels, out_channels_list, filter_sizes, subsamplings, kernel_funcs, kernel_args_list, kernel_args_trainable, **kwargs)
        self.global_pool = POOLINGS[global_pool]()
        self.global_pool.alpha = global_pool_arg
        self.out_features = out_channels_list[-1]
        self.n_class = n_class
        self.initialize_scaler(scaler)
        self.classifier = LinearMax(self.out_features, n_class, alpha=alpha, fit_bias=fit_bias, reverse_complement=reverse_complement, penalty=penalty)

    def initialize_scaler(self, scaler=None):
        pass

    def normalize_(self):
        self.ckn_model.normalize_()

    def representation_at(self, input, n=0):
        output = self.embed_layer(input)
        mask = self.embed_layer.compute_mask(input)
        output = self.ckn_model.representation(output, n)
        mask = self.ckn_model.compute_mask(mask, n)
        return output, mask

    def representation(self, input):
        output = self.embed_layer(input)
        mask = self.embed_layer.compute_mask(input)
        output = self.ckn_model(output)
        mask = self.ckn_model.compute_mask(mask)
        output = self.global_pool(output, mask)
        return output

    def forward(self, input, proba=False):
        output = self.representation(input)
        return self.classifier(output, proba)

    def unsup_train_ckn(self, data_loader, n_sampling_patches=100000, init=None, use_cuda=False, n_patches_per_batch=None):
        self.train(False)
        if use_cuda:
            self
        for i, ckn_layer in enumerate(self.ckn_model):
            None
            n_patches = 0
            if n_patches_per_batch is None:
                try:
                    n_patches_per_batch = (n_sampling_patches + len(data_loader) - 1) // len(data_loader)
                except:
                    n_patches_per_batch = 1000
            patches = torch.Tensor(n_sampling_patches, ckn_layer.patch_dim)
            if use_cuda:
                patches = patches
            for data, _ in data_loader:
                if n_patches >= n_sampling_patches:
                    break
                if use_cuda:
                    data = data
                with torch.no_grad():
                    data, mask = self.representation_at(data, i)
                    data_patches = ckn_layer.sample_patches(data, mask, n_patches_per_batch)
                size = data_patches.size(0)
                if n_patches + size > n_sampling_patches:
                    size = n_sampling_patches - n_patches
                    data_patches = data_patches[:size]
                patches[n_patches:n_patches + size] = data_patches
                n_patches += size
            None
            patches = patches[:n_patches]
            ckn_layer.unsup_train(patches, init=init)

    def unsup_train_classifier(self, data_loader, criterion=None, use_cuda=False):
        encoded_train, encoded_target = self.predict(data_loader, True, use_cuda=use_cuda)
        None
        if hasattr(self, 'scaler') and not self.scaler.fitted:
            self.scaler.fitted = True
            size = encoded_train.shape[0]
            encoded_train = self.scaler.fit_transform(encoded_train.view(-1, self.out_features)).view(size, -1)
        self.classifier.fit(encoded_train, encoded_target, criterion)

    def predict(self, data_loader, only_representation=False, proba=False, use_cuda=False):
        self.train(False)
        if use_cuda:
            self
        n_samples = len(data_loader.dataset)
        batch_start = 0
        for i, (data, target, *_) in enumerate(data_loader):
            batch_size = data.shape[0]
            if use_cuda:
                data = data
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data, proba).data.cpu()
            if self.reverse_complement:
                batch_out = torch.cat((batch_out[:batch_size], batch_out[batch_size:]), dim=-1)
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
                target_output = target.new_empty([n_samples] + list(target.shape[1:]))
            output[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = target
            batch_start += batch_size
        output.squeeze_(-1)
        return output, target_output

    def compute_motif(self, max_iter=2000):
        self.train(True)
        weights = self.classifier.weight.data.cpu().clone().numpy()
        weights = weights.ravel()
        indices = np.argsort(np.abs(weights))[::-1]
        pwm_all = []
        for index in indices:
            motif, loss = optimize_motif(index, self.ckn_model, max_iter)
            motif_norm = np.linalg.norm(motif)
            threshold = (1 - motif_norm * np.exp(-4.5)) ** 2
            if loss < threshold:
                None
                pwm_all.append(motif)
        pwm_all = np.asarray(pwm_all)
        return pwm_all


PREPROCESSORS = {'standard_col': Preprocessor, 'standard_row': RowPreprocessor}


class unsupCKN(CKN):

    def initialize_scaler(self, scaler=None):
        self.scaler = PREPROCESSORS[scaler]()

    def unsup_train(self, data_loader, n_sampling_patches=500000, use_cuda=False):
        self.train(False)
        None
        tic = timer()
        self.unsup_train_ckn(data_loader, n_sampling_patches, use_cuda=use_cuda)
        toc = timer()
        None
        None
        tic = timer()
        self.unsup_train_classifier(data_loader, use_cuda=use_cuda)
        toc = timer()
        None

    def unsup_cross_val(self, data_loader, pos_data_loader=None, n_sampling_patches=500000, alpha_grid=None, kfold=5, scoring='neg_log_loss', init_kmeans=None, balanced=False, use_cuda=False):
        self.train(False)
        if alpha_grid is None:
            alpha_grid = [1.0, 0.1, 0.01, 0.001]
        None
        tic = timer()
        if pos_data_loader is not None:
            self.unsup_train_ckn(pos_data_loader, n_sampling_patches, init=init_kmeans, use_cuda=use_cuda)
        else:
            self.unsup_train_ckn(data_loader, n_sampling_patches, init=init_kmeans, use_cuda=use_cuda)
        toc = timer()
        None
        None
        best_score = -float('inf')
        best_alpha = 0
        tic = timer()
        encoded_train, encoded_target = self.predict(data_loader, True, use_cuda=use_cuda)
        if not self.scaler.fitted:
            self.scaler.fitted = True
            size = encoded_train.shape[0]
            encoded_train = self.scaler.fit_transform(encoded_train.view(-1, self.out_features)).view(size, -1)
        if not balanced:
            clf = self.classifier
            if use_cuda:
                n_jobs = None
            else:
                n_jobs = -1
            for alpha in alpha_grid:
                None
                clf.alpha = alpha
                clf.reset_parameters()
                score = cross_val_score(clf, encoded_train.numpy(), encoded_target.numpy(), cv=kfold, scoring=scoring, n_jobs=n_jobs)
                score = score.mean()
                None
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            None
            clf.alpha = best_alpha
            clf.fit(encoded_train, encoded_target)
            toc = timer()
        else:
            for alpha in alpha_grid:
                None
                clf = LogisticRegression(C=1.0 / alpha, fit_intercept=False, class_weight='balanced', solver='liblinear')
                score = cross_val_score(clf, encoded_train.numpy(), encoded_target.numpy(), cv=kfold, scoring=scoring, n_jobs=-1)
                score = score.mean()
                None
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            None
            clf = LogisticRegression(C=1.0 / best_alpha, fit_intercept=False, class_weight='balanced', solver='liblinear')
            clf.fit(encoded_train.numpy(), encoded_target.numpy())
            toc = timer()
            self.classifier.weight.data.copy_(torch.from_numpy(clf.coef_.reshape(1, -1)))
            None

    def representation(self, input):
        output = super(unsupCKN, self).representation(input)
        return self.scaler(output)


class supCKN(CKN):

    def sup_train(self, train_loader, criterion, optimizer, lr_scheduler=None, init_train_loader=None, epochs=100, val_loader=None, n_sampling_patches=500000, unsup_init=None, use_cuda=False, early_stop=True):
        None
        tic = timer()
        if init_train_loader is not None:
            self.unsup_train_ckn(init_train_loader, n_sampling_patches, init=unsup_init, use_cuda=use_cuda)
        else:
            self.unsup_train_ckn(train_loader, n_sampling_patches, init=unsup_init, use_cuda=use_cuda)
        toc = timer()
        None
        phases = ['train']
        data_loader = {'train': train_loader}
        if val_loader is not None:
            phases.append('val')
            data_loader['val'] = val_loader
        epoch_loss = None
        best_loss = float('inf')
        best_acc = 0
        best_epoch = 0
        for epoch in range(epochs):
            None
            None
            self.train(False)
            self.unsup_train_classifier(data_loader['train'], criterion, use_cuda=use_cuda)
            for phase in phases:
                if phase == 'train':
                    if lr_scheduler is not None:
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)
                        else:
                            lr_scheduler.step()
                        None
                    self.train(True)
                else:
                    self.train(False)
                tic = timer()
                loader = data_loader[phase]
                if isinstance(loader, list):
                    epoch_loss = []
                    epoch_acc = []
                    for ids, train_l in loader:
                        e_loss, e_acc = self.one_step(phase, train_l, optimizer, criterion, use_cuda)
                        epoch_loss.append(e_loss)
                        epoch_acc.append(e_acc)
                    epoch_loss = np.mean(epoch_loss)
                    epoch_acc = np.mean(epoch_acc)
                else:
                    epoch_loss, epoch_acc = self.one_step(phase, loader, optimizer, criterion, use_cuda)
                toc = timer()
                None
                if phase == 'val' and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch + 1
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())
            None
        None
        None
        None
        if early_stop:
            self.load_state_dict(best_weights)
        return best_loss, best_acc, best_epoch

    def one_step(self, phase, train_loader, optimizer, criterion, use_cuda):
        running_loss = 0.0
        running_corrects = 0
        for data, target, *_ in train_loader:
            size = data.size(0)
            if self.n_class == 1:
                target = target.float()
            if use_cuda:
                data = data
                target = target
            if phase == 'val':
                with torch.no_grad():
                    output = self(data)
                    if self.n_class == 1:
                        output = output.view(-1)
                        pred = (output.data > 0).float()
                    else:
                        pred = output.data.argmax(dim=1)
                    loss = criterion(output, target)
            else:
                optimizer.zero_grad()
                output = self(data)
                if self.n_class == 1:
                    output = output.view(-1)
                    pred = (output.data > 0).float()
                else:
                    pred = output.data.argmax(dim=1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                self.normalize_()
            running_loss += loss.item() * size
            running_corrects += torch.sum(pred == target.data).item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        return epoch_loss, epoch_acc

    def hybrid_train(self, teacher_model, train_loader, criterion, optimizer, lr_scheduler=None, init_train_loader=None, epochs=100, val_loader=None, n_sampling_patches=500000, unsup_init=None, use_cuda=False, early_stop=True, regul=1.0):
        None
        tic = timer()
        if init_train_loader is not None:
            self.unsup_train_ckn(init_train_loader, n_sampling_patches, init=unsup_init, use_cuda=use_cuda)
        else:
            self.unsup_train_ckn(train_loader, n_sampling_patches, init=unsup_init, use_cuda=use_cuda)
        toc = timer()
        None
        phases = ['train']
        data_loader = {'train': train_loader}
        if val_loader is not None:
            phases.append('val')
            data_loader['val'] = val_loader
        epoch_loss = None
        best_loss = float('inf')
        best_acc = 0
        for epoch in range(epochs):
            None
            None
            self.train(False)
            self.hybrid_train_classifier(teacher_model, train_loader, criterion, use_cuda=use_cuda, regul=regul)
            for phase in phases:
                criterion.weight = None
                criterion.reduction = 'elementwise_mean'
                if phase == 'train':
                    if lr_scheduler is not None:
                        if isinstance(lr_scheduler, ReduceLROnPlateau):
                            if epoch_loss is not None:
                                lr_scheduler.step(epoch_loss)
                        else:
                            lr_scheduler.step()
                        None
                    self.train(True)
                else:
                    self.train(False)
                running_loss = 0.0
                running_corrects = 0
                for data, target, *mask in data_loader[phase]:
                    size = data.size(0)
                    target = target.float()
                    if use_cuda:
                        data = data
                        target = target
                    if len(mask) > 0:
                        mask = mask[0].view(-1)
                        nu = mask.sum().item()
                        nl = len(mask) - nu
                        weight = torch.ones(len(mask)) / (nl + 1)
                        weight[mask] = regul / (nu + 1)
                        if use_cuda:
                            weight = weight
                        criterion.weight = weight
                        criterion.reduction = 'sum'
                    optimizer.zero_grad()
                    if phase == 'val':
                        with torch.no_grad():
                            output = self(data).view(-1)
                            pred = (output > 0).float()
                            loss = criterion(output, target)
                    else:
                        output = self(data).view(-1)
                        pred = (output > 0).float()
                        with torch.no_grad():
                            teacher_pred = teacher_model(data, proba=True).view(-1)
                        target[mask] = (teacher_pred[mask] > 0.5).float()
                        loss = criterion(output, target)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        self.normalize_()
                    running_loss += loss.item() * size
                    running_corrects += torch.sum(pred == target.data).item()
                epoch_loss = running_loss / len(data_loader[phase].dataset)
                epoch_acc = running_corrects / len(data_loader[phase].dataset)
                None
                if phase == 'val' and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    if early_stop:
                        best_weights = copy.deepcopy(self.state_dict())
            None
        None
        None
        None
        if early_stop:
            self.load_state_dict(best_weights)
        return self

    def hybrid_train_classifier(self, teacher_model, data_loader, criterion=None, use_cuda=False, regul=1.0):
        encoded_train, encoded_target, mask = self.hybrid_predict(teacher_model, data_loader, True, use_cuda=use_cuda)
        nu = mask.sum().item()
        nl = len(mask) - nu
        weight = torch.ones(len(encoded_target))
        weight[mask] = regul * nl / (nu + 1)
        if use_cuda:
            weight = weight
        criterion.weight = weight
        self.classifier.fit(encoded_train, encoded_target, criterion)

    def hybrid_predict(self, teacher_model, data_loader, only_representation=False, proba=False, use_cuda=False):
        self.train(False)
        if use_cuda:
            self
        n_samples = len(data_loader.dataset)
        target_output = torch.Tensor(n_samples)
        mask_output = torch.ByteTensor(n_samples)
        batch_start = 0
        for i, (data, target, mask) in enumerate(data_loader):
            mask = mask.view(-1)
            batch_size = data.shape[0]
            if use_cuda:
                data = data
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data, proba).data.cpu()
                teacher_target = teacher_model(data, proba=True).data.cpu()
            teacher_target = (teacher_target > 0.5).float()
            teacher_target = teacher_target.view(-1)
            batch_out = torch.cat((batch_out[:batch_size], batch_out[batch_size:]), dim=-1)
            if i == 0:
                output = torch.Tensor(n_samples, batch_out.shape[-1])
            output[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = target
            target_output[batch_start:batch_start + batch_size][mask] = teacher_target[mask]
            mask_output[batch_start:batch_start + batch_size] = mask
            batch_start += batch_size
        output.squeeze_(-1)
        return output, target_output, mask_output


def log_sinkhorn(K, mask=None, eps=1.0, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    batch_size, in_size, out_size = K.shape

    def min_eps(u, v, dim):
        Z = (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps
        return -torch.logsumexp(Z, dim=dim)
    u = K.new_zeros((batch_size, in_size))
    v = K.new_zeros((batch_size, out_size))
    a = torch.ones_like(u).fill_(out_size / in_size)
    if mask is not None:
        a = out_size / mask.float().sum(1, keepdim=True)
    a = torch.log(a)
    for _ in range(max_iter):
        u = eps * (a + min_eps(u, v, dim=-1)) + u
        if mask is not None:
            u = u.masked_fill(~mask, -100000000.0)
        v = eps * min_eps(u, v, dim=1) + v
    if return_kernel:
        output = torch.exp((K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
        output = output / out_size
        return (output * K).sum(dim=[1, 2])
    K = torch.exp((K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
    return K


def sinkhorn(dot, mask=None, eps=0.001, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = dot.shape
    if return_kernel:
        K = torch.exp(dot / eps)
    else:
        K = dot
    u = K.new_ones((n, in_size))
    v = K.new_ones((n, out_size))
    a = float(out_size / in_size)
    if mask is not None:
        mask = mask.float()
        a = out_size / mask.sum(1, keepdim=True)
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
        if mask is not None:
            u = u * mask
        v = 1.0 / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
    if return_kernel:
        K = K / out_size
        return (K * dot).sum(dim=[1, 2])
    return K


def multihead_attn(input, weight, mask=None, eps=1.0, return_kernel=False, max_iter=100, log_domain=False, position_filter=None):
    """Comput the attention weight using Sinkhorn OT
    input: n x in_size x in_dim
    mask: n x in_size
    weight: m x out_size x in_dim (m: number of heads/ref)
    output: n x out_size x m x in_size
    """
    n, in_size, in_dim = input.shape
    m, out_size = weight.shape[:-1]
    K = torch.tensordot(input, weight, dims=[[-1], [-1]])
    K = K.permute(0, 2, 1, 3)
    if position_filter is not None:
        K = position_filter * K
    K = K.reshape(-1, in_size, out_size)
    if mask is not None:
        mask = mask.repeat_interleave(m, dim=0)
    if log_domain:
        K = log_sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    else:
        if not return_kernel:
            K = torch.exp(K / eps)
        K = sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    if return_kernel:
        return K.reshape(n, m)
    K = K.reshape(n, m, in_size, out_size)
    if position_filter is not None:
        K = position_filter * K
    K = K.permute(0, 3, 1, 2).contiguous()
    return K


def normalize(x, p=2, dim=-1, inplace=True):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    if inplace:
        x.div_(norm.clamp(min=EPS))
    else:
        x = x / norm.clamp(min=EPS)
    return x


def wasserstein_barycenter(x, c, eps=1.0, max_iter=100, sinkhorn_iter=50, log_domain=False):
    """
    x: n x in_size x in_dim
    c: out_size x in_dim
    """
    prev_c = c
    for i in range(max_iter):
        T = attn(x, c, eps=eps, log_domain=log_domain, max_iter=sinkhorn_iter)
        c = 0.5 * c + 0.5 * torch.bmm(T, x).mean(dim=0) / math.sqrt(c.shape[0])
        c /= c.norm(dim=-1, keepdim=True).clamp(min=1e-06)
        if ((c - prev_c) ** 2).sum() < 1e-06:
            break
        prev_c = c
    return c


def wasserstein_kmeans(x, n_clusters, out_size, eps=1.0, block_size=None, max_iter=100, sinkhorn_iter=50, wb=False, verbose=True, log_domain=False, use_cuda=False):
    """
    x: n x in_size x in_dim
    output: n_clusters x out_size x in_dim
    out_size <= in_size
    """
    n, in_size, in_dim = x.shape
    if n_clusters == 1:
        if use_cuda:
            x = x
        clusters = spherical_kmeans(x.view(-1, in_dim), out_size, block_size=block_size)
        if wb:
            clusters = wasserstein_barycenter(x, clusters, eps=0.1, log_domain=False)
        clusters = clusters.unsqueeze_(0)
        return clusters
    indices = torch.randperm(n)[:n_clusters]
    clusters = x[indices, :out_size, :].clone()
    if use_cuda:
        clusters = clusters
    wass_sim = x.new_empty(n)
    assign = x.new_empty(n, dtype=torch.long)
    if block_size is None or block_size == 0:
        block_size = n
    prev_sim = float('inf')
    for n_iter in range(max_iter):
        for i in range(0, n, block_size):
            end_i = min(i + block_size, n)
            x_batch = x[i:end_i]
            if use_cuda:
                x_batch = x_batch
            tmp_sim = multihead_attn(x_batch, clusters, eps=eps, return_kernel=True, max_iter=sinkhorn_iter, log_domain=log_domain)
            tmp_sim = tmp_sim.cpu()
            wass_sim[i:end_i], assign[i:end_i] = tmp_sim.max(dim=-1)
        del x_batch
        sim = wass_sim.mean()
        if verbose and (n_iter + 1) % 10 == 0:
            None
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                idx = wass_sim.argmin()
                clusters[j].copy_(x[idx, :out_size, :])
                wass_sim[idx] = 1
            else:
                xj = x[index]
                if use_cuda:
                    xj = xj
                c = spherical_kmeans(xj.view(-1, in_dim), out_size, block_size=block_size, verbose=False)
                if wb:
                    c = wasserstein_barycenter(xj, c, eps=0.001, log_domain=True, sinkhorn_iter=50)
                clusters[j] = c
        if torch.abs(prev_sim - sim) / sim.clamp(min=1e-10) < 1e-06:
            break
        prev_sim = sim
    return clusters


class OTKernel(nn.Module):

    def __init__(self, in_dim, out_size, heads=1, eps=0.1, max_iter=100, log_domain=False, position_encoding=None, position_sigma=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_size = out_size
        self.heads = heads
        self.eps = eps
        self.max_iter = max_iter
        self.weight = nn.Parameter(torch.Tensor(heads, out_size, in_dim))
        self.log_domain = log_domain
        self.position_encoding = position_encoding
        self.position_sigma = position_sigma
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1.0 / math.sqrt(self.out_size)
        for w in self.parameters():
            w.data.uniform_(-stdv, stdv)

    def get_position_filter(self, input, out_size):
        if input.ndim == 4:
            in_size1 = input.shape[1]
            in_size2 = input.shape[2]
            out_size = int(math.sqrt(out_size))
            if self.position_encoding is None:
                return self.position_encoding
            elif self.position_encoding == 'gaussian':
                sigma = self.position_sigma
                a1 = torch.arange(1.0, in_size1 + 1.0).view(-1, 1) / in_size1
                a2 = torch.arange(1.0, in_size2 + 1.0).view(-1, 1) / in_size2
                b = torch.arange(1.0, out_size + 1.0).view(1, -1) / out_size
                position_filter1 = torch.exp(-((a1 - b) / sigma) ** 2)
                position_filter2 = torch.exp(-((a2 - b) / sigma) ** 2)
                position_filter = position_filter1.view(in_size1, 1, out_size, 1) * position_filter2.view(1, in_size2, 1, out_size)
            if self.weight.is_cuda:
                position_filter = position_filter
            return position_filter.reshape(1, 1, in_size1 * in_size2, out_size * out_size)
        in_size = input.shape[1]
        if self.position_encoding is None:
            return self.position_encoding
        elif self.position_encoding == 'gaussian':
            sigma = self.position_sigma
            a = torch.arange(0.0, in_size).view(-1, 1) / in_size
            b = torch.arange(0.0, out_size).view(1, -1) / out_size
            position_filter = torch.exp(-((a - b) / sigma) ** 2)
        elif self.position_encoding == 'hard':
            sigma = self.position_sigma
            a = torch.arange(0.0, in_size).view(-1, 1) / in_size
            b = torch.arange(0.0, out_size).view(1, -1) / out_size
            position_filter = torch.abs(a - b) < sigma
            position_filter = position_filter.float()
        else:
            raise ValueError('Unrecognizied position encoding')
        if self.weight.is_cuda:
            position_filter = position_filter
        position_filter = position_filter.view(1, 1, in_size, out_size)
        return position_filter

    def get_attn(self, input, mask=None, position_filter=None):
        """Compute the attention weight using Sinkhorn OT
        input: batch_size x in_size x in_dim
        mask: batch_size x in_size
        self.weight: heads x out_size x in_dim
        output: batch_size x (out_size x heads) x in_size
        """
        return multihead_attn(input, self.weight, mask=mask, eps=self.eps, max_iter=self.max_iter, log_domain=self.log_domain, position_filter=position_filter)

    def forward(self, input, mask=None):
        """
        input: batch_size x in_size x in_dim
        output: batch_size x out_size x (heads x in_dim)
        """
        batch_size = input.shape[0]
        position_filter = self.get_position_filter(input, self.out_size)
        in_ndim = input.ndim
        if in_ndim == 4:
            input = input.view(batch_size, -1, self.in_dim)
        attn_weight = self.get_attn(input, mask, position_filter)
        output = torch.bmm(attn_weight.view(batch_size, self.out_size * self.heads, -1), input)
        if in_ndim == 4:
            out_size = int(math.sqrt(self.out_size))
            output = output.reshape(batch_size, out_size, out_size, -1)
        else:
            output = output.reshape(batch_size, self.out_size, -1)
        return output

    def unsup_train(self, input, wb=False, inplace=True, use_cuda=False):
        """K-meeans for learning parameters
        input: n_samples x in_size x in_dim
        weight: heads x out_size x in_dim
        """
        input_normalized = normalize(input, inplace=inplace)
        block_size = int(1000000000.0) // (input.shape[1] * input.shape[2] * 4)
        None
        weight = wasserstein_kmeans(input_normalized, self.heads, self.out_size, eps=self.eps, block_size=block_size, wb=wb, log_domain=self.log_domain, use_cuda=use_cuda)
        self.weight.data.copy_(weight)

    def random_sample(self, input):
        idx = torch.randint(0, input.shape[0], (1,))
        self.weight.data.copy_(input[idx].view_as(self.weight))


class Linear(nn.Linear):

    def forward(self, input):
        bias = self.bias
        if bias is not None and hasattr(self, 'scale_bias') and self.scale_bias is not None:
            bias = self.scale_bias * bias
        out = torch.nn.functional.linear(input, self.weight, bias)
        return out

    def fit(self, Xtr, ytr, criterion, reg=0.0, epochs=100, optimizer=None, use_cuda=False):
        if optimizer is None:
            optimizer = optim.LBFGS(self.parameters(), lr=1.0, history_size=10)
        if self.bias is not None:
            scale_bias = (Xtr ** 2).mean(-1).sqrt().mean().item()
            self.scale_bias = scale_bias
        self.train()
        if use_cuda:
            self
            Xtr = Xtr
            ytr = ytr

        def closure():
            optimizer.zero_grad()
            output = self(Xtr)
            loss = criterion(output, ytr)
            loss = loss + 0.5 * reg * self.weight.pow(2).sum()
            loss.backward()
            return loss
        for epoch in range(epochs):
            optimizer.step(closure)
        if self.bias is not None:
            self.bias.data.mul_(self.scale_bias)
        self.scale_bias = None

    def score(self, X, y):
        self.eval()
        with torch.no_grad():
            scores = self(X)
            scores = scores.argmax(-1)
            scores = scores.cpu()
        return torch.mean((scores == y).float()).item()


class OTLayer(nn.Module):

    def __init__(self, in_dim, out_size, heads=1, eps=0.1, max_iter=10, position_encoding=None, position_sigma=0.1, out_dim=None, dropout=0.4):
        super().__init__()
        self.out_size = out_size
        self.heads = heads
        if out_dim is None:
            out_dim = in_dim
        self.layer = nn.Sequential(OTKernel(in_dim, out_size, heads, eps, max_iter, log_domain=True, position_encoding=position_encoding, position_sigma=position_sigma), nn.Linear(heads * in_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        nn.init.xavier_uniform_(self.layer[0].weight)
        nn.init.xavier_uniform_(self.layer[1].weight)

    def forward(self, input):
        output = self.layer(input)
        return output


class SeqAttention(nn.Module):

    def __init__(self, nclass, hidden_size, filter_size, n_attn_layers, eps=0.1, heads=1, out_size=1, max_iter=10, hidden_layer=False, position_encoding=None, position_sigma=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv1d(4, hidden_size, kernel_size=filter_size), nn.ReLU(inplace=True))
        attn_layers = [OTLayer(hidden_size, out_size, heads, eps, max_iter, position_encoding, position_sigma=position_sigma)] + [OTLayer(hidden_size, out_size, heads, eps, max_iter, position_encoding, position_sigma=position_sigma) for _ in range(n_attn_layers - 1)]
        self.attn_layers = nn.Sequential(*attn_layers)
        self.out_features = out_size * hidden_size
        self.nclass = nclass
        if hidden_layer:
            self.classifier = nn.Sequential(nn.Linear(self.out_features, nclass), nn.ReLU(inplace=True), nn.Linear(nclass, nclass))
        else:
            self.classifier = nn.Linear(self.out_features, nclass)

    def representation(self, input):
        output = self.embed(input).transpose(1, 2).contiguous()
        output = self.attn_layers(output)
        output = output.reshape(output.shape[0], -1)
        return output

    def forward(self, input):
        output = self.representation(input)
        return self.classifier(output)

    def predict(self, data_loader, only_repr=False, use_cuda=False):
        n_samples = len(data_loader.dataset)
        target_output = torch.LongTensor(n_samples)
        batch_start = 0
        for i, (data, target) in enumerate(data_loader):
            batch_size = data.shape[0]
            if use_cuda:
                data = data
            with torch.no_grad():
                if only_repr:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data).data.cpu()
            if i == 0:
                output = batch_out.new_empty([n_samples] + list(batch_out.shape[1:]))
            output[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = target
            batch_start += batch_size
        return output, target_output


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (GlobalAvg1D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GlobalMax1D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OTKernel,
     lambda: ([], {'in_dim': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (OTLayer,
     lambda: ([], {'in_dim': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Preprocessor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RowPreprocessor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SeqAttention,
     lambda: ([], {'nclass': 4, 'hidden_size': 4, 'filter_size': 4, 'n_attn_layers': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

