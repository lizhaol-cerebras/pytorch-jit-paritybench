
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


import itertools


from torch.cuda.amp import autocast


import numpy as np


import time


import warnings


from torch import nn


import copy


import logging


from typing import List


from typing import Union


import torch.utils.data as torchdata


from typing import Iterable


from typing import Mapping


from torch.nn.parallel import DataParallel


from torch.nn.parallel import DistributedDataParallel


from collections import abc


from typing import Optional


import math


from collections import OrderedDict


from collections import defaultdict


from typing import Tuple


import torch.utils.checkpoint as checkpoint


import torchvision.transforms as T


from torch.nn import functional as F


import enum


import torch as th


from abc import ABC


from abc import abstractmethod


import torch.distributed as dist


from collections import namedtuple


import torch.nn as nn


import torch.nn.functional as F


from abc import ABCMeta


from typing import Any


import collections.abc


import torchvision


from typing import Dict


import typing


from collections import deque


import matplotlib.colors as mplc


from scipy.optimize import linear_sum_assignment


from copy import deepcopy


from typing import Callable


from torch.nn.init import xavier_uniform_


from torch.nn.init import constant_


from torch.nn.init import uniform_


from torch.nn.init import normal_


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.autograd import gradcheck


from torch import Tensor


from itertools import count


import torch.utils.data


import random


from functools import wraps


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


from collections import Counter


from typing import Set


EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])


EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings', 'text_mask'])


logger = logging.getLogger('detectron2')


def build_clip_text_embed(clip_model_name, labels, device='cuda', verbose=True):
    if isinstance(clip_model_name, str):
        clip, _, _ = open_clip.create_model_and_transforms(model_name=clip_model_name, pretrained='openai', device=device if torch.cuda.is_available() else 'cpu')
        if verbose:
            logger.info(f'Loading CLIP model {clip_model_name}')
    else:
        clip = clip_model_name
        if verbose:
            logger.info('Using provided CLIP model')
    clip_device = next(clip.parameters()).device
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(labels[0], str):
        labels = [[t] for t in labels]
    labels = tuple(tuple(t) for t in labels)
    assert isinstance(labels[0], (list, tuple)), f'labels should be a list of list of str, but got {type(labels[0])}'
    flatten_text = [t for sublist in labels for t in sublist]
    text_embed_list = []
    local_batch_size = 256
    for i in range(0, len(flatten_text), local_batch_size):
        cur_text = flatten_text[i:i + local_batch_size]
        text_embed = clip.encode_text(open_clip.tokenize(cur_text))
        text_embed_list.extend(list(text_embed))
    out_text_embed = torch.stack(text_embed_list)
    if verbose:
        logger.info(f'Built text_embed of shape {out_text_embed.shape} for {len(labels)} labels: {labels}')
    return out_text_embed


class ClipAdapter(nn.Module):

    def __init__(self, name='ViT-B-32', normalize=True):
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(name, pretrained='openai')
        comm.synchronize()
        openai_clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained='openai')
        super().__init__()
        self.clip = openai_clip
        self.clip_preprocess = T.Compose([*preprocess.transforms[:2], preprocess.transforms[-1]])
        self._freeze()
        self.name = name
        self.normalize = normalize

    def extra_repr(self) ->str:
        return f'name={self.name}, normalize={self.normalize}'

    def _freeze(self):
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def ignored_state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, 'ignored_state_dict'):
                module.ignored_state_dict(destination, prefix + name + '.')
        return super().state_dict(destination=destination, prefix=prefix)

    @property
    def device(self):
        return next(self.parameters()).device

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return OrderedDict()

    def train(self, mode: 'bool'=True):
        super().train(mode)
        self._freeze()
        return self

    @property
    def dim_latent(self):
        return self.clip.text_projection.shape[-1]

    @property
    def image_size(self):
        if isinstance(self.clip.visual.image_size, tuple):
            return self.clip.visual.image_size
        else:
            return self.clip.visual.image_size, self.clip.visual.image_size

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    def _encode_text(self, text):
        x = self.clip.token_embedding(text)
        x = x + self.clip.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x)
        text_encodings = x
        text_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection
        return text_embed, text_encodings

    @torch.no_grad()
    def embed_text(self, captions):
        text = open_clip.tokenize(captions)
        text = text[..., :self.max_text_len]
        text_mask = (text != 0).long()
        text_embed, text_encodings = self._encode_text(text)
        if self.normalize:
            return EmbeddedText(F.normalize(text_embed.float(), dim=-1), text_encodings.float(), text_mask)
        else:
            return EmbeddedText(text_embed.float(), text_encodings.float(), text_mask)

    def _encode_image(self, image):
        if hasattr(self.clip.visual, 'positional_embedding'):
            x = self.clip.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([self.clip.visual.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.clip.visual.positional_embedding
            x = self.clip.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.clip.visual.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip.visual.ln_post(x)
            batch_size, num_tokens, _ = x.shape
            if self.clip.visual.proj is not None:
                x = rearrange(x, 'b n c -> (b n) c', b=batch_size, n=num_tokens)
                x = x @ self.clip.visual.proj
                x = rearrange(x, '(b n) c -> b n c', b=batch_size, n=num_tokens)
            image_embed = x[:, 0, :]
            image_encodings = x[:, 1:, :]
            width = height = int(image_encodings.shape[1] ** 0.5)
            image_encodings = rearrange(image_encodings, 'b (h w) c -> b c h w', h=height, w=width)
            image_encodings = F.interpolate(image_encodings, size=(image.shape[2] // 16, image.shape[3] // 16), mode='bilinear', align_corners=False)
            return image_embed, image_encodings
        else:
            image_embed = self.clip.encode_image(image)
            return image_embed, None

    @torch.no_grad()
    def embed_image(self, image):
        image_embed, image_encodings = self._encode_image(self.clip_preprocess(image))
        if self.normalize:
            return EmbeddedImage(F.normalize(image_embed.float(), dim=-1), image_encodings)
        else:
            return EmbeddedImage(image_embed.float(), image_encodings)

    @torch.no_grad()
    def build_text_embed(self, labels):
        return build_clip_text_embed(self.clip, labels)


def ensemble_logits_with_labels(logits: 'torch.Tensor', labels: 'List[List[str]]', ensemble_method: 'str'='max'):
    """Ensemble logits.
    Args:
        logits (torch.Tensor): logits of each model. The last dim is probability.
        labels (list[list[str]]): list of list of labels.
        ensemble_method (str): ensemble method. Options are 'mean' and 'max'.
    Returns:
        torch.Tensor: logits of ensemble model.
    """
    len_list = [len(l) for l in labels]
    assert logits.shape[-1] == sum(len_list), f'{logits.shape[-1]} != {sum(len_list)}'
    assert ensemble_method in ['mean', 'max']
    ensemble_logits = torch.zeros(*logits.shape[:-1], len(labels), dtype=logits.dtype, device=logits.device)
    if ensemble_method == 'max':
        for i in range(len(labels)):
            ensemble_logits[..., i] = logits[..., sum(len_list[:i]):sum(len_list[:i + 1])].max(dim=-1).values
    elif ensemble_method == 'mean':
        for i in range(len(labels)):
            ensemble_logits[..., i] = logits[..., sum(len_list[:i]):sum(len_list[:i + 1])].mean(dim=-1)
    else:
        raise ValueError(f'Unknown ensemble method: {ensemble_method}')
    return ensemble_logits


class MaskCLIP(ClipAdapter):
    """
    Ref: https://arxiv.org/abs/2208.08984
    """

    def __init__(self, name='ViT-L-14-336'):
        super().__init__(name=name, normalize=False)

    @property
    def logit_scale(self):
        logit_scale = torch.clamp(self.clip.logit_scale.exp(), max=100)
        return logit_scale

    def _mask_clip_forward(self, x: 'torch.Tensor', attn_mask: 'torch.Tensor', num_mask_tokens: 'int'):
        x = self.clip.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.clip.visual.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.clip.visual.positional_embedding
        x = self.clip.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        cls_embed = x[0:1]
        cls_embed = cls_embed.expand(num_mask_tokens, -1, -1)
        x = torch.cat([cls_embed, x], dim=0)
        x = self.clip.visual.transformer(x, attn_mask)
        x = x.permute(1, 0, 2)
        x = self.clip.visual.ln_post(x[:, :num_mask_tokens, :])
        if self.clip.visual.proj is not None:
            x = torch.einsum('nld,dc->nlc', x, self.clip.visual.proj)
        return x

    def encode_image_with_mask(self, image, mask):
        assert hasattr(self.clip.visual, 'positional_embedding')
        image = self.clip_preprocess(image)
        batch_size = image.shape[0]
        assert batch_size == mask.shape[0]
        num_queries = mask.shape[1]
        mask = mask.sigmoid()
        patch_mask = F.max_pool2d(mask, kernel_size=self.clip.visual.conv1.kernel_size, stride=self.clip.visual.conv1.stride)
        mask_token_attn_mask = patch_mask < 0.5
        mask_token_attn_mask = mask_token_attn_mask.reshape(batch_size, num_queries, -1)
        num_mask_token = num_queries
        num_image_cls_token = self.clip.visual.positional_embedding.shape[0]
        num_image_token = num_image_cls_token - 1
        num_all_token = num_mask_token + num_image_cls_token
        attn_mask = torch.zeros((num_all_token, num_all_token), dtype=torch.bool, device=image.device)
        attn_mask[:, :num_mask_token] = True
        attn_mask = attn_mask.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        attn_mask[:, :num_mask_token, -num_image_token:] = mask_token_attn_mask
        num_heads = self.clip.visual.conv1.out_channels // 64
        attn_mask = attn_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch_size * num_heads, num_all_token, num_all_token)
        return self._mask_clip_forward(image, attn_mask, num_mask_token)

    def get_mask_embed(self, image, mask):
        image = F.interpolate(image, size=self.image_size, mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, size=image.shape[-2:], mode='bilinear', align_corners=False)
        mask_embed = self.encode_image_with_mask(image, mask)
        return mask_embed

    def pred_logits(self, mask_embed, text_embed, labels):
        logit_per_mask = torch.einsum('bqc,nc->bqn', F.normalize(mask_embed, dim=-1), F.normalize(text_embed, dim=-1)) * self.logit_scale
        logit_per_mask = ensemble_logits_with_labels(logit_per_mask, labels)
        return logit_per_mask

    def forward(self, image, mask, text_embed, labels):
        mask_embed = self.get_mask_embed(image, mask)
        output = {'mask_embed': mask_embed}
        if text_embed is not None and labels is not None:
            output['mask_pred_open_logits'] = self.pred_logits(mask_embed, text_embed, labels)
        return output


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_channels, eps=1e-06, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class FeatureExtractor(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    def ignored_state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for name, module in self._modules.items():
            if module is not None and hasattr(module, 'ignored_state_dict'):
                module.ignored_state_dict(destination, prefix + name + '.')
        return super().state_dict(destination=destination, prefix=prefix)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return OrderedDict()

    def train(self, mode: 'bool'=True):
        super().train(mode)
        self._freeze()
        return self

    def _freeze(self):
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @property
    @abstractmethod
    def feature_dims(self) ->List[int]:
        pass

    @property
    @abstractmethod
    def feature_size(self) ->int:
        pass

    @property
    @abstractmethod
    def num_groups(self) ->int:
        pass

    @property
    @abstractmethod
    def grouped_indices(self, features):
        pass


class DisableLogger:
    """Disable HF logger"""

    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


def add_device_property(cls):


    class TempClass(cls):
        pass
    TempClass.device = property(lambda m: next(m.parameters()).device)
    return TempClass


def batched_input_to_device(batched_inputs, device, exclude=()):
    if isinstance(exclude, str):
        exclude = [exclude]
    if isinstance(batched_inputs, torch.Tensor):
        batch = batched_inputs
        return batch
    elif isinstance(batched_inputs, collections.abc.Mapping):
        batch = {}
        for k in batched_inputs:
            if k not in exclude:
                batched_inputs[k] = batched_input_to_device(batched_inputs[k], device)
        return batched_inputs
    elif isinstance(batched_inputs, collections.abc.Sequence) and not isinstance(batched_inputs, str):
        return [batched_input_to_device(d, device) for d in batched_inputs]
    elif isinstance(batched_inputs, str):
        return batched_inputs
    else:
        raise TypeError(f'Unsupported type {type(batched_inputs)}')


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(x < -0.999, log_cdf_plus, th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, 'at least one argument must be a Tensor'
    logvar1, logvar2 = [(x if isinstance(x, th.Tensor) else th.tensor(x)) for x in (logvar1, logvar2)]
    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * th.exp(-logvar2))


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type, rescale_timesteps=False):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, 'betas must be 1-D'
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {ModelVarType.FIXED_LARGE: (np.append(self.posterior_variance[1], self.betas[1:]), np.log(np.append(self.posterior_variance[1], self.betas[1:]))), ModelVarType.FIXED_SMALL: (self.posterior_variance, self.posterior_log_variance_clipped)}[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)
        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {'mean': model_mean, 'variance': model_variance, 'log_variance': model_log_variance, 'pred_xstart': pred_xstart}

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev - _extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = p_mean_var['mean'].float() + p_mean_var['variance'] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        out = p_mean_var.copy()
        out['pred_xstart'] = self._predict_xstart_from_eps(x, t, eps)
        out['mean'], _, _ = self.q_posterior_mean_variance(x_start=out['pred_xstart'], x_t=x, t=t)
        return out

    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        if cond_fn is not None:
            out['mean'] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out['mean'] + nonzero_mask * th.exp(0.5 * out['log_variance']) * noise
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress):
            final = sample
        return final['sample']

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs)
                yield out
                img = out['sample']

    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        eps = self._predict_eps_from_xstart(x, t, out['pred_xstart'])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = th.randn_like(x)
        mean_pred = out['pred_xstart'] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

    def ddim_reverse_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, 'Reverse ODE only for deterministic path'
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs)
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out['pred_xstart']) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = out['pred_xstart'] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps
        return {'sample': mean_pred, 'pred_xstart': out['pred_xstart']}

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress, eta=eta):
            final = sample
        return final['sample']

    def ddim_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, eta=eta)
                yield out
                img = out['sample']

    def ddim_reverse_sample_loop(self, model, shape, noise, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        """
        Generate noise from the model using DDIM reverse.
        The input noise is actually x_{0}

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_reverse_sample_loop_progressive(model, shape, noise=noise, clip_denoised=clip_denoised, denoised_fn=denoised_fn, cond_fn=cond_fn, model_kwargs=model_kwargs, device=device, progress=progress, eta=eta):
            final = sample
        return final['sample']

    def ddim_reverse_sample_loop_progressive(self, model, shape, noise, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM reverse.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        assert noise is not None, 'input image has to be provided as noise'
        img = noise
        indices = list(range(self.num_timesteps))
        if progress:
            indices = tqdm(indices)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_reverse_sample(model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs, eta=eta)
                yield out
                img = out['sample']

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out['mean'], out['log_variance'])
        kl = mean_flat(kl) / np.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out['mean'], log_scales=0.5 * out['log_variance'])
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        output = th.where(t == 0, decoder_nll, kl)
        return {'output': output, 'pred_xstart': out['pred_xstart']}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        terms = {}
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms['loss'] = self._vb_terms_bpd(model=model, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, model_kwargs=model_kwargs)['output']
            if self.loss_type == LossType.RESCALED_KL:
                terms['loss'] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms['vb'] = self._vb_terms_bpd(model=lambda *args, r=frozen_out: r, x_start=x_start, x_t=x_t, t=t, clip_denoised=False)['output']
                if self.loss_type == LossType.RESCALED_MSE:
                    terms['vb'] *= self.num_timesteps / 1000.0
            target = {ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0], ModelMeanType.START_X: x_start, ModelMeanType.EPSILON: noise}[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms['mse'] = mean_flat((target - model_output) ** 2)
            if 'vb' in terms:
                terms['loss'] = terms['mse'] + terms['vb']
            else:
                terms['loss'] = terms['mse']
        else:
            raise NotImplementedError(self.loss_type)
        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]
        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with th.no_grad():
                out = self._vb_terms_bpd(model, x_start=x_start, x_t=x_t, t=t_batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
            vb.append(out['output'])
            xstart_mse.append(mean_flat((out['pred_xstart'] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out['pred_xstart'])
            mse.append(mean_flat((eps - noise) ** 2))
        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)
        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {'total_bpd': total_bpd, 'prior_bpd': prior_bpd, 'vb': vb, 'xstart_mse': xstart_mse, 'mse': mse}


class _WrappedModel:

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs['betas'])
        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs['betas'] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps, self.original_num_steps)

    def _scale_timesteps(self, t):
        return t


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith('ddim'):
            desired_count = int(section_counts[len('ddim'):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f'cannot create exactly {num_timesteps} steps with an integer stride')
        elif section_counts.startswith('ldm_ddim'):
            desired_count = int(section_counts[len('ldm_ddim'):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(1, num_timesteps + 1, i))
            raise ValueError(f'cannot create exactly {num_timesteps} steps with an integer stride')
        elif section_counts == 'fast27':
            steps = space_timesteps(num_timesteps, '10,10,3,2,2')
            steps.remove(num_timesteps - 1)
            steps.add(num_timesteps - 3)
            return steps
        section_counts = [int(x) for x in section_counts.split(',')]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f'cannot divide section of {size} steps into {section_count}')
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def create_gaussian_diffusion(*, steps=1000, learn_sigma=False, sigma_small=False, noise_schedule='linear', use_kl=False, predict_xstart=False, rescale_timesteps=False, rescale_learned_sigmas=False, timestep_respacing=''):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(use_timesteps=space_timesteps(steps, timestep_respacing), betas=betas, model_mean_type=gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X, model_var_type=(gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL) if not learn_sigma else gd.ModelVarType.LEARNED_RANGE, loss_type=loss_type, rescale_timesteps=rescale_timesteps)


class LatentDiffusion(nn.Module):
    LDM_CONFIGS = {'sd://v1-3': ('v1-inference.yaml', (512, 512), (64, 64)), 'sd://v1-4': ('v1-inference.yaml', (512, 512), (64, 64)), 'sd://v1-5': ('v1-inference.yaml', (512, 512), (64, 64)), 'sd://v2-0-base': ('v2-inference.yaml', (512, 512), (64, 64)), 'sd://v2-0-v': ('v2-inference.yaml', (768, 768), (96, 96)), 'sd://v2-1-base': ('v2-inference.yaml', (512, 512), (64, 64)), 'sd://v2-1-v': ('v2-inference.yaml', (768, 768), (96, 96))}

    def __init__(self, diffusion: 'Optional[GaussianDiffusion]'=None, guidance_scale: 'float'=7.5, pixel_mean: 'Tuple[float]'=(0.5, 0.5, 0.5), pixel_std: 'Tuple[float]'=(0.5, 0.5, 0.5), init_checkpoint='sd://v1-3'):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        ldm_cfg, image_size, latent_image_size = self.LDM_CONFIGS[init_checkpoint]
        with DisableLogger():
            self.ldm: '_LatentDiffusion' = build_ldm_from_cfg(ldm_cfg)
        self.ldm.cond_stage_model.__class__ = add_device_property(self.ldm.cond_stage_model.__class__)
        self.init_checkpoint = init_checkpoint
        self.load_pretrain()
        self.image_size = image_size
        self.latent_image_size = latent_image_size
        self.latent_dim = self.ldm.channels
        assert self.latent_dim == self.ldm.first_stage_model.embed_dim
        if diffusion is None:
            diffusion = create_gaussian_diffusion(steps=1000, learn_sigma=False, noise_schedule='ldm_linear')
        self.diffusion = diffusion
        self.guidance_scale = guidance_scale
        self.register_buffer('uncond_inputs', self.embed_text(['']))
        self.register_buffer('pixel_mean', torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std', torch.tensor(pixel_std).view(-1, 1, 1), False)

    def load_pretrain(self):
        LdmCheckpointer(self.ldm).load(self.init_checkpoint)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        batched_inputs = batched_input_to_device(batched_inputs, next(self.parameters()).device)
        if self.training:
            return self.forward_train(batched_inputs)
        else:
            return self.forward_test(batched_inputs)

    def forward_train(self, batched_inputs):
        raise NotImplementedError

    def apply_model_with_guidence(self, x_noisy, t, cond):
        half = x_noisy[:len(x_noisy) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.ldm.apply_model(combined, t, cond)
        eps, rest = model_out[:, :self.latent_dim], model_out[:, self.latent_dim:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def embed_text(self, text):
        return self.ldm.get_learned_conditioning(text)

    @property
    def encoder(self):
        return self.ldm.first_stage_model.encoder

    @property
    def unet(self):
        return self.ldm.model.diffusion_model

    @property
    def decoder(self):
        return self.ldm.first_stage_model.decoder

    @torch.no_grad()
    def encode_to_latent(self, input_image):
        encoder_posterior = self.ldm.encode_first_stage(input_image)
        latent_image = self.ldm.get_first_stage_encoding(encoder_posterior.mean)
        return latent_image

    @torch.no_grad()
    def decode_from_latent(self, latent_image):
        return self.ldm.decode_first_stage(latent_image)

    def forward_test(self, batched_inputs):
        caption = batched_inputs['caption']
        batch_size = len(caption)
        cond_inputs = self.embed_text(caption)
        if self.guidance_scale != 1.0:
            uncond_inputs = self.uncond_inputs.expand_as(cond_inputs)
        else:
            uncond_inputs = None
        if uncond_inputs is None:
            latent_samples = self.diffusion.ddim_sample_loop(model=self.ldm.apply_model, shape=(batch_size, self.latent_dim, *self.latent_image_size), device=self.device, clip_denoised=False, model_kwargs={'cond': cond_inputs})
        else:
            latent_samples = self.diffusion.ddim_sample_loop(model=self.apply_model_with_guidence, shape=(batch_size * 2, self.latent_dim, *self.latent_image_size), device=self.device, clip_denoised=False, model_kwargs={'cond': torch.cat([cond_inputs, uncond_inputs], dim=0)})[:batch_size]
        decoded_samples = self.ldm.decode_first_stage(latent_samples)
        out_samples = decoded_samples * self.pixel_std + self.pixel_mean
        out_samples = out_samples.clamp(0.0, 1.0)
        return out_samples


class LdmExtractor(FeatureExtractor):

    def __init__(self, ldm: 'Optional[LatentDiffusion]'=None, encoder_block_indices: 'Tuple[int, ...]'=(5, 7), unet_block_indices: 'Tuple[int, ...]'=(2, 5, 8, 11), decoder_block_indices: 'Tuple[int, ...]'=(2, 5), steps: 'Tuple[int, ...]'=(0,), share_noise: 'bool'=True, enable_resize: 'bool'=False):
        super().__init__()
        self.encoder_block_indices = encoder_block_indices
        self.unet_block_indices = unet_block_indices
        self.decoder_block_indices = decoder_block_indices
        self.steps = steps
        if ldm is not None:
            self.ldm = ldm
        else:
            self.ldm = LatentDiffusion()
        if enable_resize:
            self.image_preprocess = T.Resize(size=self.ldm.image_size, interpolation=T.InterpolationMode.BICUBIC)
        else:
            self.image_preprocess = None
        if share_noise:
            rng = torch.Generator().manual_seed(42)
            self.register_buffer('shared_noise', torch.randn(1, self.ldm.latent_dim, *self.ldm.latent_image_size, generator=rng))
        else:
            self.shared_noise = None
        self.reset_dim_stride()
        self._freeze()

    def reset_dim_stride(self):
        """Besides return dim and stride, this function also reset `self.encoder_blocks`,
        `self.unet_blocks`, `self.decoder_blocks` for feature extractor

        Returns:
            feature_dims: list of feature dimensions
            feature_strides: list of feature strides
        """
        all_encoder_blocks = []
        for i_level in range(self.ldm.encoder.num_resolutions):
            for i_block in range(self.ldm.encoder.num_res_blocks):
                all_encoder_blocks.append(self.ldm.encoder.down[i_level].block[i_block])
        encoder_dims = []
        encoder_strides = []
        encoder_blocks = []
        for idx in self.encoder_block_indices:
            encoder_dims.append(all_encoder_blocks[idx].in_channels)
            group_size = 2
            encoder_strides.append(2 ** ((idx + group_size) // group_size - 1))
            encoder_blocks.append(all_encoder_blocks[idx])
        assert set(self.unet_block_indices).issubset(set(range(len(self.ldm.unet.output_blocks))))
        unet_dims = []
        unet_strides = []
        unet_blocks = []
        for idx, block in enumerate(self.ldm.unet.output_blocks):
            if idx in self.unet_block_indices:
                unet_dims.append(block[0].channels)
                group_size = 3
                unet_strides.append(64 // 2 ** ((idx + group_size) // group_size - 1))
                unet_blocks.append(block)
        all_decoder_blocks = []
        for i_level in reversed(range(self.ldm.decoder.num_resolutions)):
            for i_block in range(self.ldm.decoder.num_res_blocks + 1):
                all_decoder_blocks.append(self.ldm.decoder.up[i_level].block[i_block])
        decoder_dims = []
        decoder_strides = []
        decoder_blocks = []
        for idx in self.decoder_block_indices:
            decoder_dims.append(all_decoder_blocks[idx].in_channels)
            group_size = 3
            decoder_strides.append(8 // 2 ** ((idx + group_size) // group_size - 1))
            decoder_blocks.append(all_decoder_blocks[idx])
        feature_dims = encoder_dims + unet_dims * len(self.steps) + decoder_dims
        feature_strides = encoder_strides + unet_strides * len(self.steps) + decoder_strides
        self.encoder_blocks = encoder_blocks
        self.unet_blocks = unet_blocks
        self.decoder_blocks = decoder_blocks
        return feature_dims, feature_strides

    @property
    def feature_size(self):
        return self.ldm.image_size

    @property
    def feature_dims(self):
        return self.reset_dim_stride()[0]

    @property
    def feature_strides(self):
        return self.reset_dim_stride()[1]

    @property
    def num_groups(self) ->int:
        num_groups = len(self.encoder_block_indices)
        num_groups += len(self.unet_block_indices)
        num_groups += len(self.decoder_block_indices)
        return num_groups

    @property
    def grouped_indices(self):
        ret = []
        for i in range(len(self.encoder_block_indices)):
            ret.append([i])
        offset = len(self.encoder_block_indices)
        for i in range(len(self.unet_block_indices)):
            cur_indices = []
            for t in range(len(self.steps)):
                cur_indices.append(i + t * len(self.unet_block_indices) + offset)
            ret.append(cur_indices)
        offset += len(self.steps) * len(self.unet_block_indices)
        for i in range(len(self.decoder_block_indices)):
            ret.append([i + offset])
        return ret

    @property
    def pixel_mean(self):
        return self.ldm.pixel_mean

    @property
    def pixel_std(self):
        return self.ldm.pixel_std

    @property
    def device(self):
        return self.ldm.device

    @torch.no_grad()
    def build_text_embed(self, text: 'List[List[str]]', batch_size=64, flatten=True):
        if isinstance(text, str):
            text = [text]
        if isinstance(text[0], str):
            text = [[t] for t in text]
        assert isinstance(text[0], list)
        flatten_text = [t for sublist in text for t in sublist]
        text_embed_list = []
        for i in range(0, len(flatten_text), batch_size):
            cur_text = flatten_text[i:i + batch_size]
            text_embed = self.ldm.embed_text(cur_text)
            text_embed_list.append(text_embed)
        return torch.concat(text_embed_list, dim=0)

    def encoder_forward(self, x):
        encoder = self.ldm.encoder
        ret_features = []
        temb = None
        hs = [encoder.conv_in(x)]
        for i_level in range(encoder.num_resolutions):
            for i_block in range(encoder.num_res_blocks):
                if encoder.down[i_level].block[i_block] in self.encoder_blocks:
                    ret_features.append(hs[-1].contiguous())
                h = encoder.down[i_level].block[i_block](hs[-1], temb)
                if len(encoder.down[i_level].attn) > 0:
                    h = encoder.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != encoder.num_resolutions - 1:
                hs.append(encoder.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = encoder.mid.block_1(h, temb)
        h = encoder.mid.attn_1(h)
        h = encoder.mid.block_2(h, temb)
        h = encoder.norm_out(h)
        h = h * torch.sigmoid(h)
        h = encoder.conv_out(h)
        return h, ret_features

    def encode_to_latent(self, image: 'torch.Tensor'):
        h, ret_features = self.encoder_forward(image)
        moments = self.ldm.ldm.first_stage_model.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        latent_image = self.ldm.ldm.scale_factor * posterior.mean
        return latent_image, ret_features

    def unet_forward(self, x, timesteps, context, cond_emb=None):
        unet = self.ldm.unet
        ret_features = []
        hs = []
        t_emb = timestep_embedding(timesteps, unet.model_channels, repeat_only=False)
        emb = unet.time_embed(t_emb)
        if cond_emb is not None:
            emb += cond_emb
        h = x
        for module in unet.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = unet.middle_block(h, emb, context)
        for module in unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if module in self.unet_blocks:
                ret_features.append(h.contiguous())
            h = module(h, emb, context)
        return unet.out(h), ret_features

    def decoder_forward(self, z):
        decoder = self.ldm.decoder
        ret_features = []
        decoder.last_z_shape = z.shape
        temb = None
        h = decoder.conv_in(z)
        h = decoder.mid.block_1(h, temb)
        h = decoder.mid.attn_1(h)
        h = decoder.mid.block_2(h, temb)
        for i_level in reversed(range(decoder.num_resolutions)):
            for i_block in range(decoder.num_res_blocks + 1):
                if decoder.up[i_level].block[i_block] in self.decoder_blocks:
                    ret_features.append(h.contiguous())
                h = decoder.up[i_level].block[i_block](h, temb)
                if len(decoder.up[i_level].attn) > 0:
                    h = decoder.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = decoder.up[i_level].upsample(h)
        if decoder.give_pre_end:
            return h
        h = decoder.norm_out(h)
        h = h * torch.sigmoid(h)
        h = decoder.conv_out(h)
        if decoder.tanh_out:
            h = torch.tanh(h)
        return h, ret_features

    def decode_to_image(self, z):
        z = 1.0 / self.ldm.ldm.scale_factor * z
        z = self.ldm.ldm.first_stage_model.post_quant_conv(z)
        dec, ret_features = self.decoder_forward(z)
        return dec, ret_features

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs (dict): expected keys: "img", Optional["caption"]

        """
        features = []
        image = batched_inputs['img']
        batch_size = image.shape[0]
        if self.image_preprocess is None:
            normalized_image = (image - self.pixel_mean) / self.pixel_std
        else:
            normalized_image = self.image_preprocess((image - self.pixel_mean) / self.pixel_std)
        if 'caption' in batched_inputs:
            captions = batched_inputs['caption']
        else:
            captions = [''] * batch_size
        latent_image, encoder_features = self.encode_to_latent(normalized_image)
        cond_inputs = batched_inputs.get('cond_inputs', self.ldm.embed_text(captions))
        unet_features = []
        for i, t in enumerate(self.steps):
            if 'cond_emb' in batched_inputs:
                cond_emb = batched_inputs['cond_emb'][:, i]
            else:
                cond_emb = None
            if t < 0:
                noisy_latent_image = latent_image
                t = torch.tensor([0], device=self.device).expand(batch_size)
            else:
                t = torch.tensor([t], device=self.device).expand(batch_size)
                if self.shared_noise is not None:
                    if self.shared_noise.shape[2:] != latent_image.shape[2:]:
                        assert self.image_preprocess is None
                        shared_noise = F.interpolate(self.shared_noise, size=latent_image.shape[2:], mode='bicubic', align_corners=False)
                    else:
                        shared_noise = self.shared_noise
                    noise = shared_noise.expand_as(latent_image)
                else:
                    noise = None
                noisy_latent_image = self.ldm.diffusion.q_sample(latent_image, t, noise)
            _, cond_unet_features = self.unet_forward(noisy_latent_image, t, cond_inputs, cond_emb=cond_emb)
            unet_features.extend(cond_unet_features)
        _, decoder_features = self.decode_to_image(latent_image)
        features = [*encoder_features, *unet_features, *decoder_features]
        assert len(features) == len(self.feature_dims), f'{len(features)} != {len(self.feature_dims)}'
        for indices in self.grouped_indices:
            for idx in indices:
                if self.image_preprocess is not None:
                    continue
                assert image.shape[-2] // self.feature_strides[idx] == features[idx].shape[-2]
                assert image.shape[-1] // self.feature_strides[idx] == features[idx].shape[-1]
        return features


class PositionalLinear(nn.Module):

    def __init__(self, in_features, out_features, seq_len=77, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, out_features))
        trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(1) + self.positional_embedding
        return x


class LdmImplicitCaptionerExtractor(nn.Module):

    def __init__(self, learnable_time_embed=True, num_timesteps=1, clip_model_name='ViT-L-14', **kwargs):
        super().__init__()
        self.ldm_extractor = LdmExtractor(**kwargs)
        self.text_embed_shape = self.ldm_extractor.ldm.embed_text(['']).shape[1:]
        self.clip = ClipAdapter(name=clip_model_name, normalize=False)
        self.clip_project = PositionalLinear(self.clip.dim_latent, self.text_embed_shape[1], self.text_embed_shape[0])
        self.alpha_cond = nn.Parameter(torch.zeros_like(self.ldm_extractor.ldm.uncond_inputs))
        self.learnable_time_embed = learnable_time_embed
        if self.learnable_time_embed:
            self.time_embed_project = PositionalLinear(self.clip.dim_latent, self.ldm_extractor.ldm.unet.time_embed[-1].out_features, num_timesteps)
            self.alpha_cond_time_embed = nn.Parameter(torch.zeros(self.ldm_extractor.ldm.unet.time_embed[-1].out_features))

    @property
    def feature_size(self):
        return self.ldm_extractor.feature_size

    @property
    def feature_dims(self):
        return self.ldm_extractor.feature_dims

    @property
    def feature_strides(self):
        return self.ldm_extractor.feature_strides

    @property
    def num_groups(self) ->int:
        return self.ldm_extractor.num_groups

    @property
    def grouped_indices(self):
        return self.ldm_extractor.grouped_indices

    def extra_repr(self):
        return f'learnable_time_embed={self.learnable_time_embed}'

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs (dict): expected keys: "img", Optional["caption"]

        """
        image = batched_inputs['img']
        prefix = self.clip.embed_image(image).image_embed
        prefix_embed = self.clip_project(prefix)
        batched_inputs['cond_inputs'] = self.ldm_extractor.ldm.uncond_inputs + torch.tanh(self.alpha_cond) * prefix_embed
        if self.learnable_time_embed:
            batched_inputs['cond_emb'] = torch.tanh(self.alpha_cond_time_embed) * self.time_embed_project(prefix)
        self.set_requires_grad(self.training)
        return self.ldm_extractor(batched_inputs)

    def set_requires_grad(self, requires_grad):
        for p in self.ldm_extractor.ldm.ldm.model.parameters():
            p.requires_grad = requires_grad


@torch.no_grad()
def _concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def get_world_batch_sizes(batch_size: 'int', device):
    batch_size = torch.as_tensor([batch_size], dtype=torch.long, device=device)
    global_batch_sizes = _concat_all_gather(batch_size)
    return global_batch_sizes


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
    padded_tensor[:tensor.shape[0]] = tensor
    tensors_gather = [torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) for _ in range(comm.get_world_size())]
    torch.distributed.all_gather(tensors_gather, padded_tensor, async_op=False)
    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])
    output = torch.cat(results, dim=0)
    return output


def dist_collect(tensor):
    """
    Performs all_gather operation on the provided tensors, with dynamic batch size.
    Use diff_dist to get gradient
    """
    if comm.get_world_size() == 1:
        return tensor
    global_batch_sizes = get_world_batch_sizes(tensor.shape[0], tensor.device)
    max_batch_size = global_batch_sizes.max().item()
    padded_tensor = torch.zeros(max_batch_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
    padded_tensor[:tensor.shape[0]] = tensor
    tensors_gather = [torch.ones((max_batch_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) for _ in range(comm.get_world_size())]
    tensors_gather = diff_dist.all_gather(tensors_gather, padded_tensor)
    results = []
    for i, batch_size in enumerate(global_batch_sizes):
        results.append(tensors_gather[i][:batch_size])
    output = torch.cat(results, dim=0)
    return output


class MaskGroundingCriterion(nn.Module):

    def __init__(self, collect_mode: 'str'='concat', loss_weight=1.0):
        super().__init__()
        self.collect_mode = collect_mode
        self.loss_weight = loss_weight
        if collect_mode == 'diff':
            self.collect_func = dist_collect
        elif collect_mode == 'concat':
            self.collect_func = concat_all_gather
        elif collect_mode is None:
            self.collect_func = lambda x: x
        else:
            raise ValueError(f'collect_mode {collect_mode} is not supported')

    def extra_repr(self) ->str:
        return f'collect_mode={self.collect_mode}, \nloss_weight={self.loss_weight} \n'

    def forward(self, outputs, targets):
        losses = {}
        losses.update(self.get_loss(outputs, targets))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.get_loss(aux_outputs, targets)
                l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

    def get_loss(self, outputs, targets):
        logit_scale = outputs['logit_scale']
        rank = comm.get_rank() if self.collect_mode is not None else 0
        mask_embed = outputs['mask_embed']
        word_embed = outputs['word_embed']
        word_valid_mask = torch.stack([t['word_valid_mask'] for t in targets], dim=0)
        mask_embed = F.normalize(mask_embed, dim=-1)
        word_embed = F.normalize(word_embed, dim=-1)
        batch_size, num_queries, embed_dim = mask_embed.shape
        assert batch_size == word_embed.shape[0], f'{batch_size} != {word_embed.shape[0]}'
        assert embed_dim == word_embed.shape[2], f'{embed_dim} != {word_embed.shape[2]}'
        num_words = word_embed.shape[1]
        mask_embed = mask_embed.reshape(batch_size * num_queries, embed_dim)
        word_embed = word_embed.reshape(batch_size * num_words, embed_dim)
        if self.collect_mode is not None and comm.get_world_size() > 1:
            global_batch_sizes = get_world_batch_sizes(batch_size, device=mask_embed.device)
            global_batch_size = global_batch_sizes.sum().item()
        else:
            global_batch_sizes = None
            global_batch_size = batch_size
        sim_global_mask_word = self.collect_func(mask_embed) @ word_embed.t() * logit_scale
        sim_global_mask_word = sim_global_mask_word.view(global_batch_size, num_queries, batch_size, num_words)
        sim_global_img_txt = (sim_global_mask_word.softmax(dim=1) * sim_global_mask_word).sum(dim=1).mean(dim=-1)
        sim_mask_global_word = mask_embed @ self.collect_func(word_embed).t() * logit_scale
        sim_mask_global_word = sim_mask_global_word.view(batch_size, num_queries, global_batch_size, num_words)
        sim_img_global_txt = (sim_mask_global_word.softmax(dim=1) * sim_mask_global_word).sum(dim=1).mean(dim=-1)
        if global_batch_sizes is None:
            labels = torch.arange(batch_size, dtype=torch.long, device=mask_embed.device) + batch_size * rank
        else:
            labels = torch.arange(batch_size, dtype=torch.long, device=mask_embed.device) + global_batch_sizes[:rank].sum()
        valid_mask = word_valid_mask.any(dim=-1)
        global_valid_mask = self.collect_func(valid_mask)
        loss_global_img_txt = F.cross_entropy(sim_global_img_txt.t(), labels, reduction='none')
        loss_global_img_txt = (loss_global_img_txt * valid_mask).mean()
        loss_img_global_txt = F.cross_entropy(sim_img_global_txt, labels, weight=global_valid_mask.float())
        if not torch.isfinite(loss_img_global_txt).all():
            loss_img_global_txt = F.cross_entropy(sim_img_global_txt, labels)
        loss = 0.5 * (loss_global_img_txt + loss_img_global_txt)
        return {'loss_mask_word': loss * self.loss_weight}


class PseudoClassEmbed(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        fg_logits = torch.ones((*x.shape[:-1], self.num_classes), dtype=x.dtype, device=x.device)
        bg_logits = torch.zeros((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
        logits = torch.cat([fg_logits, bg_logits], dim=-1)
        return logits


class MaskPooling(nn.Module):

    def __init__(self, hard_pooling=True, mask_threshold=0.5):
        super().__init__()
        self.hard_pooling = hard_pooling
        self.mask_threshold = mask_threshold

    def extra_repr(self) ->str:
        return f'hard_pooling={self.hard_pooling}\nmask_threshold={self.mask_threshold}\n'

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """
        assert x.shape[-2:] == mask.shape[-2:]
        mask = mask.detach()
        mask = mask.sigmoid()
        if self.hard_pooling:
            mask = mask > self.mask_threshold
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-08
        mask_pooled_x = torch.einsum('bchw,bqhw->bqc', x, mask / denorm)
        output = {'mask_pooled_features': mask_pooled_x}
        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PooledMaskEmbed(nn.Module):

    def __init__(self, hidden_dim, mask_dim, projection_dim, temperature=0.07):
        super().__init__()
        self.pool_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.mask_embed = nn.Sequential(nn.LayerNorm(mask_dim), MLP(mask_dim, hidden_dim, projection_dim, 3))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.mask_pooling = MaskPooling()

    def forward(self, decoder_output, input_mask_embed, mask_features, pred_logits, pred_masks):
        """
        Args:
            decoder_output: [B, Q, C]
            input_mask_embed: [B, Q, C]
            mask_features: [B, C, H, W]
            pred_logits: [B, Q, K+1]
            pred_masks: [B, Q, H, W]
        """
        mask_pooled_x = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_results = self.mask_pooling(mask_features, pred_masks)
        mask_pooled_x = mask_pooled_results['mask_pooled_features']
        outputs_mask = mask_pooled_results.get('outputs_mask', None)
        mask_pooled_x = self.pool_proj(mask_pooled_x)
        mask_pooled_x += decoder_output
        mask_embed = self.mask_embed(mask_pooled_x)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        output = {'mask_embed': mask_embed, 'mask_pooled_features': mask_pooled_x, 'logit_scale': logit_scale}
        if outputs_mask is not None:
            output['outputs_mask'] = outputs_mask
        return output


def prompt_labels(labels, prompt):
    if prompt is None:
        return labels
    labels = copy.deepcopy(labels)
    assert prompt in ['a', 'photo', 'scene']
    if prompt == 'a':
        for i in range(len(labels)):
            labels[i] = [f'a {l}' for l in labels[i]]
    elif prompt == 'photo':
        for i in range(len(labels)):
            labels[i] = [f'a photo of a {l}.' for l in labels[i]]
    elif prompt == 'scene':
        for i in range(len(labels)):
            labels[i] = [f'a photo of a {l} in the scene.' for l in labels[i]]
    else:
        raise NotImplementedError
    return labels


def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


class WordEmbed(nn.Module):

    def __init__(self, projection_dim, clip_model_name='ViT-L-14', word_dropout=0.0, word_tags='noun_phrase', num_words=8, prompt='photo'):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.clip = ClipAdapter(name=self.clip_model_name, normalize=False)
        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)
        self.test_labels = None
        self._test_text_embed_dict = OrderedDict()
        if comm.get_local_rank() == 0:
            nltk.download('popular', quiet=True)
            nltk.download('universal_tagset', quiet=True)
        comm.synchronize()
        self.nltk = nltk
        self.word_dropout = word_dropout
        self.word_tags = word_tags
        self.num_words = num_words
        self.prompt = prompt

    def extra_repr(self) ->str:
        return f'clip_model_name={self.clip_model_name},\nword_dropout={self.word_dropout},\nword_tags={self.word_tags},\nnum_words={self.num_words}'

    @property
    def device(self):
        return self.clip.device

    def _open_state_dict(self):
        return {'test_labels': self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, 'open_state_dict'):
                module.open_state_dict(destination, prefix + name + '.')
        return destination

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        return build_clip_text_embed(clip_model_name=self.clip.clip, labels=labels, verbose=verbose)

    def get_and_cache_test_text_embed(self, labels):
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            text_embed = self.build_text_embed(labels, verbose=True)
            if len(self._test_text_embed_dict) > 3:
                self._test_text_embed_dict.pop(list(self._test_text_embed_dict.keys())[0])
            self._test_text_embed_dict[labels] = text_embed.cpu()
        else:
            text_embed = self._test_text_embed_dict[labels]
        return text_embed

    def get_tag(self, caption, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for word, pos in self.nltk.pos_tag(self.nltk.word_tokenize(caption), tagset='universal'):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def _get_phrase(self, caption, with_preposition):
        if with_preposition:
            grammar = """
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                    {<NBAR>} # If pattern is not found, just a single NBAR is ok
            """
        else:
            grammar = """
                NBAR:
                    {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR>} # If pattern is not found, just a single NBAR is ok
            """
        tokenized = self.nltk.word_tokenize(caption)
        chunker = self.nltk.RegexpParser(grammar)
        chunked = chunker.parse(self.nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []
        for subtree in chunked:
            if isinstance(subtree, self.nltk.Tree):
                current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = ' '.join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
        return continuous_chunk

    def get_noun_phrase(self, caption):
        noun_phrase = []
        noun_phrase.extend(self._get_phrase(caption, with_preposition=False))
        noun_phrase.extend(self._get_phrase(caption, with_preposition=True))
        return list(set(noun_phrase))

    def prepare_targets(self, captions, targets):
        if targets is None:
            targets = [{} for _ in range(len(captions))]
        for caption, target in zip(captions, targets):
            caption = np.random.choice(caption)
            if self.word_tags == 'noun_phrase':
                words = self.get_noun_phrase(caption)
            elif 'noun_phrase' in self.word_tags:
                words = []
                words.extend(self.get_noun_phrase(caption))
                words.extend(self.get_tag(caption, tuple(set(self.word_tags) - set('noun_phrase'))))
                words = list(set(words))
            else:
                words = self.get_tag(caption, self.word_tags)
            if not len(words):
                words = ['']
            words_after_drop = [w for w in words if np.random.rand() > self.word_dropout]
            if len(words_after_drop) == 0:
                words_after_drop = words
            words = np.random.choice(words_after_drop, size=self.num_words).tolist()
            target['words'] = words
            valid_mask = [(len(w) > 0) for w in words]
            valid_mask = torch.tensor(valid_mask, device=self.device, dtype=torch.bool)
            target['word_valid_mask'] = valid_mask
        return targets

    def forward(self, outputs, targets=None):
        if self.training:
            words = [x['words'] for x in targets]
            words = prompt_labels(words, self.prompt)
            word_embed = self.build_text_embed(words)
            word_embed = torch.stack(word_embed.split([len(w) for w in words]), dim=0)
            word_embed = self.text_proj(word_embed)
            return {'word_embed': word_embed}
        else:
            assert targets is None
            assert self.test_labels is not None
            labels = self.test_labels
            labels = prompt_labels(labels, self.prompt)
            text_embed = self.get_and_cache_test_text_embed(labels)
            text_embed = self.text_proj(text_embed)
            return {'text_embed': text_embed, 'labels': labels}


class CategoryEmbed(nn.Module):

    def __init__(self, labels, projection_dim, clip_model_name='ViT-L-14', prompt=None):
        super().__init__()
        self.labels = labels
        self.clip_model_name = clip_model_name
        self.clip = ClipAdapter(name=self.clip_model_name, normalize=False)
        if projection_dim < 0:
            self.text_proj = nn.Identity()
        else:
            self.text_proj = nn.Linear(self.clip.dim_latent, projection_dim)
        self.register_buffer('text_embed', self.build_text_embed(prompt_labels(labels, prompt), verbose=True), False)
        self.null_embed = nn.Parameter(self.build_text_embed(''))
        self.prompt = prompt
        self.test_labels = None
        self._test_text_embed_dict = dict()

    def extra_repr(self) ->str:
        return f'clip_model_name={self.clip_model_name},\n'

    @property
    def device(self):
        return self.clip.device

    def _open_state_dict(self):
        return {'test_labels': self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, 'open_state_dict'):
                module.open_state_dict(destination, prefix + name + '.')
        return destination

    @torch.no_grad()
    def build_text_embed(self, labels, verbose=False):
        return build_clip_text_embed(clip_model_name=self.clip.clip, labels=labels, verbose=verbose)

    def get_and_cache_test_text_embed(self, labels):
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            text_embed = self.build_text_embed(labels, verbose=True)
            self._test_text_embed_dict[labels] = text_embed.cpu()
        else:
            text_embed = self._test_text_embed_dict[labels]
        return text_embed

    def forward(self, outputs, targets=None):
        if self.training:
            text_embed = self.text_proj(self.text_embed)
            null_embed = self.text_proj(self.null_embed)
            return {'text_embed': text_embed, 'null_embed': null_embed, 'labels': self.labels}
        else:
            assert targets is None
            assert self.test_labels is not None
            labels = self.test_labels
            text_embed = self.get_and_cache_test_text_embed(prompt_labels(labels, self.prompt))
            text_embed = self.text_proj(text_embed)
            null_embed = self.text_proj(self.null_embed)
            return {'text_embed': text_embed, 'null_embed': null_embed, 'labels': labels}


class CLIPOpenClassEmbed(nn.Module):

    def __init__(self, labels, hidden_dim, projection_modality='text', clip_model_name='ViT-L-14', with_null_embed=True, temperature=0.07, ensemble_method='max'):
        super().__init__()
        self.labels = labels
        assert projection_modality in ['text', 'image']
        self.projection_modality = projection_modality
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.clip_model_name = clip_model_name
        self.register_buffer('text_embed', self.build_text_embed(labels), False)
        if with_null_embed:
            self.null_embed = nn.Parameter(self.build_text_embed(''))
        else:
            self.null_embed = None
        if self.projection_modality == 'text':
            self.embed_projection = nn.Linear(self.text_embed.shape[-1], hidden_dim, bias=False)
        else:
            self.embed_projection = nn.Linear(hidden_dim, self.text_embed.shape[-1], bias=False)
        assert ensemble_method in ['max', 'mean'], f'ensemble_method {ensemble_method} not supported'
        self.ensemble_method = ensemble_method
        self.test_labels = None
        self._test_text_embed_dict = dict()

    def _open_state_dict(self):
        return {'test_labels': self.test_labels}

    def _save_open_state_dict(self, destination, prefix):
        for k, v in self._open_state_dict().items():
            destination[prefix + k] = v

    def open_state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()
        self._save_open_state_dict(destination, prefix)
        for name, module in self.named_modules(remove_duplicate=True):
            if module is self:
                continue
            if module is not None and hasattr(module, 'open_state_dict'):
                module.open_state_dict(destination, prefix + name + '.')
        return destination

    def extra_repr(self):
        return f'clip_model_name={self.clip_model_name}, \nensemble_method={self.ensemble_method}, \nprojection_modality={self.projection_modality}, \n'

    @torch.no_grad()
    def build_text_embed(self, labels):
        return build_clip_text_embed(clip_model_name=self.clip_model_name, labels=labels)

    def get_and_cache_test_text_embed(self, labels):
        labels = to_tuple(labels)
        if labels not in self._test_text_embed_dict:
            self._test_text_embed_dict[labels] = self.build_text_embed(labels)
        return self._test_text_embed_dict[labels]

    def forward(self, x):
        if self.projection_modality == 'image':
            x = self.embed_projection(x)
        x = F.normalize(x, dim=-1)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        if self.test_labels is None:
            labels = self.labels
            text_embed = self.text_embed
        else:
            labels = self.test_labels
            text_embed = self.get_and_cache_test_text_embed(labels)
        if self.projection_modality == 'text':
            text_embed = self.embed_projection(text_embed)
        text_embed = F.normalize(text_embed, dim=-1)
        pred = logit_scale * (x @ text_embed.t())
        pred = ensemble_logits_with_labels(pred, labels, ensemble_method=self.ensemble_method)
        if self.null_embed is not None:
            if self.projection_modality == 'text':
                null_embed = self.embed_projection(self.null_embed)
            else:
                null_embed = self.null_embed
            null_embed = F.normalize(null_embed, dim=-1)
            null_pred = logit_scale * (x @ null_embed.t())
            pred = torch.cat([pred, null_pred], dim=-1)
        return pred


def get_openseg_labels(dataset, prompt_engineered=False):
    """get the labels in double list format,
    e.g. [[background, bag, bed, ...], ["aeroplane"], ...]
    """
    invalid_name = 'invalid_class_id'
    assert dataset in ['ade20k_150', 'ade20k_847', 'coco_panoptic', 'pascal_context_59', 'pascal_context_459', 'pascal_voc_21', 'lvis_1203']
    label_path = osp.join(osp.dirname(osp.abspath(__file__)), 'datasets/openseg_labels', f'{dataset}_with_prompt_eng.txt' if prompt_engineered else f'{dataset}.txt')
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()
    categories = []
    for line in lines:
        id, name = line.split(':')
        if name == invalid_name:
            continue
        categories.append({'id': int(id), 'name': name})
    return [dic['name'].split(',') for dic in categories]


class PoolingCLIPHead(WordEmbed):

    def __init__(self, clip_model_name='ViT-L-14-336', alpha=0.35, beta=0.65, prompt='photo', train_labels=None, normalize_logits=True, bg_labels=None):
        super(WordEmbed, self).__init__()
        self.clip_model_name = clip_model_name
        self.clip = MaskCLIP(name=self.clip_model_name)
        self.alpha = alpha
        self.beta = beta
        self.test_labels = None
        self._test_text_embed_dict = dict()
        self.prompt = prompt
        if train_labels is None:
            self.train_labels = get_openseg_labels('coco_panoptic', prompt_engineered=True)
        else:
            self.train_labels = train_labels
        self.bg_labels = bg_labels
        self.normalize_logits = normalize_logits

    def extra_repr(self) ->str:
        return f'clip_model_name={self.clip_model_name},\n'

    @property
    def with_bg(self):
        return self.bg_labels is not None

    def prepare_targets(self, outputs, targets):
        target_mask_embed = self.clip.get_mask_embed(outputs['images'], outputs['pred_masks'])
        for idx in range(len(targets)):
            targets[idx]['target_mask_embed'] = target_mask_embed[idx]
        return targets

    def forward(self, outputs, targets=None):
        assert not self.training, 'PoolingCLIPHead only supports inference'
        assert targets is None
        assert self.test_labels is not None
        pred_open_logits = outputs.pop('pred_open_logits')
        labels = prompt_labels(self.test_labels, self.prompt)
        if self.with_bg and pred_open_logits.shape[-1] == len(self.test_labels) + 1:
            labels.append(self.bg_labels)
        category_overlapping_list = []
        train_labels = {l for label in self.train_labels for l in label}
        for test_label in self.test_labels:
            category_overlapping_list.append(not set(train_labels).isdisjoint(set(test_label)))
        if self.with_bg and pred_open_logits.shape[-1] == len(self.test_labels) + 1:
            category_overlapping_list.append(False)
        category_overlapping_mask = torch.tensor(category_overlapping_list, device=outputs['images'].device, dtype=torch.long)
        text_embed = self.get_and_cache_test_text_embed(labels)
        mask_pred_results = outputs['pred_masks']
        clip_results = self.clip(outputs['images'], mask_pred_results, text_embed, labels)
        mask_pred_open_logits = clip_results['mask_pred_open_logits']
        if self.normalize_logits:
            pred_open_prob = pred_open_logits.softmax(dim=-1)
            mask_pred_open_prob = mask_pred_open_logits.softmax(dim=-1)
            pred_open_logits_base = (pred_open_prob ** (1 - self.alpha) * mask_pred_open_prob ** self.alpha).log() * category_overlapping_mask
            pred_open_logits_novel = (pred_open_prob ** (1 - self.beta) * mask_pred_open_prob ** self.beta).log() * (1 - category_overlapping_mask)
        else:
            pred_open_logits_base = pred_open_logits * (1 - self.alpha) + mask_pred_open_logits * self.alpha * category_overlapping_mask
            pred_open_logits_novel = pred_open_logits * (1 - self.beta) + mask_pred_open_logits * self.beta * (1 - category_overlapping_mask)
        pred_open_logits = pred_open_logits_base + pred_open_logits_novel
        ret = {'pred_open_logits': pred_open_logits}
        if 'labels' in outputs:
            ret['labels'] = labels
        return ret


class OpenPanopticInference(nn.Module):

    def __init__(self, model, labels, metadata=None, semantic_on=True, instance_on=True, panoptic_on=True, test_topk_per_image=100):
        super().__init__()
        self.model = model
        self.labels = labels
        self.metadata = metadata
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.open_state_dict = OrderedDict()
        for k in self.model.open_state_dict():
            if k.endswith('test_labels'):
                self.open_state_dict[k] = self.labels
            elif k.endswith('metadata'):
                self.open_state_dict[k] = self.metadata
            elif k.endswith('num_classes'):
                self.open_state_dict[k] = self.num_classes
            elif k.endswith('semantic_on'):
                self.open_state_dict[k] = self.semantic_on
            elif k.endswith('instance_on'):
                self.open_state_dict[k] = self.instance_on
            elif k.endswith('panoptic_on'):
                self.open_state_dict[k] = self.panoptic_on
            elif k.endswith('test_topk_per_image'):
                self.open_state_dict[k] = self.test_topk_per_image

    @property
    def num_classes(self):
        return len(self.labels)

    def forward(self, batched_inputs):
        assert not self.training
        _open_state_dict = self.model.open_state_dict()
        self.model.load_open_state_dict(self.open_state_dict)
        results = self.model(batched_inputs)
        self.model.load_open_state_dict(_open_state_dict)
        return results


def batch_dice_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor'):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)


def batch_sigmoid_ce_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor'):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, 1 - targets)
    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: 'float'=1, cost_mask: 'float'=1, cost_dice: 'float'=1, num_points: 'int'=0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs cant be 0'
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs['pred_logits'][b].softmax(-1)
            tgt_ids = targets[b]['labels']
            cost_class = -out_prob[:, tgt_ids]
            out_mask = outputs['pred_masks'][b]
            tgt_mask = targets[b]['masks']
            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            tgt_mask = point_sample(tgt_mask, point_coords.repeat(tgt_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            out_mask = point_sample(out_mask, point_coords.repeat(out_mask.shape[0], 1, 1), align_corners=False).squeeze(1)
            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = 'Matcher ' + self.__class__.__name__
        body = ['cost_class: {}'.format(self.cost_class), 'cost_mask: {}'.format(self.cost_mask), 'cost_dice: {}'.format(self.cost_dice)]
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -torch.abs(gt_class_logits)


def dice_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor', num_masks: 'float'):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class NestedTensor(object):

    def __init__(self, tensors, mask: 'Optional[Tensor]'):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: 'List[Tensor]') ->NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32))
        max_size.append(max_size_i)
    max_size = tuple(max_size)
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)
        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), 'constant', 1)
        padded_masks.append(padded_mask)
    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)
    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: 'List[Tensor]'):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            return _onnx_nested_tensor_from_tensor_list(tensor_list)
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def sigmoid_ce_loss(inputs: 'torch.Tensor', targets: 'torch.Tensor', num_masks: 'float'):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, class_weight, mask_weight, dice_weight, num_layers, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        weight_dict = {'loss_ce': class_weight, 'loss_mask': mask_weight, 'loss_dice': dice_weight}
        aux_weight_dict = {}
        for i in range(num_layers):
            aux_weight_dict.update({(k + f'_{i}'): v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        masks = [t['masks'] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks
        target_masks = target_masks[tgt_idx]
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(src_masks, lambda logits: calculate_uncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        losses = {'loss_mask': sigmoid_ce_loss_jit(point_logits, point_labels, num_masks), 'loss_dice': dice_loss_jit(point_logits, point_labels, num_masks)}
        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {'labels': self.loss_labels, 'masks': self.loss_masks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def __repr__(self):
        head = 'Criterion ' + self.__class__.__name__
        body = ['matcher: {}'.format(self.matcher.__repr__(_repr_indent=8)), 'losses: {}'.format(self.losses), 'weight_dict: {}'.format(self.weight_dict), 'num_classes: {}'.format(self.num_classes), 'eos_coef: {}'.format(self.eos_coef), 'num_points: {}'.format(self.num_points), 'oversample_ratio: {}'.format(self.oversample_ratio), 'importance_sample_ratio: {}'.format(self.importance_sample_ratio)]
        _repr_indent = 4
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.
    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.
    Args:
        func: a stateless callable that takes tensor-like objects as arguments
    Returns:
        a callable which retries `func` if OOM is encountered.
    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU
    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.
        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == 'cuda' and hasattr(x, 'to')
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device='cpu')
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)
        logger = logging.getLogger(__name__)
        logger.info('Attempting to copy inputs to CPU due to CUDA OOM')
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        with autocast(enabled=False):
            return func(*new_args, **new_kwargs)
    return wrapped


class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None, use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs['res{}'.format(i + 2)] = out
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, 'forward_features', None)
    if not callable(forward_features):
        raise ValueError(f'Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. Please implement forward_features for {name} to only return mask features.')
    return model


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = 'Positional encoding ' + self.__class__.__name__
        body = ['num_pos_feats: {}'.format(self.num_pos_feats), 'temperature: {}'.format(self.temperature), 'normalize: {}'.format(self.normalize), 'scale: {}'.format(self.scale)]
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoderOnly(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


class MSDeformAttnFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def _is_power_of_2(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError('invalid input for _is_power_of_2: {} (type: {})'.format(n, type(n)))
    return n & n - 1 == 0 and n != 0


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation.")
        self.im2col_step = 128
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        try:
            output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        except:
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output


class MSDeformAttnTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device), torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return output


class MSDeformAttnTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation='relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoderOnly(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, activation='relu', num_feature_levels=4, enc_n_points=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        return memory, spatial_shapes, level_start_index


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation='relu', normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class SemanticSegmentorWithTTA(nn.Module):
    """
    A SemanticSegmentor with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`SemanticSegmentor.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=1):
        """
        Args:
            cfg (CfgNode):
            model (SemanticSegmentor): a SemanticSegmentor to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()
        self.model = model
        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`SemanticSegmentor.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if 'image' not in ret:
                image = read_image(ret.pop('file_name'), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
                ret['image'] = image
            if 'height' not in ret and 'width' not in ret:
                ret['height'] = image.shape[1]
                ret['width'] = image.shape[2]
            return ret
        processed_results = []
        for x in batched_inputs:
            result = self._inference_one_image(_maybe_read_image(x))
            processed_results.append(result)
        return processed_results

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor
        Returns:
            dict: one output dict
        """
        orig_shape = input['height'], input['width']
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        final_predictions = None
        count_predictions = 0
        for input, tfm in zip(augmented_inputs, tfms):
            count_predictions += 1
            with torch.no_grad():
                if final_predictions is None:
                    if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                        final_predictions = self.model([input])[0].pop('sem_seg').flip(dims=[2])
                    else:
                        final_predictions = self.model([input])[0].pop('sem_seg')
                elif any(isinstance(t, HFlipTransform) for t in tfm.transforms):
                    final_predictions += self.model([input])[0].pop('sem_seg').flip(dims=[2])
                else:
                    final_predictions += self.model([input])[0].pop('sem_seg')
        final_predictions = final_predictions / count_predictions
        return {'sem_seg': final_predictions}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop('transforms') for x in augmented_inputs]
        return augmented_inputs, tfms


class VideoSetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        target_masks = torch.cat([t['masks'][i] for t, (_, i) in zip(targets, indices)])
        src_masks = src_masks.flatten(0, 1)[:, None]
        target_masks = target_masks.flatten(0, 1)[:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(src_masks, lambda logits: calculate_uncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = point_sample(target_masks, point_coords, align_corners=False).squeeze(1)
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        losses = {'loss_mask': sigmoid_ce_loss_jit(point_logits, point_labels, num_masks), 'loss_dice': dice_loss_jit(point_logits, point_labels, num_masks)}
        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {'labels': self.loss_labels, 'masks': self.loss_masks}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def __repr__(self):
        head = 'Criterion ' + self.__class__.__name__
        body = ['matcher: {}'.format(self.matcher.__repr__(_repr_indent=8)), 'losses: {}'.format(self.losses), 'weight_dict: {}'.format(self.weight_dict), 'num_classes: {}'.format(self.num_classes), 'eos_coef: {}'.format(self.eos_coef), 'num_points: {}'.format(self.num_points), 'oversample_ratio: {}'.format(self.oversample_ratio), 'importance_sample_ratio: {}'.format(self.importance_sample_ratio)]
        _repr_indent = 4
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


class VideoHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: 'float'=1, cost_mask: 'float'=1, cost_dice: 'float'=1, num_points: 'int'=0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs cant be 0'
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []
        for b in range(bs):
            out_prob = outputs['pred_logits'][b].softmax(-1)
            tgt_ids = targets[b]['labels']
            cost_class = -out_prob[:, tgt_ids]
            out_mask = outputs['pred_masks'][b]
            tgt_mask = targets[b]['masks']
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            tgt_mask = point_sample(tgt_mask, point_coords.repeat(tgt_mask.shape[0], 1, 1), align_corners=False).flatten(1)
            out_mask = point_sample(out_mask, point_coords.repeat(out_mask.shape[0], 1, 1), align_corners=False).flatten(1)
            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = 'Matcher ' + self.__class__.__name__
        body = ['cost_class: {}'.format(self.cost_class), 'cost_mask: {}'.format(self.cost_mask), 'cost_dice: {}'.format(self.cost_dice)]
        lines = [head] + [(' ' * _repr_indent + line) for line in body]
        return '\n'.join(lines)


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        assert x.dim() == 5, f'{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead'
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(1), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-06
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t_z = torch.arange(self.num_pos_feats * 2, dtype=torch.float32, device=x.device)
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_pos_feats * 2))
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(0, 1, 4, 2, 3)
        return pos


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CrossAttentionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (FFNLayer,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm2d,
     lambda: ([], {'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MLP,
     lambda: ([], {'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MaskPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionEmbeddingSine,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PositionEmbeddingSine3D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (PseudoClassEmbed,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttentionLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (TransformerDecoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {})),
]

