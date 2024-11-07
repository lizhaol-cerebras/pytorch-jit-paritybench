import sys
_module = sys.modules[__name__]
del sys
cluster = _module
kmeans = _module
train_cluster = _module
compress_model = _module
data_utils = _module
diffusion = _module
data_loaders = _module
diffusion = _module
diffusion_onnx = _module
dpm_solver_pytorch = _module
infer_gt_mel = _module
logger = _module
saver = _module
utils = _module
onnx_export = _module
solver = _module
uni_pc = _module
unit2mel = _module
vocoder = _module
wavenet = _module
tts = _module
tts_voices = _module
export_index_for_onnx = _module
flask_api = _module
flask_api_full_song = _module
inference = _module
infer_tool = _module
infer_tool_grad = _module
slicer = _module
inference_main = _module
models = _module
DSConv = _module
CrepeF0Predictor = _module
DioF0Predictor = _module
F0Predictor = _module
FCPEF0Predictor = _module
HarvestF0Predictor = _module
PMF0Predictor = _module
RMVPEF0Predictor = _module
crepe = _module
fcpe = _module
model = _module
nvSTFT = _module
pcmer = _module
rmvpe = _module
constants = _module
deepunet = _module
inference = _module
model = _module
seq = _module
spec = _module
utils = _module
modules = _module
attentions = _module
commons = _module
enhancer = _module
losses = _module
mel_processing = _module
modules = _module
onnx_export = _module
onnx_export_old = _module
model_onnx = _module
model_onnx_speaker_mix = _module
preprocess_flist_config = _module
preprocess_hubert_f0 = _module
pretrain = _module
meta = _module
resample = _module
spkmix = _module
train = _module
train_diff = _module
train_index = _module
utils = _module
vdecoder = _module
env = _module
models = _module
nvSTFT = _module
utils = _module
alias = _module
act = _module
filter = _module
resample = _module
models = _module
nvSTFT = _module
utils = _module
models = _module
nvSTFT = _module
utils = _module
CNHubertLarge = _module
ContentVec256L12_Onnx = _module
ContentVec256L9 = _module
ContentVec256L9_Onnx = _module
ContentVec768L12 = _module
ContentVec768L12_Onnx = _module
ContentVec768L9_Onnx = _module
DPHubert = _module
HubertSoft = _module
HubertSoft_Onnx = _module
WavLMBasePlus = _module
WhisperPPG = _module
WhisperPPGLarge = _module
vencoder = _module
dphubert = _module
components = _module
hardconcrete = _module
model = _module
pruning_utils = _module
import_huggingface_wavlm = _module
encoder = _module
hubert = _module
hubert_model = _module
hubert_model_onnx = _module
WavLM = _module
modules = _module
whisper = _module
audio = _module
decoding = _module
model = _module
tokenizer = _module
wav_upload = _module
webUI = _module

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


import torch


from sklearn.cluster import KMeans


from time import time


import numpy as np


from torch.nn.functional import normalize


import logging


import time


from sklearn.cluster import MiniBatchKMeans


from collections import OrderedDict


import random


import torch.utils.data


from torch.utils.data import Dataset


from collections import deque


from functools import partial


from inspect import isfunction


import torch.nn.functional as F


from torch import nn


import math


from torch.nn import Conv1d


from torch.nn import Mish


import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter


import torch.nn as nn


from torch import autocast


from torch.cuda.amp import GradScaler


from torchaudio.transforms import Resample


from math import sqrt


import torchaudio


from torch.nn import Conv2d


from torch.nn import functional as F


from torch.nn.utils import spectral_norm


from torch.nn.utils import weight_norm


from torch.nn.utils import remove_weight_norm


from typing import Union


from typing import Optional


from functools import reduce


from torch.nn.modules.module import _addindent


from random import shuffle


import torch.multiprocessing as mp


import torch.distributed as dist


from torch.cuda.amp import autocast


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.data import DataLoader


from torch.optim import lr_scheduler


import re


from scipy.io.wavfile import read


from torch.nn import AvgPool1d


from torch.nn import ConvTranspose1d


import matplotlib.pylab as plt


from torch import pow


from torch import sin


from torch.nn import Parameter


import matplotlib


from collections import defaultdict


from typing import List


from typing import Tuple


from torch import Tensor


from torch.nn import Module


from typing import Any


from typing import Dict


import copy


import torch.nn.functional as t_func


from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


from torch.nn import LayerNorm


import warnings


from functools import lru_cache


from typing import TYPE_CHECKING


from typing import Iterable


from typing import Sequence


from torch.distributions import Categorical


from itertools import chain


class AfterDiffusion(nn.Module):

    def __init__(self, spec_max, spec_min, v_type='a'):
        super().__init__()
        self.spec_max = spec_max
        self.spec_min = spec_min
        self.type = v_type

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        mel_out = (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
        if self.type == 'nsf-hifigan-log10':
            mel_out = mel_out * 0.434294
        return mel_out.transpose(2, 1)


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]


class DPM_Solver:

    def __init__(self, model_fn, noise_schedule, algorithm_type='dpmsolver++', correcting_x0_fn=None, correcting_xt_fn=None, thresholding_max_val=1.0, dynamic_thresholding_ratio=0.995):
        """Construct a DPM-Solver. 

        We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).

        We also support the "dynamic thresholding" method in Imagen[1]. For pixel-space diffusion models, you
        can set both `algorithm_type="dpmsolver++"` and `correcting_x0_fn="dynamic_thresholding"` to use the
        dynamic thresholding. The "dynamic thresholding" can greatly improve the sample quality for pixel-space
        DPMs with large guidance scales. Note that the thresholding method is **unsuitable** for latent-space
        DPMs (such as stable-diffusion).

        To support advanced algorithms in image-to-image applications, we also support corrector functions for
        both x0 and xt.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
                The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".
            correcting_x0_fn: A `str` or a function with the following format:
                ```
                def correcting_x0_fn(x0, t):
                    x0_new = ...
                    return x0_new
                ```
                This function is to correct the outputs of the data prediction model at each sampling step. e.g.,
                ```
                x0_pred = data_pred_model(xt, t)
                if correcting_x0_fn is not None:
                    x0_pred = correcting_x0_fn(x0_pred, t)
                xt_1 = update(x0_pred, xt, t)
                ```
                If `correcting_x0_fn="dynamic_thresholding"`, we use the dynamic thresholding proposed in Imagen[1].
            correcting_xt_fn: A function with the following format:
                ```
                def correcting_xt_fn(xt, t, step):
                    x_new = ...
                    return x_new
                ```
                This function is to correct the intermediate samples xt at each sampling step. e.g.,
                ```
                xt = ...
                xt = correcting_xt_fn(xt, t, step)
                ```
            thresholding_max_val: A `float`. The max value for thresholding.
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
            dynamic_thresholding_ratio: A `float`. The ratio for dynamic thresholding (see Imagen[1] for details).
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,
            Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models
            with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ['dpmsolver', 'dpmsolver++']
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == 'dynamic_thresholding':
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method. 
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model. 
        """
        if self.algorithm_type == 'dpmsolver++':
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3] * (K - 1) + [1]
            else:
                orders = [3] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2] * K
            else:
                K = steps // 2 + 1
                orders = [2] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0] + orders), 0)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        if self.algorithm_type == 'dpmsolver++':
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)
        if self.algorithm_type == 'dpmsolver++':
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = sigma_s1 / sigma_s * x - alpha_s1 * phi_11 * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s - 0.5 / r1 * (alpha_t * phi_1) * (model_s1 - model_s)
            elif solver_type == 'taylor':
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s + 1.0 / r1 * (alpha_t * (phi_1 / h + 1.0)) * (model_s1 - model_s)
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == 'dpmsolver':
                x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s - 0.5 / r1 * (sigma_t * phi_1) * (model_s1 - model_s)
            elif solver_type == 'taylor':
                x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s - 1.0 / r1 * (sigma_t * (phi_1 / h - 1.0)) * (model_s1 - model_s)
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(self, x, s, t, r1=1.0 / 3.0, r2=2.0 / 3.0, model_s=None, model_s1=None, return_intermediate=False, solver_type='dpmsolver'):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(s1), ns.marginal_log_mean_coeff(s2), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)
        if self.algorithm_type == 'dpmsolver++':
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.0
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = sigma_s1 / sigma_s * x - alpha_s1 * phi_11 * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = sigma_s2 / sigma_s * x - alpha_s2 * phi_12 * model_s + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s + 1.0 / r2 * (alpha_t * phi_2) * (model_s2 - model_s)
            elif solver_type == 'taylor':
                D1_0 = 1.0 / r1 * (model_s1 - model_s)
                D1_1 = 1.0 / r2 * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s + alpha_t * phi_2 * D1 - alpha_t * phi_3 * D2
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - sigma_s1 * phi_11 * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = torch.exp(log_alpha_s2 - log_alpha_s) * x - sigma_s2 * phi_12 * model_s - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == 'dpmsolver':
                x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s - 1.0 / r2 * (sigma_t * phi_2) * (model_s2 - model_s)
            elif solver_type == 'taylor':
                D1_0 = 1.0 / r1 * (model_s1 - model_s)
                D1_1 = 1.0 / r2 * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = torch.exp(log_alpha_t - log_alpha_s) * x - sigma_t * phi_1 * model_s - sigma_t * phi_2 * D1 - sigma_t * phi_3 * D2
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ['dpmsolver', 'taylor']:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = 1.0 / r0 * (model_prev_0 - model_prev_1)
        if self.algorithm_type == 'dpmsolver++':
            phi_1 = torch.expm1(-h)
            if solver_type == 'dpmsolver':
                x_t = sigma_t / sigma_prev_0 * x - alpha_t * phi_1 * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
            elif solver_type == 'taylor':
                x_t = sigma_t / sigma_prev_0 * x - alpha_t * phi_1 * model_prev_0 + alpha_t * (phi_1 / h + 1.0) * D1_0
        else:
            phi_1 = torch.expm1(h)
            if solver_type == 'dpmsolver':
                x_t = torch.exp(log_alpha_t - log_alpha_prev_0) * x - sigma_t * phi_1 * model_prev_0 - 0.5 * (sigma_t * phi_1) * D1_0
            elif solver_type == 'taylor':
                x_t = torch.exp(log_alpha_t - log_alpha_prev_0) * x - sigma_t * phi_1 * model_prev_0 - sigma_t * (phi_1 / h - 1.0) * D1_0
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='dpmsolver'):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_2), ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = 1.0 / r0 * (model_prev_0 - model_prev_1)
        D1_1 = 1.0 / r1 * (model_prev_1 - model_prev_2)
        D1 = D1_0 + r0 / (r0 + r1) * (D1_0 - D1_1)
        D2 = 1.0 / (r0 + r1) * (D1_0 - D1_1)
        if self.algorithm_type == 'dpmsolver++':
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = sigma_t / sigma_prev_0 * x - alpha_t * phi_1 * model_prev_0 + alpha_t * phi_2 * D1 - alpha_t * phi_3 * D2
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = torch.exp(log_alpha_t - log_alpha_prev_0) * x - sigma_t * phi_1 * model_prev_0 - sigma_t * phi_2 * D1 - sigma_t * phi_3 * D2
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='dpmsolver', r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError('Solver order must be 1 or 2 or 3, got {}'.format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError('Solver order must be 1 or 2 or 3, got {}'.format(order))

    def dpm_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-05, solver_type='dpmsolver'):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the 
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,))
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s))
        h = h_init * torch.ones_like(s)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5

            def lower_update(x, s, t):
                return self.dpm_solver_first_update(x, s, t, return_intermediate=True)

            def higher_update(x, s, t, **kwargs):
                return self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1.0 / 3.0, 2.0 / 3.0

            def lower_update(x, s, t):
                return self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)

            def higher_update(x, s, t, **kwargs):
                return self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError('For adaptive step size solver, order must be 2 or 3, got {}'.format(order))
        while torch.abs(s - t_0).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))

            def norm_fn(v):
                return torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.0):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1.0 / order).float(), lambda_0 - lambda_s)
            nfe += order
        None
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver', atol=0.0078, rtol=0.05, return_intermediate=False):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, 'Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array'
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type, method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type, atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform', method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver', atol=0.0078, rtol=0.05, return_intermediate=False):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver. 
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, 'Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array'
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], 'Cannot use adaptive solver when saving intermediate values'
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], 'Cannot use adaptive solver when correcting_xt_fn is not None'
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'adaptive':
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            elif method == 'multistep':
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError('Got wrong method {}'.format(method))
            if denoise_to_zero:
                t = torch.ones((1,)) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x


class ResidualBlock(nn.Module):

    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffNet(nn.Module):

    def __init__(self, in_dims, n_layers, n_chans, n_hidden):
        super().__init__()
        self.encoder_hidden = n_hidden
        self.residual_layers = n_layers
        self.residual_channels = n_chans
        self.input_projection = Conv1d(in_dims, self.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        dim = self.residual_channels
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim))
        self.residual_layers = nn.ModuleList([ResidualBlock(self.encoder_hidden, self.residual_channels, 1) for i in range(self.residual_layers)])
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        self.output_projection = Conv1d(self.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        x = spec.squeeze(0)
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = diffusion_step.float()
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        x, skip = self.residual_layers[0](x, cond, diffusion_step)
        for layer in self.residual_layers[1:]:
            x, skip_connection = layer.forward(x, cond, diffusion_step)
            skip = skip + skip_connection
        x = skip / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x.unsqueeze(1)


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(torch.eq(x_idx, 0), torch.tensor(1, device=x.device), torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx))
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(torch.eq(x_idx, 0), torch.tensor(0, device=x.device), torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx))
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


class NoiseScheduleVP:

    def __init__(self, schedule='discrete', betas=None, alphas_cumprod=None, continuous_beta_0=0.1, continuous_beta_1=20.0, dtype=torch.float32):
        """Create a wrapper class for the forward SDE (VP type).
        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***
        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)
        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
            t = self.inverse_lambda(lambda_t)
        ===============================================================
        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).
        1. For discrete-time DPMs:
            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.
            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)
            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.
            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \\hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \\sqrt{\\hat{alpha_n}} * x_0, (1 - \\hat{alpha_n}) * I ).
                Therefore, the notation \\hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \\sqrt{\\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\\hat{alpha_n}).
        2. For continuous-time DPMs:
            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:
            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.
        ===============================================================
        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        
        ===============================================================
        Example:
        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)
        # For discrete-time DPMs, given alphas_cumprod (the \\hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
        """
        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(schedule))
        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1))
            self.log_alpha_array = log_alphas.reshape((1, -1))
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi) * 2.0 * (1.0 + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.schedule = schedule
            if schedule == 'cosine':
                self.T = 0.9946
            else:
                self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array, self.log_alpha_array).reshape(-1)
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':

            def log_alpha_fn(s):
                return torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)), -2.0 * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array, [1]), torch.flip(self.t_array, [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)))

            def t_fn(log_alpha_t):
                return torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2.0 * (1.0 + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))


class Pred(nn.Module):

    def __init__(self, alphas_cumprod):
        super().__init__()
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x_1, noise_t, t_1, t_prev):
        a_t = extract(self.alphas_cumprod, t_1).cpu()
        a_prev = extract(self.alphas_cumprod, t_prev).cpu()
        a_t_sq, a_prev_sq = a_t.sqrt().cpu(), a_prev.sqrt().cpu()
        x_delta = (a_prev - a_t) * (1 / (a_t_sq * (a_t_sq + a_prev_sq)) * x_1 - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x_1 + x_delta.cpu()
        return x_pred


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos((x / steps + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, a_min=0, a_max=0.999)


def linear_beta_schedule(timesteps, max_beta=0.02):
    """
    linear schedule
    """
    betas = np.linspace(0.0001, max_beta, timesteps)
    return betas


beta_schedule = {'cosine': cosine_beta_schedule, 'linear': linear_beta_schedule}


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def model_wrapper(model, noise_schedule, model_type='noise', model_kwargs={}, guidance_type='uncond', condition=None, unconditional_condition=None, guidance_scale=1.0, classifier_fn=None, classifier_kwargs={}):
    """Create a wrapper function for the noise prediction model.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1.0 / noise_schedule.total_N) * noise_schedule.total_N
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == 'noise':
            return output
        elif model_type == 'x_start':
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == 'v':
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == 'score':
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == 'uncond':
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == 'classifier':
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == 'classifier-free':
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)
    assert model_type in ['noise', 'x_start', 'v']
    assert guidance_type in ['uncond', 'classifier', 'classifier-free']
    return model_fn


def noise_like(shape, device, repeat=False):

    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise():
        return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def predict_stage0(noise_pred, noise_pred_prev):
    return (noise_pred + noise_pred_prev) / 2


def predict_stage1(noise_pred, noise_list):
    return (noise_pred * 3 - noise_list[-1]) / 2


def predict_stage2(noise_pred, noise_list):
    return (noise_pred * 23 - noise_list[-1] * 16 + noise_list[-2] * 5) / 12


def predict_stage3(noise_pred, noise_list):
    return (noise_pred * 55 - noise_list[-1] * 59 + noise_list[-2] * 37 - noise_list[-3] * 9) / 24


class GaussianDiffusion(nn.Module):

    def __init__(self, out_dims=128, n_layers=20, n_chans=384, n_hidden=256, timesteps=1000, k_step=1000, max_beta=0.02, spec_min=-12, spec_max=2):
        super().__init__()
        self.denoise_fn = DiffNet(out_dims, n_layers, n_chans, n_hidden)
        self.out_dims = out_dims
        self.mel_bins = out_dims
        self.n_hidden = n_hidden
        betas = beta_schedule['linear'](timesteps, max_beta=max_beta)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.k_step = k_step
        self.noise_list = deque(maxlen=4)
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))
        self.register_buffer('spec_min', torch.FloatTensor([spec_min])[None, None, :out_dims])
        self.register_buffer('spec_max', torch.FloatTensor([spec_max])[None, None, :out_dims])
        self.ad = AfterDiffusion(self.spec_max, self.spec_min)
        self.xp = Pred(self.alphas_cumprod)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)))
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
            x_delta = (a_prev - a_t) * (1 / (a_t_sq * (a_t_sq + a_prev_sq)) * x - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta
            return x_pred
        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)
        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24
        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)
        return x_prev

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, t, cond, noise=None, loss_type='l2'):
        noise = default(noise, lambda : torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)
        if loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()
        return loss

    def org_forward(self, condition, init_noise=None, gt_spec=None, infer=True, infer_speedup=100, method='pndm', k_step=1000, use_tqdm=True):
        """
            conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        cond = condition
        b, device = condition.shape[0], condition.device
        if not infer:
            spec = self.norm_spec(gt_spec)
            t = torch.randint(0, self.k_step, (b,), device=device).long()
            norm_spec = spec.transpose(1, 2)[:, None, :, :]
            return self.p_losses(norm_spec, t, cond=cond)
        else:
            shape = cond.shape[0], 1, self.out_dims, cond.shape[2]
            if gt_spec is None:
                t = self.k_step
                if init_noise is None:
                    x = torch.randn(shape, device=device)
                else:
                    x = init_noise
            else:
                t = k_step
                norm_spec = self.norm_spec(gt_spec)
                norm_spec = norm_spec.transpose(1, 2)[:, None, :, :]
                x = self.q_sample(x_start=norm_spec, t=torch.tensor([t - 1], device=device).long())
            if method is not None and infer_speedup > 1:
                if method == 'dpm-solver':
                    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t])

                    def my_wrapper(fn):

                        def wrapped(x, t, **kwargs):
                            ret = fn(x, t, **kwargs)
                            if use_tqdm:
                                self.bar.update(1)
                            return ret
                        return wrapped
                    model_fn = model_wrapper(my_wrapper(self.denoise_fn), noise_schedule, model_type='noise', model_kwargs={'cond': cond})
                    dpm_solver = DPM_Solver(model_fn, noise_schedule)
                    steps = t // infer_speedup
                    if use_tqdm:
                        self.bar = tqdm(desc='sample time step', total=steps)
                    x = dpm_solver.sample(x, steps=steps, order=3, skip_type='time_uniform', method='singlestep')
                    if use_tqdm:
                        self.bar.close()
                elif method == 'pndm':
                    self.noise_list = deque(maxlen=4)
                    if use_tqdm:
                        for i in tqdm(reversed(range(0, t, infer_speedup)), desc='sample time step', total=t // infer_speedup):
                            x = self.p_sample_plms(x, torch.full((b,), i, device=device, dtype=torch.long), infer_speedup, cond=cond)
                    else:
                        for i in reversed(range(0, t, infer_speedup)):
                            x = self.p_sample_plms(x, torch.full((b,), i, device=device, dtype=torch.long), infer_speedup, cond=cond)
                else:
                    raise NotImplementedError(method)
            elif use_tqdm:
                for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
                    x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            else:
                for i in reversed(range(0, t)):
                    x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
            x = x.squeeze(1).transpose(1, 2)
            return self.denorm_spec(x).transpose(2, 1)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def get_x_pred(self, x_1, noise_t, t_1, t_prev):
        a_t = extract(self.alphas_cumprod, t_1)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
        x_delta = (a_prev - a_t) * (1 / (a_t_sq * (a_t_sq + a_prev_sq)) * x_1 - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x_1 + x_delta
        return x_pred

    def OnnxExport(self, project_name=None, init_noise=None, hidden_channels=256, export_denoise=True, export_pred=True, export_after=True):
        cond = torch.randn([1, self.n_hidden, 10]).cpu()
        if init_noise is None:
            x = torch.randn((1, 1, self.mel_bins, cond.shape[2]), dtype=torch.float32).cpu()
        else:
            x = init_noise
        pndms = 100
        org_y_x = self.org_forward(cond, init_noise=x)
        device = cond.device
        n_frames = cond.shape[2]
        step_range = torch.arange(0, self.k_step, pndms, dtype=torch.long, device=device).flip(0)
        plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
        noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)
        ot = step_range[0]
        ot_1 = torch.full((1,), ot, device=device, dtype=torch.long)
        if export_denoise:
            torch.onnx.export(self.denoise_fn, (x.cpu(), ot_1.cpu(), cond.cpu()), f'{project_name}_denoise.onnx', input_names=['noise', 'time', 'condition'], output_names=['noise_pred'], dynamic_axes={'noise': [3], 'condition': [2]}, opset_version=16)
        for t in step_range:
            t_1 = torch.full((1,), t, device=device, dtype=torch.long)
            noise_pred = self.denoise_fn(x, t_1, cond)
            t_prev = t_1 - pndms
            t_prev = t_prev * (t_prev > 0)
            if plms_noise_stage == 0:
                if export_pred:
                    torch.onnx.export(self.xp, (x.cpu(), noise_pred.cpu(), t_1.cpu(), t_prev.cpu()), f'{project_name}_pred.onnx', input_names=['noise', 'noise_pred', 'time', 'time_prev'], output_names=['noise_pred_o'], dynamic_axes={'noise': [3], 'noise_pred': [3]}, opset_version=16)
                x_pred = self.get_x_pred(x, noise_pred, t_1, t_prev)
                noise_pred_prev = self.denoise_fn(x_pred, t_prev, cond=cond)
                noise_pred_prime = predict_stage0(noise_pred, noise_pred_prev)
            elif plms_noise_stage == 1:
                noise_pred_prime = predict_stage1(noise_pred, noise_list)
            elif plms_noise_stage == 2:
                noise_pred_prime = predict_stage2(noise_pred, noise_list)
            else:
                noise_pred_prime = predict_stage3(noise_pred, noise_list)
            noise_pred = noise_pred.unsqueeze(0)
            if plms_noise_stage < 3:
                noise_list = torch.cat((noise_list, noise_pred), dim=0)
                plms_noise_stage = plms_noise_stage + 1
            else:
                noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)
            x = self.get_x_pred(x, noise_pred_prime, t_1, t_prev)
        if export_after:
            torch.onnx.export(self.ad, x.cpu(), f'{project_name}_after.onnx', input_names=['x'], output_names=['mel_out'], dynamic_axes={'x': [3]}, opset_version=16)
        x = self.ad(x)
        None
        return x

    def forward(self, condition=None, init_noise=None, pndms=None, k_step=None):
        cond = condition
        x = init_noise
        device = cond.device
        n_frames = cond.shape[2]
        step_range = torch.arange(0, k_step.item(), pndms.item(), dtype=torch.long, device=device).flip(0)
        plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
        noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)
        for t in step_range:
            t_1 = torch.full((1,), t, device=device, dtype=torch.long)
            noise_pred = self.denoise_fn(x, t_1, cond)
            t_prev = t_1 - pndms
            t_prev = t_prev * (t_prev > 0)
            if plms_noise_stage == 0:
                x_pred = self.get_x_pred(x, noise_pred, t_1, t_prev)
                noise_pred_prev = self.denoise_fn(x_pred, t_prev, cond=cond)
                noise_pred_prime = predict_stage0(noise_pred, noise_pred_prev)
            elif plms_noise_stage == 1:
                noise_pred_prime = predict_stage1(noise_pred, noise_list)
            elif plms_noise_stage == 2:
                noise_pred_prime = predict_stage2(noise_pred, noise_list)
            else:
                noise_pred_prime = predict_stage3(noise_pred, noise_list)
            noise_pred = noise_pred.unsqueeze(0)
            if plms_noise_stage < 3:
                noise_list = torch.cat((noise_list, noise_pred), dim=0)
                plms_noise_stage = plms_noise_stage + 1
            else:
                noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)
            x = self.get_x_pred(x, noise_pred_prime, t_1, t_prev)
        x = self.ad(x)
        return x


class WaveNet(nn.Module):

    def __init__(self, in_dims=128, n_layers=20, n_chans=384, n_hidden=256):
        super().__init__()
        self.input_projection = Conv1d(in_dims, n_chans, 1)
        self.diffusion_embedding = SinusoidalPosEmb(n_chans)
        self.mlp = nn.Sequential(nn.Linear(n_chans, n_chans * 4), Mish(), nn.Linear(n_chans * 4, n_chans))
        self.residual_layers = nn.ModuleList([ResidualBlock(encoder_hidden=n_hidden, residual_channels=n_chans, dilation=1) for i in range(n_layers)])
        self.skip_projection = Conv1d(n_chans, n_chans, 1)
        self.output_projection = Conv1d(n_chans, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)
        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x[:, None, :, :]


class Unit2Mel(nn.Module):

    def __init__(self, input_channel, n_spk, use_pitch_aug=False, out_dims=128, n_layers=20, n_chans=384, n_hidden=256, timesteps=1000, k_step_max=1000):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        self.timesteps = timesteps if timesteps is not None else 1000
        self.k_step_max = k_step_max if k_step_max is not None and k_step_max > 0 and k_step_max < self.timesteps else self.timesteps
        self.n_hidden = n_hidden
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), timesteps=self.timesteps, k_step=self.k_step_max, out_dims=out_dims)
        self.input_channel = input_channel

    def init_spkembed(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        """
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        """
        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                spk_embed_mix = torch.zeros((1, 1, self.hidden_size))
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]]))
                    spk_embeddd = self.spk_embed(spk_id_torch)
                    self.speaker_map[k] = spk_embeddd
                    spk_embed_mix = spk_embed_mix + v * spk_embeddd
                x = x + spk_embed_mix
            else:
                x = x + self.spk_embed(spk_id - 1)
        self.speaker_map = self.speaker_map.unsqueeze(0)
        self.speaker_map = self.speaker_map.detach()
        return x.transpose(1, 2)

    def init_spkmix(self, n_spk):
        self.speaker_map = torch.zeros((n_spk, 1, 1, self.n_hidden))
        hubert_hidden_size = self.input_channel
        n_frames = 10
        hubert = torch.randn((1, n_frames, hubert_hidden_size))
        f0 = torch.randn((1, n_frames))
        volume = torch.randn((1, n_frames))
        spks = {}
        for i in range(n_spk):
            spks.update({i: 1.0 / float(self.n_spk)})
        self.init_spkembed(hubert, f0.unsqueeze(-1), volume.unsqueeze(-1), spk_mix_dict=spks)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=300, use_tqdm=True):
        """
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        """
        if not self.training and gt_spec is not None and k_step > self.k_step_max:
            raise Exception('The shallow diffusion k_step is greater than the maximum diffusion k_step(k_step_max)!')
        if not self.training and gt_spec is None and self.k_step_max != self.timesteps:
            raise Exception('This model can only be used for shallow diffusion and can not infer alone!')
        x = self.unit_embed(units) + self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]]))
                    x = x + v * self.spk_embed(spk_id_torch)
            elif spk_id.shape[1] > 1:
                g = spk_id.reshape((spk_id.shape[0], spk_id.shape[1], 1, 1, 1))
                g = g * self.speaker_map
                g = torch.sum(g, dim=1)
                g = g.transpose(0, -1).transpose(0, -2).squeeze(0)
                x = x + g
            else:
                x = x + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x, gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
        return x


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)
    except Exception as ex:
        None
        None
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)
    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(data) > 2
    if np.issubdtype(data.dtype, np.integer):
        max_mag = -np.iinfo(data.dtype).min
    else:
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = 2 ** 31 + 1 if max_mag > 2 ** 15 else 2 ** 15 + 1 if max_mag > 1.01 else 1.0
    data = torch.FloatTensor(data.astype(np.float32)) / max_mag
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    return data, sampling_rate


class STFT:

    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-05):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False):
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        if torch.min(y) < -1.0:
            None
        if torch.max(y) > 1.0:
            None
        mel_basis_key = str(fmax) + '_' + str(y.device)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float()
        keyshift_key = str(keyshift) + '_' + str(y.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_size_new)
        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
        if pad_right < y.size(-1):
            mode = 'reflect'
        else:
            mode = 'constant'
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode=mode)
        y = y.squeeze(1)
        spec = torch.stft(y, n_fft_new, hop_length=hop_length_new, win_length=win_size_new, window=self.hann_window[keyshift_key], center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-09)
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new
        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        return spec

    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1])))])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    @torch.no_grad()
    def forward(self, f0, upp):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        fn = torch.multiply(f0, torch.arange(1, self.dim + 1, device=f0.device).reshape((1, 1, -1)))
        rad_values = fn / self.sampling_rate % 1
        rand_ini = torch.rand(fn.shape[0], fn.shape[2], device=fn.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        is_half = rad_values.dtype is not torch.float32
        tmp_over_one = torch.cumsum(rad_values.double(), 1)
        if is_half:
            tmp_over_one = tmp_over_one.half()
        else:
            tmp_over_one = tmp_over_one.float()
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(tmp_over_one.transpose(2, 1), scale_factor=upp, mode='linear', align_corners=True).transpose(2, 1)
        rad_values = F.interpolate(rad_values.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        tmp_over_one %= 1
        tmp_over_one_idx = tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :] < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        rad_values = rad_values.double()
        cumsum_shift = cumsum_shift.double()
        sine_waves = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        if is_half:
            sine_waves = sine_waves.half()
        else:
            sine_waves = sine_waves.float()
        sine_waves = sine_waves * self.sine_amp
        uv = self._f02uv(f0)
        uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge


class Generator(torch.nn.Module):

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=8)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // 2 ** (i + 1)
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel // 2 ** i, h.upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2)))
            if i + 1 < len(h.upsample_rates):
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x, f0):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(model_path):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h


def load_model(model_path, device='cuda'):
    h = load_config(model_path)
    generator = Generator(h)
    cp_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(cp_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h


class NsfHifiGAN(torch.nn.Module):

    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        None
        self.model, self.h = load_model(model_path, device=self.device)

    def sample_rate(self):
        return self.h.sampling_rate

    def hop_size(self):
        return self.h.hop_size

    def forward(self, audio, f0):
        stft = STFT(self.h.sampling_rate, self.h.num_mels, self.h.n_fft, self.h.win_size, self.h.hop_size, self.h.fmin, self.h.fmax)
        with torch.no_grad():
            mel = stft.get_mel(audio)
            enhanced_audio = self.model(mel, f0[:, :mel.size(-1)]).view(-1)
            return enhanced_audio, self.h.sampling_rate


class NsfHifiGANLog10(NsfHifiGAN):

    def forward(self, mel, f0):
        if self.model is None:
            None
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio


class Conv1d(nn.Conv1d):

    def _conv_forward(self, x: 'Tensor', weight: 'Tensor', bias: 'Optional[Tensor]') ->Tensor:
        return super()._conv_forward(x, weight, None if bias is None else bias)


class ResidualCouplingBlock(nn.Module):

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0, share_parameter=False):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = nn.ModuleList()
        self.wn = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=gin_channels) if share_parameter else None
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, wn_sharing_parameter=self.wn))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class TransformerCouplingBlock(nn.Module):

    def __init__(self, channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, n_flows=4, gin_channels=0, share_parameter=False):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = nn.ModuleList()
        self.wn = attentions.FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, isflow=True, gin_channels=self.gin_channels) if share_parameter else None
        for i in range(n_flows):
            self.flows.append(modules.TransformerCouplingLayer(channels, hidden_channels, kernel_size, n_layers, n_heads, p_dropout, filter_channels, mean_only=True, wn_sharing_parameter=self.wn, gin_channels=self.gin_channels))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


def prune_layer_norm(layernorm: 'Union[nn.LayerNorm, nn.GroupNorm]', index: 'torch.LongTensor'):
    """Prune layer norm or group norm in place."""
    layernorm.weight = nn.Parameter(layernorm.weight.index_select(0, index).clone().detach())
    layernorm.bias = nn.Parameter(layernorm.bias.index_select(0, index).clone().detach())
    if isinstance(layernorm, nn.LayerNorm):
        layernorm.normalized_shape = len(index),
    elif isinstance(layernorm, nn.GroupNorm):
        layernorm.num_groups = len(index)
        layernorm.num_channels = len(index)


def prune_linear_layer(layer: 'nn.Linear', index: 'torch.LongTensor', dim: 'str'):
    """Prune linear layer in place."""
    if dim == 'input':
        dim = 1
        layer.in_features = len(index)
    elif dim == 'output':
        dim = 0
        layer.out_features = len(index)
    else:
        raise ValueError
    layer.weight = nn.Parameter(layer.weight.index_select(dim, index).clone().detach())
    if layer.bias is not None and dim == 0:
        layer.bias = nn.Parameter(layer.bias.index_select(0, index).clone().detach())


class Encoder(Module):

    def __init__(self, feature_projection: 'Module', transformer: 'Module'):
        super().__init__()
        self.feature_projection = feature_projection
        self.transformer = transformer

    def _preprocess(self, features: 'Tensor', lengths: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)
        mask: 'Optional[Tensor]' = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            mask = -10000.0 * mask[:, None, None, :]
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    def forward(self, features: 'Tensor', lengths: 'Optional[Tensor]'=None) ->Tensor:
        x, mask = self._preprocess(features, lengths)
        x = self.transformer(x, attention_mask=mask)
        return x

    def extract_features(self, features: 'Tensor', lengths: 'Optional[Tensor]'=None, num_layers: 'Optional[int]'=None) ->List[Tensor]:
        x, masks = self._preprocess(features, lengths)
        interm = self.transformer.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)
        return [x] + interm

    def get_num_params(self, in_features):
        """Calculate the current model size."""
        feature_projection_size = self.feature_projection.get_num_params(in_features)
        transformer_size = self.transformer.get_num_params()
        return feature_projection_size + transformer_size

    def prune(self, conv_out_index):
        """In-place pruning of submodules."""
        prune_layer_norm(self.feature_projection.layer_norm, conv_out_index)
        prune_linear_layer(self.feature_projection.projection, conv_out_index, 'input')
        transformer_config = self.transformer.prune()
        return transformer_config


class TextEncoder(nn.Module):

    def __init__(self, out_channels, hidden_channels, kernel_size, n_layers, gin_channels=0, filter_channels=None, n_heads=None, p_dropout=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)
        self.enc_ = attentions.Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)

    def forward(self, x, x_mask, f0=None, z=None):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + z * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class DiscriminatorP(torch.nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))), norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([norm_f(Conv1d(1, 128, 15, 1, padding=7)), norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)), norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)), norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)), norm_f(Conv1d(1024, 1024, 5, 1, padding=2))])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpeakerEncoder(torch.nn.Module):

    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)
        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]
        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)
            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
        else:
            with torch.no_grad():
                embed = self(last_mel)
        return embed


class F0Decoder(nn.Module):

    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels
        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions.FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if spk_emb is not None:
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


f0_bin = 256


f0_max = 1100.0


f0_mel_max = 1127 * np.log(1 + f0_max / 700)


f0_min = 50.0


f0_mel_min = 1127 * np.log(1 + f0_min / 700)


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + (f0_coarse < 1) * 1
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + (f0_coarse >= f0_bin) * (f0_bin - 1)
    return f0_coarse


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels, ssl_dim, n_speakers, sampling_rate=44100, vol_embedding=False, vocoder_name='nsf-hifigan', use_depthwise_conv=False, use_automatic_f0_prediction=True, flow_share_parameter=False, n_flow_layer=4, n_layers_trans_flow=3, use_transformer_flow=False, **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.n_layers_trans_flow = n_layers_trans_flow
        if vol_embedding:
            self.emb_vol = nn.Linear(1, hidden_channels)
        self.pre = nn.Conv1d(ssl_dim, hidden_channels, kernel_size=5, padding=2)
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels=filter_channels, n_heads=n_heads, n_layers=n_layers, kernel_size=kernel_size, p_dropout=p_dropout)
        hps = {'sampling_rate': sampling_rate, 'inter_channels': inter_channels, 'resblock': resblock, 'resblock_kernel_sizes': resblock_kernel_sizes, 'resblock_dilation_sizes': resblock_dilation_sizes, 'upsample_rates': upsample_rates, 'upsample_initial_channel': upsample_initial_channel, 'upsample_kernel_sizes': upsample_kernel_sizes, 'gin_channels': gin_channels, 'use_depthwise_conv': use_depthwise_conv}
        modules.set_Conv1dModel(self.use_depthwise_conv)
        if vocoder_name == 'nsf-hifigan':
            self.dec = Generator(h=hps)
        elif vocoder_name == 'nsf-snake-hifigan':
            self.dec = Generator(h=hps)
        else:
            None
            self.dec = Generator(h=hps)
        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        if use_transformer_flow:
            self.flow = TransformerCouplingBlock(inter_channels, hidden_channels, filter_channels, n_heads, n_layers_trans_flow, 5, p_dropout, n_flow_layer, gin_channels=gin_channels, share_parameter=flow_share_parameter)
        else:
            self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, n_flow_layer, gin_channels=gin_channels, share_parameter=flow_share_parameter)
        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(1, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, spk_channels=gin_channels)
        self.emb_uv = nn.Embedding(2, hidden_channels)
        self.predict_f0 = False
        self.speaker_map = []
        self.export_mix = False

    def export_chara_mix(self, speakers_mix):
        self.speaker_map = torch.zeros((len(speakers_mix), 1, 1, self.gin_channels))
        i = 0
        for key in speakers_mix.keys():
            spkidx = speakers_mix[key]
            self.speaker_map[i] = self.emb_g(torch.LongTensor([[spkidx]]))
            i = i + 1
        self.speaker_map = self.speaker_map.unsqueeze(0)
        self.export_mix = True

    def forward(self, c, f0, mel2ph, uv, noise=None, g=None, vol=None):
        decoder_inp = F.pad(c, [0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(2).repeat([1, 1, c.shape[-1]])
        c = torch.gather(decoder_inp, 1, mel2ph_).transpose(1, 2)
        if self.export_mix:
            g = g.reshape((g.shape[0], g.shape[1], 1, 1, 1))
            g = g * self.speaker_map
            g = torch.sum(g, dim=1)
            g = g.transpose(0, -1).transpose(0, -2).squeeze(0)
        else:
            if g.dim() == 1:
                g = g.unsqueeze(0)
            g = self.emb_g(g).transpose(1, 2)
        x_mask = torch.unsqueeze(torch.ones_like(f0), 1)
        vol = self.emb_vol(vol[:, :, None]).transpose(1, 2) if vol is not None and self.vol_embedding else 0
        x = self.pre(c) * x_mask + self.emb_uv(uv.long()).transpose(1, 2) + vol
        if self.use_automatic_f0_prediction and self.predict_f0:
            lf0 = 2595.0 * torch.log10(1.0 + f0.unsqueeze(1) / 700.0) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0), z=noise)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o


class Depthwise_Separable_Conv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.depth_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device, dtype=dtype)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name='weight')
        self.point_conv = weight_norm(self.point_conv, name='weight')

    def remove_weight_norm(self):
        self.depth_conv = remove_weight_norm(self.depth_conv, name='weight')
        self.point_conv = remove_weight_norm(self.point_conv, name='weight')


class Depthwise_Separable_TransposeConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.depth_conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, stride=stride, output_padding=output_padding, padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        self.point_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias, device=device, dtype=dtype)

    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

    def weight_norm(self):
        self.depth_conv = weight_norm(self.depth_conv, name='weight')
        self.point_conv = weight_norm(self.point_conv, name='weight')

    def remove_weight_norm(self):
        remove_weight_norm(self.depth_conv, name='weight')
        remove_weight_norm(self.point_conv, name='weight')


class MaskedAvgPool1d(nn.Module):

    def __init__(self, kernel_size: 'int', stride: 'Optional[int]'=None, padding: 'Optional[int]'=0):
        """An implementation of mean pooling that supports masked values.

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """
        super(MaskedAvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3, 'Input tensor must have 2 or 3 dimensions (batch_size, channels, width)'
        if mask is None:
            mask = ~torch.isnan(x)
        assert x.shape == mask.shape, 'Input tensor and mask must have the same shape'
        masked_x = torch.where(mask, x, torch.zeros_like(x))
        ones_kernel = torch.ones(x.size(1), 1, self.kernel_size, device=x.device)
        sum_pooled = nn.functional.conv1d(masked_x, ones_kernel, stride=self.stride, padding=self.padding, groups=x.size(1))
        valid_count = nn.functional.conv1d(mask.float(), ones_kernel, stride=self.stride, padding=self.padding, groups=x.size(1))
        valid_count = valid_count.clamp(min=1)
        avg_pooled = sum_pooled / valid_count
        avg_pooled[avg_pooled == 0] = float('nan')
        if ndim == 2:
            return avg_pooled.squeeze(1)
        return avg_pooled


class MaskedMedianPool1d(nn.Module):

    def __init__(self, kernel_size: 'int', stride: 'Optional[int]'=None, padding: 'Optional[int]'=0):
        """An implementation of median pooling that supports masked values.

        This implementation is inspired by the median pooling implementation in
        https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """
        super(MaskedMedianPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3, 'Input tensor must have 2 or 3 dimensions (batch_size, channels, width)'
        if mask is None:
            mask = ~torch.isnan(x)
        assert x.shape == mask.shape, 'Input tensor and mask must have the same shape'
        masked_x = torch.where(mask, x, torch.zeros_like(x))
        x = F.pad(masked_x, (self.padding, self.padding), mode='reflect')
        mask = F.pad(mask.float(), (self.padding, self.padding), mode='constant', value=0)
        x = x.unfold(2, self.kernel_size, self.stride)
        mask = mask.unfold(2, self.kernel_size, self.stride)
        x = x.contiguous().view(x.size()[:3] + (-1,))
        mask = mask.contiguous().view(mask.size()[:3] + (-1,))
        x_masked = torch.where(mask.bool(), x, torch.FloatTensor([float('inf')]))
        x_sorted, _ = torch.sort(x_masked, dim=-1)
        valid_count = mask.sum(dim=-1)
        median_idx = torch.div(valid_count - 1, 2, rounding_mode='trunc').clamp(min=0)
        median_pooled = x_sorted.gather(-1, median_idx.unsqueeze(-1).long()).squeeze(-1)
        median_pooled[torch.isinf(median_pooled)] = float('nan')
        if ndim == 2:
            return median_pooled.squeeze(1)
        return median_pooled


class DepthWiseConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class GLU(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class Swish(nn.Module):
    """Swish function
    """

    def __init__(self):
        """Construct an MultiHeadedAttention object."""
        super(Swish, self).__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


class Transpose(nn.Module):

    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return pad, pad - (kernel_size + 1) % 2


class ConformerConvModule(nn.Module):

    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.net = nn.Sequential(nn.LayerNorm(dim), Transpose((1, 2)), nn.Conv1d(dim, inner_dim * 2, 1), GLU(dim=1), DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding), Swish(), nn.Conv1d(inner_dim, dim, 1), Transpose((1, 2)), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class HardConcrete(nn.Module):
    """A HarcConcrete module.
    Use this module to create a mask of size N, which you can
    then use to perform L0 regularization.

    To obtain a mask, simply run a forward pass through the module
    with no input data. The mask is sampled in training mode, and
    fixed during evaluation mode, e.g.:

    >>> module = HardConcrete(n_in=100)
    >>> mask = module()
    >>> norm = module.l0_norm()
    """

    def __init__(self, n_in: 'int', init_mean: 'float'=0.5, init_std: 'float'=0.01, temperature: 'float'=2 / 3, stretch: 'float'=0.1, eps: 'float'=1e-06) ->None:
        """Initialize the HardConcrete module.
        Parameters
        ----------
        n_in : int
            The number of hard concrete variables in this mask.
        init_mean : float, optional
            Initial drop rate for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        temperature : float, optional
            Temperature used to control the sharpness of the
            distribution, by default 1.0
        stretch : float, optional
            Stretch the sampled value from [0, 1] to the interval
            [-stretch, 1 + stretch], by default 0.1.
        """
        super().__init__()
        self.n_in = n_in
        self.limit_l = -stretch
        self.limit_r = 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.zeros(n_in))
        self.beta = temperature
        self.init_mean = init_mean
        self.init_std = init_std
        self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)
        self.eps = eps
        self.compiled_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of this module."""
        self.compiled_mask = None
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) ->torch.Tensor:
        """Compute the expected L0 norm of this mask.
        Returns
        -------
        torch.Tensor
            The expected L0 norm.
        """
        return (self.log_alpha + self.bias).sigmoid().sum()

    def forward(self) ->torch.Tensor:
        """Sample a hard concrete mask.
        Returns
        -------
        torch.Tensor
            The sampled binary mask
        """
        if self.training:
            self.compiled_mask = None
            u = self.log_alpha.new(self.n_in).uniform_(self.eps, 1 - self.eps)
            s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            mask = s.clamp(min=0.0, max=1.0)
        else:
            if self.compiled_mask is None:
                expected_num_zeros = self.n_in - self.l0_norm().item()
                num_zeros = round(expected_num_zeros)
                soft_mask = torch.sigmoid(self.log_alpha / self.beta * 0.8)
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.0
                self.compiled_mask = soft_mask
            mask = self.compiled_mask
        return mask

    def extra_repr(self) ->str:
        return str(self.n_in)

    def __repr__(self) ->str:
        return '{}({})'.format(self.__class__.__name__, self.extra_repr())


class SelfAttention(Module):
    """Multihead Self Attention module

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probability on attn_output_weights. Default: ``0.0``
    """

    def __init__(self, embed_dim: 'int', num_heads: 'int', head_dim: 'int', dropout: 'float'=0.0, prune_heads: 'bool'=False, prune_layer: 'bool'=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=True)
        if prune_heads:
            self.hard_concrete_for_heads = HardConcrete(n_in=num_heads, init_mean=0.01)
        else:
            self.hard_concrete_for_heads = None
        if prune_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(self, x: 'Tensor', attention_mask: 'Optional[Tensor]'=None, position_bias: 'Optional[Tensor]'=None, key_padding_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
            position_bias: Not used. Only for the compatibility with :py:class:`WavLMSelfAttention`.
            key_padding_mask (Tensor or ``None``): Not used. Only for the compatibility with
                :py:class:`WavLMSelfAttention`.
        Returns:
            (Tensor, ``None``): The resulting attention output and ``None`` (necessary for compatibility
                with :py:class:`WavLMSelAttention`).
                Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(f'The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). Found {x.shape}.')
        batch_size, length, embed_dim = x.size()
        shape = batch_size, length, self.num_heads, self.head_dim
        q = self.q_proj(x).view(*shape).transpose(2, 1)
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)
        v = self.v_proj(x).view(*shape).transpose(2, 1)
        weights = self.scaling * q @ k
        if attention_mask is not None:
            weights += attention_mask
        weights = weights - weights.max(dim=-1, keepdim=True)[0]
        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v
        if self.hard_concrete_for_heads is not None:
            head_mask = self.hard_concrete_for_heads()
            output = output * head_mask.unsqueeze(-1).unsqueeze(-1)
        output = output.transpose(2, 1).reshape(batch_size, length, self.num_heads * self.head_dim)
        output = self.out_proj(output)
        if self.hard_concrete_for_layer is not None:
            layer_mask = self.hard_concrete_for_layer()
            output = output * layer_mask
        return output, None

    def get_num_params(self):
        if self.hard_concrete_for_heads is not None:
            num_heads = self.hard_concrete_for_heads.l0_norm()
        else:
            num_heads = self.num_heads
        num_params = (self.embed_dim + 1) * num_heads * self.head_dim * 3 + (num_heads * self.head_dim + 1) * self.embed_dim
        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        return num_params

    def prune(self):
        new_config = {'use_attention': True, 'num_heads': self.num_heads}
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer()
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            if layer_mask == 0:
                new_config['use_attention'] = False
            self.hard_concrete_for_layer = None
        if self.hard_concrete_for_heads is not None:
            assert not self.hard_concrete_for_heads.training
            head_mask = self.hard_concrete_for_heads()
            new_config['num_heads'] = len(head_mask.nonzero())
            if new_config['num_heads'] == 0:
                new_config['use_attention'] = False
            else:
                full_mask = head_mask.repeat_interleave(self.head_dim)
                full_index = full_mask.nonzero().squeeze(-1)
                prune_linear_layer(self.k_proj, full_index, 'output')
                prune_linear_layer(self.v_proj, full_index, 'output')
                prune_linear_layer(self.q_proj, full_index, 'output')
                self.out_proj.weight.data *= full_mask
                prune_linear_layer(self.out_proj, full_index, 'input')
            self.hard_concrete_for_heads = None
        return new_config


class _EncoderLayer(nn.Module):
    """One layer of the encoder.
    
    Attributes:
        attn: (:class:`mha.MultiHeadAttention`): The attention mechanism that is used to read the input sequence.
        feed_forward (:class:`ffl.FeedForwardLayer`): The feed-forward layer on top of the attention mechanism.
    """

    def __init__(self, parent: 'PCmer'):
        """Creates a new instance of ``_EncoderLayer``.
        
        Args:
            parent (Encoder): The encoder that the layers is created for.
        """
        super().__init__()
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = SelfAttention(dim=parent.dim_model, heads=parent.num_heads, causal=False)

    def forward(self, phone, mask=None):
        phone = phone + self.attn(self.norm(phone), mask=mask)
        phone = phone + self.conformer(phone)
        return phone


class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""

    def __init__(self, num_layers, num_heads, dim_model, dim_keys, dim_values, residual_dropout, attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        for i, layer in enumerate(self._layers):
            phone = layer(phone, mask)
        return phone


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


class FCPE(nn.Module):

    def __init__(self, input_channel=128, out_dims=360, n_layers=12, n_chans=512, use_siren=False, use_full=False, loss_mse_scale=10, loss_l2_regularization=False, loss_l2_regularization_scale=1, loss_grad1_mse=False, loss_grad1_mse_scale=1, f0_max=1975.5, f0_min=32.7, confidence=False, threshold=0.05, use_input_conv=True):
        super().__init__()
        if use_siren is True:
            raise ValueError('Siren is not supported yet.')
        if use_full is True:
            raise ValueError('Full model is not supported yet.')
        self.loss_mse_scale = loss_mse_scale if loss_mse_scale is not None else 10
        self.loss_l2_regularization = loss_l2_regularization if loss_l2_regularization is not None else False
        self.loss_l2_regularization_scale = loss_l2_regularization_scale if loss_l2_regularization_scale is not None else 1
        self.loss_grad1_mse = loss_grad1_mse if loss_grad1_mse is not None else False
        self.loss_grad1_mse_scale = loss_grad1_mse_scale if loss_grad1_mse_scale is not None else 1
        self.f0_max = f0_max if f0_max is not None else 1975.5
        self.f0_min = f0_min if f0_min is not None else 32.7
        self.confidence = confidence if confidence is not None else False
        self.threshold = threshold if threshold is not None else 0.05
        self.use_input_conv = use_input_conv if use_input_conv is not None else True
        self.cent_table_b = torch.Tensor(np.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0], out_dims))
        self.register_buffer('cent_table', self.cent_table_b)
        _leaky = nn.LeakyReLU()
        self.stack = nn.Sequential(nn.Conv1d(input_channel, n_chans, 3, 1, 1), nn.GroupNorm(4, n_chans), _leaky, nn.Conv1d(n_chans, n_chans, 3, 1, 1))
        self.decoder = PCmer(num_layers=n_layers, num_heads=8, dim_model=n_chans, dim_keys=n_chans, dim_values=n_chans, residual_dropout=0.1, attention_dropout=0.1)
        self.norm = nn.LayerNorm(n_chans)
        self.n_out = out_dims
        self.dense_out = weight_norm(nn.Linear(n_chans, self.n_out))

    def forward(self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder='local_argmax'):
        """
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        """
        if cdecoder == 'argmax':
            self.cdecoder = self.cents_decoder
        elif cdecoder == 'local_argmax':
            self.cdecoder = self.cents_local_decoder
        if self.use_input_conv:
            x = self.stack(mel.transpose(1, 2)).transpose(1, 2)
        else:
            x = mel
        x = self.decoder(x)
        x = self.norm(x)
        x = self.dense_out(x)
        x = torch.sigmoid(x)
        if not infer:
            gt_cent_f0 = self.f0_to_cent(gt_f0)
            gt_cent_f0 = self.gaussian_blurred_cent(gt_cent_f0)
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, gt_cent_f0)
            if self.loss_l2_regularization:
                loss_all = loss_all + l2_regularization(model=self, l2_alpha=self.loss_l2_regularization_scale)
            x = loss_all
        if infer:
            x = self.cdecoder(x)
            x = self.cent_to_f0(x)
            if not return_hz_f0:
                x = (1 + x / 700).log()
        return x

    def cents_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        rtn = torch.sum(ci * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)
        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float('-INF')
            rtn = rtn * confident_mask
        if self.confidence:
            return rtn, confident
        else:
            return rtn

    def cents_local_decoder(self, y, mask=True):
        B, N, _ = y.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.arange(0, 9) + (max_index - 4)
        local_argmax_index[local_argmax_index < 0] = 0
        local_argmax_index[local_argmax_index >= self.n_out] = self.n_out - 1
        ci_l = torch.gather(ci, -1, local_argmax_index)
        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(ci_l * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)
        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float('-INF')
            rtn = rtn * confident_mask
        if self.confidence:
            return rtn, confident
        else:
            return rtn

    def cent_to_f0(self, cent):
        return 10.0 * 2 ** (cent / 1200.0)

    def f0_to_cent(self, f0):
        return 1200.0 * torch.log2(f0 / 10.0)

    def gaussian_blurred_cent(self, cents):
        mask = (cents > 0.1) & (cents < 1200.0 * np.log2(self.f0_max / 10.0))
        B, N, _ = cents.size()
        ci = self.cent_table[None, None, :].expand(B, N, -1)
        return torch.exp(-torch.square(ci - cents) / 1250) * mask.float()


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t, (q, r))
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q[:remaining_rows])
    final_matrix = torch.cat(block_list)
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt(float(nb_columns)) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    return torch.diag(multiplier) @ final_matrix


def linear_attention(q, k, v):
    if v is None:
        out = torch.einsum('...ed,...nd->...ne', k, q)
        return out
    else:
        k_cumsum = k.sum(dim=-2)
        D_inv = 1.0 / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-08)
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
        return out


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=0.0001, device=None):
    b, h, *_ = data.shape
    data_normalizer = data.shape[-1] ** -0.25 if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)
    data_dash = torch.einsum('...id,...jd->...ij', data_normalizer * data, projection)
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = diag_data / 2.0 * data_normalizer ** 2
    diag_data = diag_data.unsqueeze(dim=-1)
    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * torch.exp(data_dash - diag_data + eps)
    return data_dash.type_as(data)


class FastAttention(nn.Module):

    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device
        if self.no_projection:
            q = q.softmax(dim=-1)
            k = torch.exp(k) if self.causal else k.softmax(dim=-2)
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        if v is None:
            out = attn_fn(q, k, None)
            return out
        else:
            out = attn_fn(q, k, v)
            return out


class ConvBlockRes(nn.Module):

    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU(), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


class ResEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class ResDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), output_padding=out_padding, bias=False), nn.BatchNorm2d(out_channels, momentum=momentum), nn.ReLU())
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Intermediate(nn.Module):

    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for i in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class FFN(nn.Module):

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == 'gelu':
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x


class Linear(nn.Linear):

    def forward(self, x: 'Tensor') ->Tensor:
        return F.linear(x, self.weight, None if self.bias is None else self.bias)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_state: 'int', n_head: 'int'):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(self, x: 'Tensor', xa: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None, kv_cache: 'Optional[dict]'=None):
        q = self.query(x)
        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]
        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', mask: 'Optional[Tensor]'=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()
        w = F.softmax(qk, dim=-1)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class Decoder(nn.Module):

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, proximal_bias=False, proximal_init=True, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, h, h_mask):
        """
    x: decoder input
    h: encoder output
    """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2))
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)
            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TimbreFilter(nn.Module):

    def __init__(self, latent_rep_channels):
        super(TimbreFilter, self).__init__()
        self.layers = nn.ModuleList()
        for latent_rep in latent_rep_channels:
            self.layers.append(ConvBlockRes(latent_rep[0], latent_rep[0]))

    def forward(self, x_tensors):
        out_tensors = []
        for i, layer in enumerate(self.layers):
            out_tensors.append(layer(x_tensors[i]))
        return out_tensors


N_MELS = 80


class DeepUnet(nn.Module):

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        concat_tensors = self.tf(concat_tensors)
        x = self.decoder(x, concat_tensors)
        return x


class DeepUnet0(nn.Module):

    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet0, self).__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks)
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class BiGRU(nn.Module):

    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


SAMPLE_RATE = 16000


MEL_FMAX = SAMPLE_RATE // 2


MEL_FMIN = 30


class MelSpectrogram(torch.nn.Module):

    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-05):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + '_' + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new)
        fft = torch.stft(audio, n_fft=n_fft_new, hop_length=hop_length_new, win_length=win_length_new, window=self.hann_window[keyshift_key], center=center, return_complex=True)
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


N_CLASS = 360


WINDOW_LENGTH = 1024


class E2E(nn.Module):

    def __init__(self, hop_length, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E, self).__init__()
        self.mel = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, x):
        mel = self.mel(x.reshape(-1, x.shape[-1])).transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        hidden_vec = 0
        if len(self.fc) == 4:
            for i in range(len(self.fc)):
                x = self.fc[i](x)
                if i == 0:
                    hidden_vec = x
        return hidden_vec, x


class E2E0(nn.Module):

    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(E2E0, self).__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(BiGRU(3 * N_MELS, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())
        else:
            self.fc = nn.Sequential(nn.Linear(3 * N_MELS, N_CLASS), nn.Dropout(0.25), nn.Sigmoid())

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class BiLSTM(nn.Module):

    def __init__(self, input_features, hidden_features, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]


def weight_norm_modules(module, name='weight', dim=0):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(module, Depthwise_Separable_TransposeConv1D):
        module.weight_norm()
        return module
    else:
        return weight_norm(module, name, dim)


class FFT(nn.Module):

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0.0, proximal_bias=False, proximal_init=True, isflow=False, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        if isflow:
            cond_layer = torch.nn.Conv1d(kwargs['gin_channels'], 2 * hidden_channels * n_layers, 1)
            self.cond_pre = torch.nn.Conv1d(hidden_channels, 2 * hidden_channels, 1)
            self.cond_layer = weight_norm_modules(cond_layer, name='weight')
            self.gin_channels = kwargs['gin_channels']
        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
            self.norm_layers_1.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g=None):
        """
    x: decoder input
    h: encoder output
    """
        if g is not None:
            g = self.cond_layer(g)
        self_attn_mask = commons.subsequent_mask(x_mask.size(2))
        x = x * x_mask
        for i in range(self.n_layers):
            if g is not None:
                x = self.cond_pre(x)
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                x = commons.fused_add_tanh_sigmoid_multiply(x, g_l, torch.IntTensor([self.hidden_channels]))
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
        x = x * x_mask
        return x


class LayerNorm(nn.LayerNorm):

    def forward(self, x: 'Tensor') ->Tensor:
        return super().forward(x.float()).type(x.dtype)


Conv1dModel = nn.Conv1d


class ConvReluNorm(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, 'Number of layers should be larger than 0.'
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(Conv1dModel(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(Conv1dModel(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


def remove_weight_norm_modules(module, name='weight'):
    if isinstance(module, Depthwise_Separable_Conv1D) or isinstance(module, Depthwise_Separable_TransposeConv1D):
        module.remove_weight_norm()
    else:
        remove_weight_norm(module, name)


class WN(torch.nn.Module):

    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = weight_norm_modules(cond_layer, name='weight')
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = Conv1dModel(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding)
            in_layer = weight_norm_modules(in_layer, name='weight')
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm_modules(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm_modules(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm_modules(l)
        for l in self.res_skip_layers:
            remove_weight_norm_modules(l)


class Log(nn.Module):

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-05)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0))
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Module):

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False, wn_sharing_parameter=None):
        assert channels % 2 == 0, 'channels should be divisible by 2'
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class TransformerCouplingLayer(nn.Module):

    def __init__(self, channels, hidden_channels, kernel_size, n_layers, n_heads, p_dropout=0, filter_channels=0, mean_only=False, wn_sharing_parameter=None, gin_channels=0):
        assert channels % 2 == 0, 'channels should be divisible by 2'
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = attentions.FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, isflow=True, gin_channels=gin_channels) if wn_sharing_parameter is None else wn_sharing_parameter
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)
    return filter


class LowPassFilter1d(nn.Module):

    def __init__(self, cutoff=0.5, half_width=0.6, stride: 'int'=1, padding: 'bool'=True, padding_mode: 'str'='replicate', kernel_size: 'int'=12, C=None):
        super().__init__()
        if cutoff < -0.0:
            raise ValueError('Minimum cutoff must be larger than zero.')
        if cutoff > 0.5:
            raise ValueError('A cutoff above 0.5 does not make sense.')
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer('filter', filter)
        self.conv1d_block = None
        if C is not None:
            self.conv1d_block = [nn.Conv1d(C, C, kernel_size, stride=self.stride, groups=C, bias=False)]
            self.conv1d_block[0].weight = nn.Parameter(self.filter.expand(C, -1, -1))
            self.conv1d_block[0].requires_grad_(False)

    def forward(self, x):
        if self.conv1d_block[0].weight.device != x.device:
            self.conv1d_block[0] = self.conv1d_block[0]
        if self.conv1d_block is None:
            _, C, _ = x.shape
            if self.padding:
                x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
            out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        else:
            if self.padding:
                x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
            out = self.conv1d_block[0](x)
        return out


class DownSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None, C=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size, C=C)

    def forward(self, x):
        xx = self.lowpass(x)
        return xx


class UpSample1d(nn.Module):

    def __init__(self, ratio=2, kernel_size=None, C=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer('filter', filter)
        self.conv_transpose1d_block = None
        if C is not None:
            self.conv_transpose1d_block = [nn.ConvTranspose1d(C, C, kernel_size=self.kernel_size, stride=self.stride, groups=C, bias=False)]
            self.conv_transpose1d_block[0].weight = nn.Parameter(self.filter.expand(C, -1, -1).clone())
            self.conv_transpose1d_block[0].requires_grad_(False)

    def forward(self, x, C=None):
        if self.conv_transpose1d_block[0].weight.device != x.device:
            self.conv_transpose1d_block[0] = self.conv_transpose1d_block[0]
        if self.conv_transpose1d_block is None:
            if C is None:
                _, C, _ = x.shape
            x = F.pad(x, (self.pad, self.pad), mode='replicate')
            x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
            x = x[..., self.pad_left:-self.pad_right]
        else:
            x = F.pad(x, (self.pad, self.pad), mode='replicate')
            x = self.ratio * self.conv_transpose1d_block[0](x)
            x = x[..., self.pad_left:-self.pad_right]
        return x


class Activation1d(nn.Module):

    def __init__(self, activation, up_ratio: 'int'=2, down_ratio: 'int'=2, up_kernel_size: 'int'=12, down_kernel_size: 'int'=12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta = x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + 1.0 / (beta + self.no_div_by_zero) * pow(sin(x * alpha), 2)
        return x


class Mish(nn.Module):
    """
    Mish activation function is proposed in "Mish: A Self 
    Regularized Non-Monotonic Neural Activation Function" 
    paper, https://arxiv.org/abs/1908.08681.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SnakeAlias(nn.Module):

    def __init__(self, channels, up_ratio: 'int'=2, down_ratio: 'int'=2, up_kernel_size: 'int'=12, down_kernel_size: 'int'=12, C=None):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = SnakeBeta(channels, alpha_logscale=True)
        self.upsample = UpSample1d(up_ratio, up_kernel_size, C)
        self.downsample = DownSample1d(down_ratio, down_kernel_size, C)

    def forward(self, x, C=None):
        x = self.upsample(x, C)
        x = self.act(x)
        x = self.downsample(x)
        return x


class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int', bias: 'bool', layer_norm: 'Optional[Module]', prune_conv_channels: 'bool'=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        if prune_conv_channels:
            self.hard_concrete = HardConcrete(n_in=out_channels, init_mean=0.01)
        else:
            self.hard_concrete = None

    def forward(self, x: 'Tensor', length: 'Optional[Tensor]') ->Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)
        if self.hard_concrete is not None:
            channel_mask = self.hard_concrete()
            x = x * channel_mask.unsqueeze(-1)
        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode='floor') + 1
            length = torch.max(torch.zeros_like(length), length)
        return x, length

    def get_num_params_and_out_channels(self, in_channels):
        if self.hard_concrete is not None:
            out_channels = self.hard_concrete.l0_norm()
        else:
            out_channels = self.conv.out_channels
        num_params = in_channels * out_channels * self.kernel_size
        if self.conv.bias is not None:
            num_params += out_channels
        if self.layer_norm is not None:
            num_params += out_channels * 2
        return num_params, out_channels


class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 5, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 3, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = t_func.gelu(self.norm0(self.conv0(x)))
        x = t_func.gelu(self.conv1(x))
        x = t_func.gelu(self.conv2(x))
        x = t_func.gelu(self.conv3(x))
        x = t_func.gelu(self.conv4(x))
        x = t_func.gelu(self.conv5(x))
        x = t_func.gelu(self.conv6(x))
        return x


class FeatureProjection(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class ConvolutionalPositionalEmbedding(Module):
    """Positional embedding which is placed at the beginning of Transformer.

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """

    def __init__(self, embed_dim: 'int', kernel_size: 'int', groups: 'int'):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups)
        self.conv = nn.utils.weight_norm(self.conv, name='weight', dim=2)
        self.num_remove: 'int' = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        for hook in self.conv._forward_pre_hooks.values():
            if hook.__module__ == 'torch.nn.utils.weight_norm' and hook.__class__.__name__ == 'WeightNorm':
                torch.nn.utils.remove_weight_norm(self.conv)
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., :-self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class WavLMSelfAttention(SelfAttention):
    """Multi-headed self-attention for WavLM model :cite:`chen2022wavlm`.

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional): Dropout probability on attn_output_weights. (Default: to ``0.0``)
        bias (bool, optional): If ``True``, add bias to input / output projection layers. (Default: ``True``)
        has_relative_attention_bias (bool, optional): If ``True``, apply relative position embedding.
            Necessary in the first encoder layer, but not in the subsequent ones. (Default: ``False``)
        num_buckets (int, optional): Number of buckets for relative position embedding. (Default: ``32``)
        max_distance (int, optional): Naximum distance for relative position embedding. (Default: ``128``)
        gru_rel_pos (bool, optional): If ``True``, apply gated relative position embedding. (Default: ``False``)
    """

    def __init__(self, embed_dim: 'int', total_num_heads: 'int', remaining_heads: 'Optional[List[int]]'=None, dropout: 'float'=0.0, bias: 'bool'=True, has_relative_attention_bias: 'bool'=False, num_buckets: 'int'=32, max_distance: 'int'=128, gru_rel_pos: 'bool'=True, prune_heads: 'bool'=False, prune_layer: 'bool'=False):
        self.total_num_heads = total_num_heads
        if remaining_heads is None:
            self.remaining_heads = list(range(total_num_heads))
        else:
            self.remaining_heads = remaining_heads
        self.head_dim = embed_dim // total_num_heads
        super().__init__(embed_dim, len(self.remaining_heads), self.head_dim, dropout, prune_heads, prune_layer)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if has_relative_attention_bias:
            self.rel_attn_embed = nn.Embedding(num_buckets, total_num_heads)
        else:
            self.rel_attn_embed = None
        self.k_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, len(self.remaining_heads) * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(len(self.remaining_heads) * self.head_dim, embed_dim, bias=bias)
        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)
            self.gru_rel_pos_const = nn.Parameter(torch.ones(1, total_num_heads, 1, 1))
        self.has_position_bias = True

    def compute_bias(self, query_length: 'int', key_length: 'int') ->Tensor:
        """Compute relative position embeddings for WavLM model.
        Args:
            query_length (int): Query position can take values between 0 and ``query_length - 1``.
            key_length (int): Key position can take values between 0 and ``key_length - 1``.
        Returns:
            Tensor of shape `(num_heads, query_length, key_length)`, relative positions embeddings
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket
        values = self.rel_attn_embed(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: 'Tensor', bidirectional: 'bool'=True):
        """Compute relative position buckets for WavLM model. Computation similar to formula (5) in WavLM
           paper :cite:`chen2022wavlm`.
        Args:
            relative_positions (Tensor): Relative offsets between query and key positions,
                of shape ``(query_length, key_length)``.
            bidirectional (bool): If ``True``, values will be filled both above and below the diagonal in the resulting
                matrix. If ``False``, the elements above the diagonal (i.e. with negative relative offsets) will be set
                to zero. (Default ``True``)
        Returns:
            Tensor of shape ``(query_length, key_length)`` filled bucketed values of with relative positions.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = torch.zeros_like(relative_positions, dtype=torch.long)
        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        relative_postion_if_large = max_exact + torch.log(relative_positions.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def forward(self, query: 'Tensor', attention_mask: 'Optional[Tensor]'=None, position_bias: 'Optional[Tensor]'=None, key_padding_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query (Tensor): Input of shape ``(batch_size, src_len, embed_dim)``.
            key_padding_mask (Tensor or None, optional): Mask to exclude keys that are pads, of shape
                `(batch, src_len)`, where padding elements are indicated by 1s. (Default: ``None``)
            attn_mask: Needs to be ``None``. The argument exists for compatibility with
                ``EncoderLayer``. (Default: ``None``)
            position_bias (Tensor or None, optional): Position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``. When used inside WavLM model encoder, will be
                generated in the first layer and then passed from each encoder layer to the next one.
                (Default: ``None``)
        Returns:
            attn_output (Tensor): Attention output of shape ``(batch_size, src_len, embed_dim)``.
            position_bias (Tensor or None): Position bias of shape ``(batch_size * num_heads, src_len, src_len)``.
        """
        bsz, seq_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert key_padding_mask is None
        if self.rel_attn_embed is not None and position_bias is None:
            position_bias = self.compute_bias(seq_len, seq_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.total_num_heads, seq_len, seq_len)
        attn_mask_rel_pos: 'Optional[Tensor]' = None
        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos:
                query_layer = query.view(bsz, seq_len, self.total_num_heads, -1)
                query_layer = query_layer.permute(0, 2, 1, 3)
                gate_a, gate_b = torch.sigmoid(self.gru_rel_pos_linear(query_layer).view(bsz, self.total_num_heads, seq_len, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz * self.total_num_heads, -1, 1) * position_bias
            attn_mask_rel_pos = attn_mask_rel_pos.view((-1, seq_len, seq_len))
            attn_mask_rel_pos = attn_mask_rel_pos.reshape(bsz, self.total_num_heads, seq_len, seq_len)[:, self.remaining_heads, :, :]
        attn_mask = attn_mask_rel_pos
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask
        if key_padding_mask is not None:
            attn_mask = attn_mask.masked_fill(key_padding_mask.reshape(bsz, 1, 1, seq_len), float('-inf'))
        attn_output, _ = super().forward(query, attention_mask=attn_mask)
        return attn_output, position_bias

    def prune(self):
        new_config = {'use_attention': True, 'remaining_heads': self.remaining_heads}
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer()
            self.out_proj.weight.data *= layer_mask
            self.out_proj.bias.data *= layer_mask
            if layer_mask == 0:
                new_config['use_attention'] = False
            self.hard_concrete_for_layer = None
        if self.hard_concrete_for_heads is not None:
            assert not self.hard_concrete_for_heads.training
            head_mask = self.hard_concrete_for_heads()
            new_config['remaining_heads'] = head_mask.nonzero().squeeze(-1).tolist()
            if len(new_config['remaining_heads']) == 0:
                new_config['use_attention'] = False
            else:
                full_mask = head_mask.repeat_interleave(self.head_dim)
                full_index = full_mask.nonzero().squeeze(-1)
                prune_linear_layer(self.k_proj, full_index, 'output')
                prune_linear_layer(self.v_proj, full_index, 'output')
                prune_linear_layer(self.q_proj, full_index, 'output')
                self.out_proj.weight.data *= full_mask
                prune_linear_layer(self.out_proj, full_index, 'input')
            self.hard_concrete_for_heads = None
        return new_config


class FeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(self, io_features: 'int', intermediate_features: 'int', intermediate_dropout: 'float', output_dropout: 'float', prune_intermediate: 'bool'=False, prune_layer: 'bool'=False):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)
        if prune_intermediate:
            self.hard_concrete_for_intermediate = HardConcrete(n_in=intermediate_features, init_mean=0.5)
        else:
            self.hard_concrete_for_intermediate = None
        if prune_layer:
            self.hard_concrete_for_layer = HardConcrete(n_in=1, init_mean=0.01)
        else:
            self.hard_concrete_for_layer = None

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)
        if self.hard_concrete_for_intermediate is not None:
            intermediate_mask = self.hard_concrete_for_intermediate()
            x = x * intermediate_mask
        x = self.output_dense(x)
        x = self.output_dropout(x)
        if self.hard_concrete_for_layer is not None:
            layer_mask = self.hard_concrete_for_layer()
            x = x * layer_mask
        return x

    def get_num_params(self):
        io_features = self.intermediate_dense.in_features
        if self.hard_concrete_for_intermediate is not None:
            intermediate_features = self.hard_concrete_for_intermediate.l0_norm()
        else:
            intermediate_features = self.intermediate_dense.out_features
        num_params = (io_features + 1) * intermediate_features + (intermediate_features + 1) * io_features
        if self.hard_concrete_for_layer is not None:
            num_params *= self.hard_concrete_for_layer.l0_norm()
        return num_params

    def prune(self):
        new_config = {'use_feed_forward': True, 'ff_interm_features': self.intermediate_dense.out_features}
        if self.hard_concrete_for_layer is not None:
            assert not self.hard_concrete_for_layer.training
            layer_mask = self.hard_concrete_for_layer()
            self.output_dense.weight.data *= layer_mask
            self.output_dense.bias.data *= layer_mask
            if layer_mask == 0:
                new_config['use_feed_forward'] = False
            self.hard_concrete_for_layer = None
        if self.hard_concrete_for_intermediate is not None:
            assert not self.hard_concrete_for_intermediate.training
            interm_mask = self.hard_concrete_for_intermediate()
            interm_index = interm_mask.nonzero().squeeze(-1)
            new_config['ff_interm_features'] = len(interm_index)
            if new_config['ff_interm_features'] == 0:
                new_config['use_feed_forward'] = False
            else:
                prune_linear_layer(self.intermediate_dense, interm_index, 'output')
                self.output_dense.weight.data *= interm_mask
                prune_linear_layer(self.output_dense, interm_index, 'input')
            self.hard_concrete_for_intermediate = None
        return new_config


class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(self, attention: 'Optional[Module]', dropout: 'float', layer_norm_first: 'bool', feed_forward: 'Optional[Module]', embed_dim: 'int'):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: 'Tensor', attention_mask: 'Optional[Tensor]'=None, position_bias: 'Optional[Tensor]'=None, key_padding_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Input of shape ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
            position_bias (Tensor or ``None``, optional): position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``.
                Only necessary for WavLM model, ``None`` otherwise. (Default: ``None``)
            key_padding_mask (Tensor or ``None``, optional): key padding mask of shape ``(batch_size, src_len)``.
                Only used for WavLM model, ignored otherwise. (Default: ``None``)
        Returns:
            (x, position_bias): Shapes are the same as in the input. Position bias is only relevant for WaLM model,
                ``None`` otherwise.
        """
        if self.attention is not None:
            residual = x
            if self.layer_norm_first:
                x = self.layer_norm(x)
            x, position_bias = self.attention(x, attention_mask=attention_mask, position_bias=position_bias, key_padding_mask=key_padding_mask)
            x = self.dropout(x)
            x = residual + x
        if self.layer_norm_first:
            if self.feed_forward is not None:
                x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            if self.feed_forward is not None:
                x = x + self.feed_forward(x)
            x = self.final_layer_norm(x)
        return x, position_bias

    def get_num_params(self):
        num_params = self.embed_dim * 2 * 2
        if self.attention is not None:
            num_params += self.attention.get_num_params()
        if self.feed_forward is not None:
            num_params += self.feed_forward.get_num_params()
        return num_params


class Transformer(Module):

    def __init__(self, pos_conv_embed: 'Module', dropout: 'float', layers: 'Module', layer_norm_first: 'bool', layer_drop: 'float'):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: 'Tensor'):
        x = x + self.pos_conv_embed(x)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        x = self.dropout(x)
        return x

    def forward(self, x: 'Tensor', attention_mask: 'Optional[Tensor]'=None, position_bias: 'Optional[Tensor]'=None) ->Tensor:
        x = self._preprocess(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x, position_bias = layer(x, attention_mask, position_bias=position_bias)
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(self, x: 'Tensor', attention_mask: 'Optional[Tensor]'=None, num_layers: 'Optional[int]'=None, position_bias: 'Optional[Tensor]'=None) ->List[Tensor]:
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(f'`num_layers` must be between [1, {len(self.layers)}]')
        ret: 'List[Tensor]' = []
        x = self._preprocess(x)
        for layer in self.layers:
            x, position_bias = layer(x, attention_mask, position_bias=position_bias)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret

    def get_num_params(self):
        num_params = sum(p.numel() for p in self.pos_conv_embed.parameters()) + self.pos_conv_embed.embed_dim * 2
        for layer in self.layers:
            num_params += layer.get_num_params()
        return num_params

    def prune(self):
        new_config = defaultdict(list)
        for layer in self.layers:
            attention_config = layer.attention.prune()
            new_config['use_attention'].append(attention_config['use_attention'])
            if 'remaining_heads' in attention_config:
                new_config['remaining_heads'].append(attention_config['remaining_heads'])
            else:
                new_config['num_heads'].append(attention_config['num_heads'])
            if not attention_config['use_attention']:
                layer.attention = None
            ff_config = layer.feed_forward.prune()
            new_config['use_feed_forward'].append(ff_config['use_feed_forward'])
            new_config['ff_interm_features'].append(ff_config['ff_interm_features'])
            if not ff_config['use_feed_forward']:
                layer.feed_forward = None
        return new_config


class Wav2Vec2Model(Module):
    """Acoustic model used in *wav2vec 2.0* :cite:`baevski2020wav2vec`.

    Note:
        To build the model, please use one of the factory functions.
        :py:func:`wav2vec2_model`, :py:func:`wav2vec2_base`, :py:func:`wav2vec2_large`,
        :py:func:`wav2vec2_large_lv60k`, :py:func:`hubert_base`, :py:func:`hubert_large`,
        and :py:func:`hubert_xlarge`.

    See Also:
        * :class:`torchaudio.pipelines.Wav2Vec2Bundle`: Pretrained models (without fine-tuning)
        * :class:`torchaudio.pipelines.Wav2Vec2ASRBundle`: ASR pipelines with pretrained models.

    Args:
        feature_extractor (torch.nn.Module):
            Feature extractor that extracts feature vectors from raw audio Tensor.

        encoder (torch.nn.Module):
            Encoder that converts the audio features into the sequence of probability
            distribution (in negative log-likelihood) over labels.

        aux (torch.nn.Module or None, optional):
            Auxiliary module. If provided, the output from encoder is passed to this module.
    """

    def __init__(self, normalize_waveform: 'bool', feature_extractor: 'Module', encoder: 'Module', aux: 'Optional[Module]'=None):
        super().__init__()
        self.normalize_waveform = normalize_waveform
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux

    @torch.jit.export
    def extract_features(self, waveforms: 'Tensor', lengths: 'Optional[Tensor]'=None, num_layers: 'Optional[int]'=None) ->Tuple[List[Tensor], Optional[Tensor]]:
        """Extract feature vectors from raw waveforms

        This returns the list of outputs from the intermediate layers of
        transformer block in encoder.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that the entire audio waveform
                length is valid.
            num_layers (int or None, optional):
                If given, limit the number of intermediate layers to go through.
                Providing `1` will stop the computation after going through one
                intermediate layers. If not given, the outputs from all the
                intermediate layers are returned.

        Returns:
            (List[Tensor], Optional[Tensor]):
            List of Tensors
                Features from requested layers.
                Each Tensor is of shape: `(batch, time frame, feature dimension)`
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of each feature Tensor.
        """
        if self.normalize_waveform:
            if lengths is not None:
                waveforms = [F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
            else:
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths

    def get_num_params(self):
        """Calculate the current size."""
        feature_extractor_size, encoder_in_features = self.feature_extractor.get_num_params_and_final_out_channels()
        encoder_size = self.encoder.get_num_params(encoder_in_features)
        return feature_extractor_size + encoder_size

    def prune(self):
        self.eval()
        conv_config, conv_out_index = self.feature_extractor.prune()
        transformer_config = self.encoder.prune(conv_out_index)
        use_attention = transformer_config['use_attention']
        use_feed_forward = transformer_config['use_feed_forward']
        num_heads = transformer_config['num_heads']
        remaining_heads = transformer_config['remaining_heads']
        ff_interm_features = transformer_config['ff_interm_features']
        return conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features

    def forward(self, waveforms: 'Tensor', lengths: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The sequences of probability distribution (in logit) over labels.
                Shape: `(batch, frames, num labels)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """
        if self.normalize_waveform:
            if lengths is not None:
                waveforms = [F.layer_norm(wave[:length], (length,)) for wave, length in zip(waveforms, lengths)]
                waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
            else:
                waveforms = F.layer_norm(waveforms, waveforms.shape[-1:])
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x, lengths


class PositionalConvEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(768, 768, kernel_size=128, padding=128 // 2, groups=16)
        self.conv = nn.utils.weight_norm(self.conv, name='weight', dim=2)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        x = t_func.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


class SamePad(nn.Module):

    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class GLU_Linear(nn.Module):

    def __init__(self, input_dim, output_dim, glu_type='sigmoid', bias_in_glu=True):
        super(GLU_Linear, self).__init__()
        self.glu_type = glu_type
        self.output_dim = output_dim
        if glu_type == 'sigmoid':
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == 'swish':
            self.glu_act = Swish()
        elif glu_type == 'relu':
            self.glu_act = torch.nn.ReLU()
        elif glu_type == 'gelu':
            self.glu_act = torch.nn.GELU()
        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        x = self.linear(x)
        if self.glu_type == 'bilinear':
            x = x[:, :, 0:self.output_dim] * x[:, :, self.output_dim:self.output_dim * 2]
        else:
            x = x[:, :, 0:self.output_dim] * self.glu_act(x[:, :, self.output_dim:self.output_dim * 2])
        return x


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """
    if p <= 0:
        return module
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4
    if not is_conv:
        assert module.weight.size(1) % block_size == 0, 'Input features must be a multiple of block sizes'
    elif module.kernel_size == (1, 1):
        assert module.in_channels % block_size == 0, 'Input channels must be a multiple of block sizes'
    else:
        k = module.kernel_size[0] * module.kernel_size[1]
        assert k % block_size == 0, 'Kernel size must be a multiple of block size'

    def _forward_pre_hook(mod, input):
        if mod.training:
            if not is_conv:
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)
                mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
            mask = mask
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8, has_relative_attention_bias=False, num_buckets=32, max_distance=128, gru_rel_pos=False, rescale_init=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size'
        k_bias = True
        if rescale_init:
            k_bias = False
        k_embed_dim = embed_dim
        q_embed_dim = embed_dim
        self.k_proj = quant_noise(nn.Linear(self.kdim, k_embed_dim, bias=k_bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.q_proj = quant_noise(nn.Linear(embed_dim, q_embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        relative_postion_if_large = max_exact + torch.log(relative_positions.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_postion_if_large = torch.min(relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values

    def forward(self, query, key: 'Optional[Tensor]', value: 'Optional[Tensor]', key_padding_mask: 'Optional[Tensor]'=None, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]'=None, need_weights: 'bool'=True, static_kv: 'bool'=False, attn_mask: 'Optional[Tensor]'=None, before_softmax: 'bool'=False, need_head_weights: 'bool'=False, position_bias: 'Optional[Tensor]'=None) ->Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        is_tpu = query.device.type == 'xla'
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]
        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1).view(bsz * self.num_heads, tgt_len, src_len)
        if not is_tpu and incremental_state is None and not static_kv and not torch.jit.is_scripting() and self.q_head_dim == self.head_dim:
            assert key is not None and value is not None
            assert attn_mask is None
            attn_mask_rel_pos = None
            if position_bias is not None:
                attn_mask_rel_pos = position_bias
                if self.gru_rel_pos:
                    query_layer = query.transpose(0, 1)
                    new_x_shape = query_layer.size()[:-1] + (self.num_heads, -1)
                    query_layer = query_layer.view(*new_x_shape)
                    query_layer = query_layer.permute(0, 2, 1, 3)
                    _B, _H, _L, __ = query_layer.size()
                    gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                    attn_mask_rel_pos = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias
                attn_mask_rel_pos = attn_mask_rel_pos.view((-1, tgt_len, tgt_len))
            k_proj_bias = self.k_proj.bias
            if k_proj_bias is None:
                k_proj_bias = torch.zeros_like(self.q_proj.bias)
            x, attn = F.multi_head_attention_forward(query, key, value, self.embed_dim, self.num_heads, torch.empty([0]), torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)), self.bias_k, self.bias_v, self.add_zero_attn, self.dropout_module.p, self.out_proj.weight, self.out_proj.bias, self.training, key_padding_mask, need_weights, attn_mask_rel_pos, use_separate_proj_weight=True, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight)
            return x, attn, position_bias
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and 'prev_key' in saved_state:
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                _prev_key = saved_state['prev_key']
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if 'prev_value' in saved_state:
                _prev_value = saved_state['prev_value']
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: 'Optional[Tensor]' = None
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(key_padding_mask=key_padding_mask, prev_key_padding_mask=prev_key_padding_mask, batch_size=bsz, src_len=k.size(1), static_kv=static_kv)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if before_softmax:
            return attn_weights, v, position_bias
        if position_bias is not None:
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                position_bias = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias
            position_bias = position_bias.view(attn_weights.size())
            attn_weights = attn_weights + position_bias
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: 'Optional[Tensor]' = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: 'Optional[Tensor]', prev_key_padding_mask: 'Optional[Tensor]', batch_size: 'int', src_len: 'int', static_kv: 'bool') ->Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        elif prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - prev_key_padding_mask.size(1)), device=prev_key_padding_mask.device)
                new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            else:
                new_key_padding_mask = prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros((batch_size, src_len - key_padding_mask.size(1)), device=key_padding_mask.device)
                new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            else:
                new_key_padding_mask = key_padding_mask.float()
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    def _get_input_buffer(self, incremental_state: 'Optional[Dict[str, Dict[str, Optional[Tensor]]]]') ->Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: 'Dict[str, Optional[Tensor]]' = {}
            return empty_result

    def _set_input_buffer(self, incremental_state: 'Dict[str, Dict[str, Optional[Tensor]]]', buffer: 'Dict[str, Optional[Tensor]]'):
        return self.set_incremental_state(incremental_state, 'attn_state', buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: 'int', src_len: 'int', bsz: 'int'):
        return attn_weights


def gelu(x: 'torch.Tensor') ->torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


def gelu_accurate(x):
    if not hasattr(gelu_accurate, '_a'):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return 0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))


def get_activation_fn(activation: 'str'):
    """Returns the activation function corresponding to `activation`"""
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return gelu
    elif activation == 'gelu_fast':
        warnings.warn('--activation-fn=gelu_fast has been renamed to gelu_accurate')
        return gelu_accurate
    elif activation == 'gelu_accurate':
        return gelu_accurate
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'linear':
        return lambda x: x
    elif activation == 'glu':
        return lambda x: x
    else:
        raise RuntimeError('--activation-fn {} not supported'.format(activation))


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, embedding_dim: 'float'=768, ffn_embedding_dim: 'float'=3072, num_attention_heads: 'float'=8, dropout: 'float'=0.1, attention_dropout: 'float'=0.1, activation_dropout: 'float'=0.1, activation_fn: 'str'='relu', layer_norm_first: 'bool'=False, has_relative_attention_bias: 'bool'=False, num_buckets: 'int'=0, max_distance: 'int'=0, rescale_init: 'bool'=False, gru_rel_pos: 'bool'=False) ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(self.embedding_dim, num_attention_heads, dropout=attention_dropout, self_attention=True, has_relative_attention_bias=has_relative_attention_bias, num_buckets=num_buckets, max_distance=max_distance, rescale_init=rescale_init, gru_rel_pos=gru_rel_pos)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        if self.activation_name == 'glu':
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, 'swish')
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(self, x: 'torch.Tensor', self_attn_mask: 'torch.Tensor'=None, self_attn_padding_mask: 'torch.Tensor'=None, need_weights: 'bool'=False, pos_bias=None):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=False, attn_mask=self_attn_mask, position_bias=pos_bias)
            x = self.dropout1(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == 'glu':
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask, need_weights=need_weights, attn_mask=self_attn_mask, position_bias=pos_bias)
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            residual = x
            if self.activation_name == 'glu':
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        return x, attn, pos_bias


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class TransformerEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.pos_conv = nn.Conv1d(self.embedding_dim, self.embedding_dim, kernel_size=args.conv_pos, padding=args.conv_pos // 2, groups=args.conv_pos_groups)
        dropout = 0
        std = math.sqrt(4 * (1.0 - dropout) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name='weight', dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())
        if hasattr(args, 'relative_position_embedding'):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0
        self.layers = nn.ModuleList([TransformerSentenceEncoderLayer(embedding_dim=self.embedding_dim, ffn_embedding_dim=args.encoder_ffn_embed_dim, num_attention_heads=args.encoder_attention_heads, dropout=self.dropout, attention_dropout=args.attention_dropout, activation_dropout=args.activation_dropout, activation_fn=args.activation_fn, layer_norm_first=args.layer_norm_first, has_relative_attention_bias=self.relative_position_embedding and i == 0, num_buckets=self.num_buckets, max_distance=self.max_distance, gru_rel_pos=args.gru_rel_pos) for i in range(args.encoder_layers)])
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x[padding_mask] = 0
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or dropout_probability > self.layerdrop:
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False, self_attn_mask=streaming_mask, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break
        if r is not None:
            x = r
        x = x.transpose(0, 1)
        return x, layer_results


def _compute_mask(shape: 'Tuple[int, int]', mask_prob: 'float', mask_length: 'int', device: 'torch.device', min_masks: 'int'=0) ->torch.Tensor:
    batch_size, sequence_length = shape
    if mask_length < 1:
        raise ValueError('`mask_length` has to be bigger than 0.')
    if mask_length > sequence_length:
        raise ValueError(f'`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`')
    num_masked_spans = int(mask_prob * sequence_length / mask_length + random.random())
    num_masked_spans = max(num_masked_spans, min_masks)
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length
    mask = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)
    uniform_dist = torch.ones((batch_size, sequence_length - (mask_length - 1)), device=device)
    mask_indices = torch.multinomial(uniform_dist, num_masked_spans)
    mask_indices = mask_indices.unsqueeze(dim=-1).expand((batch_size, num_masked_spans, mask_length)).reshape(batch_size, num_masked_spans * mask_length)
    offsets = torch.arange(mask_length, device=device)[None, None, :].expand((batch_size, num_masked_spans, mask_length)).reshape(batch_size, num_masked_spans * mask_length)
    mask_idxs = mask_indices + offsets
    mask = mask.scatter(1, mask_idxs, True)
    return mask


class Hubert(nn.Module):

    def __init__(self, num_label_embeddings: 'int'=100, mask: 'bool'=True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(nn.TransformerEncoderLayer(768, 12, 3072, activation='gelu', batch_first=True), 12)
        self.proj = nn.Linear(768, 256)
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(768).uniform_())
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: 'torch.Tensor') ->Tuple[torch.Tensor, torch.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.size(0), x.size(1)), 0.8, 10, x.device, 2)
            x[mask] = self.masked_spec_embed
        return x, mask

    def encode(self, x: 'torch.Tensor', layer: 'Optional[int]'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: 'torch.Tensor') ->torch.Tensor:
        logits = torch.cosine_similarity(x.unsqueeze(2), self.label_embedding.weight.unsqueeze(0).unsqueeze(0), dim=-1)
        return logits / 0.1


class HubertSoft(Hubert):

    def __init__(self):
        super().__init__()

    def units(self, wav: 'torch.Tensor') ->torch.Tensor:
        wav = t_func.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)

    def forward(self, x):
        return self.units(x)


class Fp32GroupNorm(nn.GroupNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(input.float(), self.num_groups, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(input.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)


class TransposeLast(nn.Module):

    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class ConvFeatureExtractionModel(nn.Module):

    def __init__(self, conv_layers: 'List[Tuple[int, int, int]]', dropout: 'float'=0.0, mode: 'str'='default', conv_bias: 'bool'=False, conv_type: 'str'='default'):
        super().__init__()
        assert mode in {'default', 'layer_norm'}

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):

            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            assert (is_layer_norm and is_group_norm) is False, 'layer norm and group norm are exclusive'
            if is_layer_norm:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.Sequential(TransposeLast(), Fp32LayerNorm(dim, elementwise_affine=True), TransposeLast()), nn.GELU())
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), Fp32GroupNorm(dim, dim, affine=True), nn.GELU())
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
        self.conv_type = conv_type
        if self.conv_type == 'default':
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, 'invalid conv definition: ' + str(cl)
                dim, k, stride = cl
                self.conv_layers.append(block(in_d, dim, k, stride, is_layer_norm=mode == 'layer_norm', is_group_norm=mode == 'default' and i == 0, conv_bias=conv_bias))
                in_d = dim
        elif self.conv_type == 'conv2d':
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                dim, k, stride = cl
                self.conv_layers.append(torch.nn.Conv2d(in_d, dim, k, stride))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == 'custom':
            in_d = 1
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                dim, k, stride = cl
                self.conv_layers.append(torch.nn.Conv2d(in_d, dim, k, stride, padding=1))
                self.conv_layers.append(torch.nn.LayerNorm([dim, idim]))
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(torch.nn.MaxPool2d(2, stride=2, ceil_mode=True))
                    idim = int(math.ceil(idim / 2))
        else:
            pass

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        if self.conv_type == 'custom':
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == 'conv2d':
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x


class GradMultiply(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def compute_mask_indices(shape: 'Tuple[int, int]', padding_mask: 'Optional[torch.Tensor]', mask_prob: 'float', mask_length: 'int', mask_type: 'str'='static', mask_other: 'float'=0.0, min_masks: 'int'=0, no_overlap: 'bool'=False, min_space: 'int'=0) ->np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask
        if mask_type == 'static':
            lengths = np.full(num_mask, mask_length)
        elif mask_type == 'uniform':
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == 'normal':
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == 'poisson':
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception('unknown mask selection ' + mask_type)
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)
        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))
                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts
            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter((e - s if e - s >= length + min_space else 0 for s, e in parts), np.int)
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1
            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
            mask_idc = np.asarray([(mask_idc[j] + offset) for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))
    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask


logger = logging.getLogger(__name__)


class WavLM(nn.Module):

    def __init__(self, cfg: 'WavLMConfig') ->None:
        super().__init__()
        logger.info(f'WavLM Config: {cfg.__dict__}')
        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(conv_layers=feature_enc_layers, dropout=0.0, mode=cfg.extractor_mode, conv_bias=cfg.conv_bias)
        self.post_extract_proj = nn.Linear(self.embed, cfg.encoder_embed_dim) if self.embed != cfg.encoder_embed_dim else None
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = cfg.feature_grad_mult
        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices((B, T), padding_mask, self.mask_prob, self.mask_length, self.mask_selection, self.mask_other, min_masks=2, no_overlap=self.no_mask_overlap, min_space=self.mask_min_space)
            mask_indices = torch.from_numpy(mask_indices)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None
        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices((B, C), None, self.mask_channel_prob, self.mask_channel_length, self.mask_channel_selection, self.mask_channel_other, no_overlap=self.no_mask_channel_overlap, min_space=self.mask_channel_min_space)
            mask_channel_indices = torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0
        return x, mask_indices

    def forward_padding_mask(self, features: 'torch.Tensor', padding_mask: 'torch.Tensor') ->torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_features(self, source: 'torch.Tensor', padding_mask: 'Optional[torch.Tensor]'=None, mask: 'bool'=False, ret_conv: 'bool'=False, output_layer: 'Optional[int]'=None, ret_layer_results: 'bool'=False):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features
        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=None if output_layer is None else output_layer - 1)
        res = {'x': x, 'padding_mask': padding_mask, 'features': features, 'layer_results': layer_results}
        feature = res['features'] if ret_conv else res['x']
        if ret_layer_results:
            feature = feature, res['layer_results']
        return feature, res['padding_mask']


class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_state: 'int', n_head: 'int', cross_attention: 'bool'=False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x: 'Tensor', xa: 'Optional[Tensor]'=None, mask: 'Optional[Tensor]'=None, kv_cache: 'Optional[dict]'=None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AudioEncoder(nn.Module):

    def __init__(self, n_mels: 'int', n_ctx: 'int', n_state: 'int', n_head: 'int', n_layer: 'int'):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer('positional_embedding', sinusoids(n_ctx, n_state))
        self.blocks: 'Iterable[ResidualAttentionBlock]' = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: 'Tensor'):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        len_x = x.shape[1]
        len_e = self.positional_embedding.shape[0]
        assert len_x <= len_e, 'incorrect audio shape'
        pos_e = self.positional_embedding[:len_x, :]
        x = x + pos_e
        for block in self.blocks:
            x = block(x)
        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):

    def __init__(self, n_vocab: 'int', n_ctx: 'int', n_state: 'int', n_head: 'int', n_layer: 'int'):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks: 'Iterable[ResidualAttentionBlock]' = nn.ModuleList([ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)])
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer('mask', mask, persistent=False)

    def forward(self, x: 'Tensor', xa: 'Tensor', kv_cache: 'Optional[dict]'=None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset:offset + x.shape[-1]]
        x = x
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight, 0, 1)).float()
        return logits


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AfterDiffusion,
     lambda: ([], {'spec_max': 4, 'spec_min': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (AudioEncoder,
     lambda: ([], {'n_mels': 4, 'n_ctx': 4, 'n_state': 4, 'n_head': 4, 'n_layer': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (BiGRU,
     lambda: ([], {'input_features': 4, 'hidden_features': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (BiLSTM,
     lambda: ([], {'input_features': 4, 'hidden_features': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConformerConvModule,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ConvBlockRes,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ConvLayerBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'bias': 4, 'layer_norm': torch.nn.ReLU()}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (ConvolutionalPositionalEmbedding,
     lambda: ([], {'embed_dim': 4, 'kernel_size': 4, 'groups': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (Depthwise_Separable_Conv1D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Depthwise_Separable_TransposeConv1D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (DiscriminatorP,
     lambda: ([], {'period': 4}),
     lambda: ([torch.rand([4, 1, 4])], {})),
    (DiscriminatorS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {})),
    (ElementwiseAffine,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FeatureProjection,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 512, 512])], {})),
    (FeedForward,
     lambda: ([], {'io_features': 4, 'intermediate_features': 4, 'intermediate_dropout': 0.5, 'output_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Flip,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fp32GroupNorm,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Fp32LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (GLU,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (GLU_Linear,
     lambda: ([], {'input_dim': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 4, 0, 4])], {})),
    (HardConcrete,
     lambda: ([], {'n_in': 4}),
     lambda: ([], {})),
    (Intermediate,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'n_inters': 4, 'n_blocks': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Log,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (MaskedAvgPool1d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MaskedMedianPool1d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MultiHeadAttention,
     lambda: ([], {'n_state': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MultiScaleDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {})),
    (MultiheadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (PositionalConvEmbedding,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 768, 768])], {})),
    (ResBlock1,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResBlock2,
     lambda: ([], {'h': 4, 'channels': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (ResEncoderBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'n_state': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ResidualBlock,
     lambda: ([], {'encoder_hidden': 4, 'residual_channels': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (SamePad,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SelfAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4, 'head_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SnakeBeta,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SpeakerEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 80, 80])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transpose,
     lambda: ([], {'dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {})),
    (TransposeLast,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (WavLMSelfAttention,
     lambda: ([], {'embed_dim': 4, 'total_num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
]

