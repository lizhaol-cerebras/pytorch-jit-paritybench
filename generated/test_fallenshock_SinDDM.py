
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


from inspect import isfunction


import numpy as np


import math


from torch import nn


from functools import partial


import torch.nn.functional as F


from torchvision import utils


from matplotlib import pyplot as plt


import copy


from torch.utils import data


from torchvision import transforms


from torch.optim import Adam


from torch.optim.lr_scheduler import MultiStepLR


import warnings


from typing import Any


from typing import Union


from typing import List


from torchvision.transforms import Compose


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from collections import OrderedDict


from typing import Tuple


import torchvision


from collections.abc import Sequence


import numbers


import torchvision.transforms as T


from torchvision.transforms.functional import InterpolationMode


from torchvision.transforms.functional import _interpolation_modes_from_int


from torchvision.transforms.functional import get_image_num_channels


from torchvision.transforms.functional import get_image_size


from torchvision.transforms.functional import perspective


from torchvision.transforms.functional import crop


from torch.nn import functional as F


from torchvision.transforms import InterpolationMode


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


def exists(x):
    return x is not None


class SinDDMConvBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=1):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, time_emb_dim)) if exists(time_emb_dim) else None
        self.time_reshape = nn.Conv2d(time_emb_dim, dim, 1)
        self.ds_conv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.net = nn.Sequential(nn.Conv2d(dim, dim_out * mult, 3, padding=1), nn.GELU(), nn.Conv2d(dim_out * mult, dim_out, 3, padding=1))
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            condition = self.time_reshape(condition)
            h = h + condition
        h = self.net(h)
        return h + self.res_conv(x)


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class SinDDMNet(nn.Module):

    def __init__(self, dim, out_dim=None, channels=3, with_time_emb=True, multiscale=False, device=None):
        super().__init__()
        self.device = device
        self.channels = channels
        self.multiscale = multiscale
        if with_time_emb:
            time_dim = 32
            if multiscale:
                self.SinEmbTime = SinusoidalPosEmb(time_dim)
                self.SinEmbScale = SinusoidalPosEmb(time_dim)
                self.time_mlp = nn.Sequential(nn.Linear(time_dim * 2, time_dim * 4), nn.GELU(), nn.Linear(time_dim * 4, time_dim))
            else:
                self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 4), nn.GELU(), nn.Linear(time_dim * 4, time_dim))
        else:
            time_dim = None
            self.time_mlp = None
        half_dim = int(dim / 2)
        self.l1 = SinDDMConvBlock(channels, half_dim, time_emb_dim=time_dim)
        self.l2 = SinDDMConvBlock(half_dim, dim, time_emb_dim=time_dim)
        self.l3 = SinDDMConvBlock(dim, dim, time_emb_dim=time_dim)
        self.l4 = SinDDMConvBlock(dim, half_dim, time_emb_dim=time_dim)
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(nn.Conv2d(half_dim, out_dim, 1))

    def forward(self, x, time, scale=None):
        if exists(self.multiscale):
            scale_tensor = torch.ones(size=time.shape) * scale
            t = self.SinEmbTime(time)
            s = self.SinEmbScale(scale_tensor)
            t_s_vec = torch.cat((t, s), dim=1)
            cond_vec = self.time_mlp(t_s_vec)
        else:
            t = self.time_mlp(time) if exists(self.time_mlp) else None
            cond_vec = t
        x = self.l1(x, cond_vec)
        x = self.l2(x, cond_vec)
        x = self.l3(x, cond_vec)
        x = self.l4(x, cond_vec)
        return self.final_conv(x)


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


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda : torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda : torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def thresholded_grad(grad, quantile=0.8):
    """
    Receives the calculated CLIP gradients and outputs the soft-tresholded gradients based on the given quantization.
    Also outputs the mask that corresponds to remaining gradients positions.
    """
    grad_energy = torch.norm(grad, dim=1)
    grad_energy_reshape = torch.reshape(grad_energy, (grad_energy.shape[0], -1))
    enery_quant = torch.quantile(grad_energy_reshape, q=quantile, dim=1, interpolation='nearest')[:, None, None]
    gead_energy_minus_energy_quant = grad_energy - enery_quant
    grad_mask = (gead_energy_minus_energy_quant > 0)[:, None, :, :]
    gead_energy_minus_energy_quant_clamp = torch.clamp(gead_energy_minus_energy_quant, min=0)[:, None, :, :]
    unit_grad_energy = grad / grad_energy[:, None, :, :]
    unit_grad_energy[torch.isnan(unit_grad_energy)] = 0
    sparse_grad = gead_energy_minus_energy_quant_clamp * unit_grad_energy
    return sparse_grad, grad_mask


class MultiScaleGaussianDiffusion(nn.Module):

    def __init__(self, denoise_fn, *, save_interm=False, results_folder='/Results', n_scales, scale_factor, image_sizes, scale_mul=(1, 1), channels=3, timesteps=100, train_full_t=False, scale_losses=None, loss_factor=1, loss_type='l1', betas=None, device=None, reblurring=True, sample_limited_t=False, omega=0):
        super().__init__()
        self.device = device
        self.save_interm = save_interm
        self.results_folder = Path(results_folder)
        self.channels = channels
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.image_sizes = ()
        self.scale_mul = scale_mul
        self.sample_limited_t = sample_limited_t
        self.reblurring = reblurring
        self.img_prev_upsample = None
        self.clip_guided_sampling = False
        self.guidance_sub_iters = None
        self.stop_guidance = None
        self.quantile = 0.8
        self.clip_model = None
        self.clip_strength = None
        self.clip_text = ''
        self.text_embedds = None
        self.text_embedds_hr = None
        self.text_embedds_lr = None
        self.clip_text_features = None
        self.clip_score = []
        self.clip_mask = None
        self.llambda = 0
        self.x_recon_prev = None
        self.clip_roi_bb = []
        self.omega = omega
        self.roi_guided_sampling = False
        self.roi_bbs = []
        self.roi_bbs_stat = []
        self.roi_target_patch = []
        for i in range(n_scales):
            self.image_sizes += (image_sizes[i][1], image_sizes[i][0]),
        self.denoise_fn = denoise_fn
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.num_timesteps_trained = []
        self.num_timesteps_ideal = []
        self.num_timesteps_trained.append(self.num_timesteps)
        self.num_timesteps_ideal.append(self.num_timesteps)
        self.loss_type = loss_type
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
        sigma_t = np.sqrt(1.0 - alphas_cumprod) / np.sqrt(alphas_cumprod)
        if scale_losses is not None:
            for i in range(n_scales - 1):
                self.num_timesteps_ideal.append(int(np.argmax(sigma_t > loss_factor * scale_losses[i])))
                if train_full_t:
                    self.num_timesteps_trained.append(int(timesteps))
                else:
                    self.num_timesteps_trained.append(self.num_timesteps_ideal[i + 1])
        gammas = torch.zeros(size=(n_scales - 1, self.num_timesteps), device=self.device)
        for i in range(n_scales - 1):
            gammas[i, :] = (torch.tensor(sigma_t, device=self.device) / (loss_factor * scale_losses[i])).clamp(min=0, max=1)
        self.register_buffer('gammas', gammas)

    def roi_patch_modification(self, x_recon, scale=0, eta=0.8):
        x_modified = x_recon
        for bb in self.roi_bbs:
            bb = [int(bb_i / np.power(self.scale_factor, self.n_scales - scale - 1)) for bb_i in bb]
            bb_y, bb_x, bb_h, bb_w = bb
            target_patch_resize = F.interpolate(self.roi_target_patch[scale], size=(bb_h, bb_w))
            x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w] = eta * target_patch_resize + (1 - eta) * x_modified[:, :, bb_y:bb_y + bb_h, bb_x:bb_x + bb_w]
        return x_modified

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, s, noise):
        x_recon_ddpm = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        if not self.reblurring or s == 0:
            return x_recon_ddpm, x_recon_ddpm
        else:
            cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)
            x_tm1_mix = (x_recon_ddpm - extract(cur_gammas, t, x_recon_ddpm.shape) * self.img_prev_upsample) / (1 - extract(cur_gammas, t, x_recon_ddpm.shape))
            x_t_mix = x_recon_ddpm
            return x_tm1_mix, x_t_mix

    def q_posterior(self, x_start, x_t_mix, x_t, t, s):
        if not self.reblurring or s == 0:
            posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        elif t[0] > 0:
            x_tm1_mix = x_start
            posterior_variance_low = torch.zeros(x_t.shape, device=self.device)
            posterior_variance_high = 1 - extract(self.alphas_cumprod, t - 1, x_t.shape)
            omega = self.omega
            posterior_variance = (1 - omega) * posterior_variance_low + omega * posterior_variance_high
            posterior_log_variance_clipped = torch.log(posterior_variance.clamp(1e-20, None))
            var_t = posterior_variance
            posterior_mean = extract(self.sqrt_alphas_cumprod, t - 1, x_t.shape) * x_tm1_mix + torch.sqrt(1 - extract(self.alphas_cumprod, t - 1, x_t.shape) - var_t) * (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t_mix) / extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        else:
            posterior_mean = x_start
            posterior_variance = extract(self.posterior_variance, t, x_t.shape)
            posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.enable_grad()
    def p_mean_variance(self, x, t, s, clip_denoised: 'bool'):
        pred_noise = self.denoise_fn(x, t, scale=s)
        x_recon, x_t_mix = self.predict_start_from_noise(x, t=t, s=s, noise=pred_noise)
        cur_gammas = self.gammas[s - 1].reshape(-1).clamp(0, 0.55)
        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (x_recon.clamp(-1.0, 1.0) + 1) * 0.5
            utils.save_image(final_img, str(final_results_folder / f'denoised_t-{t[0]:03}_s-{s}.png'), nrow=4)
        if self.clip_guided_sampling and (self.stop_guidance <= t[0] or s < self.n_scales - 1) and self.guidance_sub_iters[s] > 0:
            if clip_denoised:
                x_recon.clamp_(-1.0, 1.0)
            if self.clip_mask is not None:
                x_recon = x_recon * (1 - self.clip_mask) + ((1 - self.llambda) * self.x_recon_prev + self.llambda * x_recon) * self.clip_mask
            x_recon.requires_grad_(True)
            x_recon_renorm = (x_recon + 1) * 0.5
            for i in range(self.guidance_sub_iters[s]):
                self.clip_model.zero_grad()
                if s > 0:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_hr)
                else:
                    score = -self.clip_model.calculate_clip_loss(x_recon_renorm, self.text_embedds_lr)
                clip_grad = torch.autograd.grad(score, x_recon, create_graph=False)[0]
                if self.clip_mask is None:
                    clip_grad, clip_mask = thresholded_grad(grad=clip_grad, quantile=self.quantile)
                    self.clip_mask = clip_mask.float()
                if self.save_interm:
                    final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
                    final_results_folder.mkdir(parents=True, exist_ok=True)
                    final_mask = self.clip_mask.type(torch.float64)
                    utils.save_image(final_mask, str(final_results_folder / f'clip_mask_s-{s}.png'), nrow=4)
                    utils.save_image((x_recon.clamp(-1.0, 1.0) + 1) * 0.5, str(final_results_folder / f'clip_out_s-{s}_t-{t[0]}_subiter_{i}.png'), nrow=4)
                division_norm = torch.linalg.vector_norm(x_recon * self.clip_mask, dim=(1, 2, 3), keepdim=True) / torch.linalg.vector_norm(clip_grad * self.clip_mask, dim=(1, 2, 3), keepdim=True)
                x_recon += self.clip_strength * division_norm * clip_grad * self.clip_mask
                x_recon.clamp_(-1.0, 1.0)
                x_recon_renorm = (x_recon + 1) * 0.5
                self.clip_score.append(score.detach().cpu())
            self.x_recon_prev = x_recon.detach()
            plt.rcParams['figure.figsize'] = [16, 8]
            plt.plot(self.clip_score)
            plt.grid(True)
            plt.savefig(str(self.results_folder / 'clip_score'))
            plt.clf()
        elif self.roi_guided_sampling and s < self.n_scales - 1:
            x_recon = self.roi_patch_modification(x_recon, scale=s)
        if int(s) > 0 and t[0] > 0 and self.reblurring:
            x_tm1_mix = extract(cur_gammas, t - 1, x_recon.shape) * self.img_prev_upsample + (1 - extract(cur_gammas, t - 1, x_recon.shape)) * x_recon
        else:
            x_tm1_mix = x_recon
        if clip_denoised:
            x_tm1_mix.clamp_(-1.0, 1.0)
            x_t_mix.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_tm1_mix, x_t_mix=x_t_mix, x_t=x, t=t, s=s)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        nonzero_mask_s = torch.tensor([True], device=self.device).float()
        return model_mean + nonzero_mask_s * nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, s):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img, str(final_results_folder / f'input_noise_s-{s}.png'), nrow=4)
        if self.sample_limited_t and s < self.n_scales - 1:
            t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img, str(final_results_folder / f'output_t-{i:03}_s-{s}.png'), nrow=4)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, scale_0_size=None, s=0):
        """
        Sample from the first scale (without conditioning on a previous scale's output).
        """
        if scale_0_size is not None:
            image_size = scale_0_size
        else:
            image_size = self.image_sizes[0]
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size[0], image_size[1]), s=s)

    @torch.no_grad()
    def p_sample_via_scale_loop(self, batch_size, img, s, custom_t=None):
        device = self.betas.device
        if custom_t is None:
            total_t = self.num_timesteps_ideal[min(s, self.n_scales - 1)] - 1
        else:
            total_t = custom_t
        b = batch_size
        self.img_prev_upsample = img
        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img, str(final_results_folder / f'clean_input_s_{s}.png'), nrow=4)
        img = self.q_sample(x_start=img, t=torch.Tensor.expand(torch.tensor(total_t, device=device), batch_size), noise=None)
        if self.save_interm:
            final_results_folder = Path(str(self.results_folder / f'interm_samples_scale_{s}'))
            final_results_folder.mkdir(parents=True, exist_ok=True)
            final_img = (img + 1) * 0.5
            utils.save_image(final_img, str(final_results_folder / f'noisy_input_s_{s}.png'), nrow=4)
        if self.clip_mask is not None:
            if s > 0:
                mul_size = [int(self.image_sizes[s][0] * self.scale_mul[0]), int(self.image_sizes[s][1] * self.scale_mul[1])]
                self.clip_mask = F.interpolate(self.clip_mask, size=mul_size, mode='bilinear')
                self.x_recon_prev = F.interpolate(self.x_recon_prev, size=mul_size, mode='bilinear')
            else:
                self.clip_mask = None
        if self.sample_limited_t and s < self.n_scales - 1:
            t_min = self.num_timesteps_ideal[s + 1]
        else:
            t_min = 0
        for i in tqdm(reversed(range(t_min, total_t)), desc='sampling loop time step', total=total_t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), s)
            if self.save_interm:
                final_img = (img + 1) * 0.5
                utils.save_image(final_img, str(final_results_folder / f'output_t-{i:03}_s-{s}.png'), nrow=4)
        return img

    @torch.no_grad()
    def sample_via_scale(self, batch_size, img, s, scale_mul=(1, 1), custom_sample=False, custom_img_size_idx=0, custom_t=None, custom_image_size=None):
        """
        Sampling at a given scale s conditioned on the output of a previous scale.
        """
        if custom_sample:
            if custom_img_size_idx >= self.n_scales:
                size = self.image_sizes[self.n_scales - 1]
                factor = self.scale_factor ** (custom_img_size_idx + 1 - self.n_scales)
                size = int(size[0] * factor), int(size[1] * factor)
            else:
                size = self.image_sizes[custom_img_size_idx]
        else:
            size = self.image_sizes[s]
        image_size = int(size[0] * scale_mul[0]), int(size[1] * scale_mul[1])
        if custom_image_size is not None:
            image_size = custom_image_size
        img = F.interpolate(img, size=image_size, mode='bilinear')
        return self.p_sample_via_scale_loop(batch_size, img, s, custom_t=custom_t)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda : torch.randn_like(x_start))
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(self, x_start, t, s, noise=None, x_orig=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda : torch.randn_like(x_start))
        if int(s) > 0:
            cur_gammas = self.gammas[s - 1].reshape(-1)
            x_mix = extract(cur_gammas, t, x_start.shape) * x_start + (1 - extract(cur_gammas, t, x_start.shape)) * x_orig
            x_noisy = self.q_sample(x_start=x_mix, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t, s)
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t, s)
        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        elif self.loss_type == 'l1_pred_img':
            if int(s) > 0:
                cur_gammas = self.gammas[s - 1].reshape(-1)
                if t[0] > 0:
                    x_mix_prev = extract(cur_gammas, t - 1, x_start.shape) * x_start + (1 - extract(cur_gammas, t - 1, x_start.shape)) * x_orig
                else:
                    x_mix_prev = x_orig
            else:
                x_mix_prev = x_start
            loss = (x_mix_prev - x_recon).abs().mean()
        else:
            raise NotImplementedError()
        return loss

    def forward(self, x, s, *args, **kwargs):
        if int(s) > 0:
            x_orig = x[0]
            x_recon = x[1]
            b, c, h, w = x_orig.shape
            device = x_orig.device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x_recon, t, s, *args, x_orig=x_orig, **kwargs)
        else:
            b, c, h, w = x[0].shape
            device = x[0].device
            img_size = self.image_sizes[s]
            assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
            t = torch.randint(0, self.num_timesteps_trained[s], (b,), device=device).long()
            return self.p_losses(x[0], t, s, *args, **kwargs)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: 'torch.Tensor'):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: 'torch.Tensor'):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: 'torch.Tensor'):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: 'int', n_head: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: 'torch.Tensor'):
        self.attn_mask = self.attn_mask if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: 'torch.Tensor'):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width: 'int', layers: 'int', heads: 'int', attn_mask: 'torch.Tensor'=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: 'torch.Tensor'):
        return self.resblocks(x)


class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: 'int', patch_size: 'int', width: 'int', layers: 'int', heads: 'int', output_dim: 'int'):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def interpolate_pos_encoding(self, x, w, h):
        positional_embedding = self.positional_embedding.unsqueeze(0)
        patch_size = self.conv1.kernel_size[0]
        npatch = x.shape[1] - 1
        N = positional_embedding.shape[1] - 1
        if npatch == N and w == h:
            return positional_embedding
        class_pos_embed = positional_embedding[:, 0]
        patch_pos_embed = positional_embedding[:, 1:]
        dim = x.shape[-1]
        w0 = w // patch_size
        h0 = h // patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), mode='bicubic')
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x: 'torch.Tensor'):
        x = self.transformer_first_blocks_forward(x)
        x = self.transformer.resblocks[-1](x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def transformer_first_blocks_forward(self, x):
        h, w = x.shape[-2:]
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        positional_embedding = self.interpolate_pos_encoding(x, w, h)
        x = x + positional_embedding
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer.resblocks[:-1](x)
        return x

    @staticmethod
    def attn_cosine_sim(x, eps=1e-08):
        norm = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm @ norm.permute(0, 2, 1), min=eps)
        sim_matrix = x @ x.permute(0, 2, 1) / factor
        return sim_matrix


class CLIP(nn.Module):

    def __init__(self, embed_dim: 'int', image_resolution: 'int', vision_layers: 'Union[Tuple[int, int, int, int], int]', vision_width: 'int', vision_patch_size: 'int', context_length: 'int', vocab_size: 'int', transformer_width: 'int', transformer_heads: 'int', transformer_layers: 'int'):
        super().__init__()
        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers, output_dim=embed_dim, heads=vision_heads, input_resolution=image_resolution, width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim)
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads, attn_mask=self.build_attention_mask())
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith('bn3.weight'):
                        nn.init.zeros_(param)
        proj_std = self.transformer.width ** -0.5 * (2 * self.transformer.layers) ** -0.5
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def calculate_self_sim(self, x: 'torch.Tensor'):
        tokens = self.visual.transformer_first_blocks_forward(x.type(self.dtype))
        tokens = tokens.permute(1, 0, 2)
        ssim = self.visual.attn_cosine_sim(tokens)
        return ssim

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def compose_text_with_templates(text: 'str', templates) ->list:
    return [template.format(text) for template in templates]


def cosine_loss(x, y, scaling=1.2):
    return scaling * (1 - F.cosine_similarity(x, y).mean())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ClipExtractor(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model = clip.load(cfg['clip_model_name'], device=device)[0]
        self.model = model.eval().requires_grad_(False)
        self.text_criterion = cosine_loss
        self.clip_input_size = 224
        self.clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.basic_transform = T.Compose([T.Resize(self.clip_input_size, max_size=380), self.clip_normalize])
        self.augs = T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomApply([T.RandomAffine(degrees=15, translate=(0.1, 0.1), fill=cfg['clip_affine_transform_fill'], interpolation=InterpolationMode.BILINEAR)], p=0.8), T.RandomPerspective(distortion_scale=0.4, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=cfg['clip_affine_transform_fill']), T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.7), T.RandomGrayscale(p=0.15)])
        self.n_aug = cfg['n_aug']

    def augment_input(self, input, n_aug=None, clip_input_size=None):
        if n_aug is None:
            n_aug = self.n_aug
        if clip_input_size is None:
            clip_input_size = self.clip_input_size
        cutouts = []
        cutout = T.Resize(clip_input_size, max_size=320)(input)
        cutout_h, cutout_w = cutout.shape[-2:]
        cutout = self.augs(cutout)
        cutouts.append(cutout)
        sideY, sideX = input.shape[2:4]
        for _ in range(n_aug - 1):
            s = torch.zeros(1).uniform_(0.6, 1).item()
            h = int(sideY * s)
            w = int(sideX * s)
            cutout = T.RandomCrop(size=(h, w))(input)
            cutout = T.Resize((cutout_h, cutout_w))(cutout)
            cutout = self.augs(cutout)
            cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        return cutouts

    def get_image_embedding(self, x, aug=True):
        if aug:
            views = self.augment_input(x)
        else:
            views = self.basic_transform(x)
        if type(views) == list:
            image_embeds = []
            for view in views:
                image_embeds.append(self.encode_image(self.clip_normalize(view)))
            image_embeds = torch.cat(image_embeds)
        else:
            image_embeds = self.encode_image(self.clip_normalize(views))
        return image_embeds

    def encode_image(self, x):
        return self.model.encode_image(x)

    def get_text_embedding(self, text, template, average_embeddings=False):
        if type(text) == str:
            text = [text]
        embeddings = []
        for prompt in text:
            with torch.no_grad():
                embedding = self.model.encode_text(clip.tokenize(compose_text_with_templates(prompt, template)))
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings)
        if average_embeddings:
            embeddings = embeddings.mean(dim=0, keepdim=True)
        return embeddings

    def get_self_sim(self, x):
        x = self.basic_transform(x)
        return self.model.calculate_self_sim(x)

    def calculate_clip_loss(self, outputs, target_embeddings):
        n_embeddings = np.random.randint(1, len(target_embeddings) + 1)
        target_embeddings = target_embeddings[torch.randint(len(target_embeddings), (n_embeddings,))]
        loss = 0.0
        for img in outputs:
            img_e = self.get_image_embedding(img.unsqueeze(0))
            for target_embedding in target_embeddings:
                loss += self.text_criterion(img_e, target_embedding.unsqueeze(0))
        loss /= len(target_embeddings)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentionPool2d,
     lambda: ([], {'spacial_dim': 4, 'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Bottleneck,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (QuickGELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResidualAttentionBlock,
     lambda: ([], {'d_model': 4, 'n_head': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Transformer,
     lambda: ([], {'width': 4, 'layers': 1, 'heads': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (VisionTransformer,
     lambda: ([], {'input_resolution': 4, 'patch_size': 4, 'width': 4, 'layers': 1, 'heads': 4, 'output_dim': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

