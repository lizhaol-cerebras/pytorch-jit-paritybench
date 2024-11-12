
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


from torch.utils.data import DataLoader


import torch


import numpy as np


from torch.utils.data import Dataset


from torchvision import transforms


from torch import nn


from torch.utils.tensorboard import SummaryWriter


import time


from torch.nn.utils import clip_grad_norm_


import math


from torch.utils.data import Subset


from torch.nn import functional as F


from torch.distributions.normal import Normal


from torch.distributions.kl import kl_divergence


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions import Normal


from torch.distributions import kl_divergence


from torch.distributions import RelaxedBernoulli


from torch.distributions.utils import broadcast_all


from torch.optim import Adam


from torch.optim import RMSprop


from collections import defaultdict


from collections import deque


from torchvision.utils import make_grid


import matplotlib


class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
        
        :param x: (B, L)
        :param width: W
        :param height: H
        :return: (B, L + 2, W, H)
        """
        B, L = x.size()
        x = x[:, :, None, None]
        x = x.expand(B, L, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))
        coords = torch.stack((xx, yy), dim=0)
        coords = coords[None].expand(B, 2, height, width)
        x = torch.cat((x, coords), dim=1)
        return x


class CompDecoder(nn.Module):
    """
    Decoder z_comp into component image
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_broadcast = SpatialBroadcast()
        self.decoder = nn.Sequential(nn.Conv2d(arch.z_comp_dim + 2, 32, 3, 1), nn.BatchNorm2d(32), nn.ELU(), nn.Conv2d(32, 32, 3, 1), nn.BatchNorm2d(32), nn.ELU(), nn.Conv2d(32, 32, 3, 1), nn.BatchNorm2d(32), nn.ELU(), nn.Conv2d(32, 32, 3, 1), nn.BatchNorm2d(32), nn.ELU(), nn.Conv2d(32, 3, 1, 1))

    def forward(self, z_comp):
        """
        :param z_comp: (B, L)
        :return: component image (B, 3, H, W)
        """
        h, w = arch.img_shape
        z_comp = self.spatial_broadcast(z_comp, h + 8, w + 8)
        comp = self.decoder(z_comp)
        comp = torch.sigmoid(comp)
        return comp


class CompDecoderStrong(nn.Module):

    def __init__(self):
        super(CompDecoderStrong, self).__init__()
        self.dec = nn.Sequential(nn.Conv2d(arch.z_comp_dim, 256, 1), nn.CELU(), nn.GroupNorm(16, 256), nn.Conv2d(256, 256 * 4 * 4, 1), nn.PixelShuffle(4), nn.CELU(), nn.GroupNorm(16, 256), nn.Conv2d(256, 256, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, 256), nn.Conv2d(256, 128 * 4 * 4, 1), nn.PixelShuffle(4), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 128, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 64 * 2 * 2, 1), nn.PixelShuffle(2), nn.CELU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 64, 3, 1, 1), nn.CELU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 16 * 4 * 4, 1), nn.PixelShuffle(4), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 16, 3, 1, 1), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 16, 3, 1, 1), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 3, 3, 1, 1))

    def forward(self, x):
        """

        :param x: (B, L)
        :return:
        """
        x = x.view(*x.size(), 1, 1)
        comp = torch.sigmoid(self.dec(x))
        return comp


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class CompEncoder(nn.Module):
    """
    Predict component latent parameters given image and predicted mask concatenated
    """

    def __init__(self):
        nn.Module.__init__(self)
        embed_size = arch.img_shape[0] // 16
        self.enc = nn.Sequential(nn.Conv2d(4, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ELU(), nn.Conv2d(32, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ELU(), nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ELU(), Flatten(), nn.Linear(64 * embed_size ** 2, arch.z_comp_dim * 2))

    def forward(self, x):
        """
        Predict component latent parameters given image and predicted mask concatenated
        
        :param x: (B, 3+1, H, W). Image and mask concatenated
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.enc(x)
        z_comp_loc = x[:, :arch.z_comp_dim]
        z_comp_scale = F.softplus(x[:, arch.z_comp_dim:]) + 0.0001
        return z_comp_loc, z_comp_scale


class ImageEncoderBg(nn.Module):
    """Background image encoder"""

    def __init__(self):
        embed_size = arch.img_shape[0] // 16
        nn.Module.__init__(self)
        self.enc = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ELU(), nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ELU(), nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ELU(), nn.Conv2d(64, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ELU(), Flatten(), nn.Linear(64 * embed_size ** 2, arch.img_enc_dim_bg), nn.ELU())

    def forward(self, x):
        """
        Encoder image into a feature vector
        Args:
            x: (B, 3, H, W)
        Returns:
            enc: (B, D)
        """
        return self.enc(x)


class MaskDecoder(nn.Module):
    """Decode z_mask into mask"""

    def __init__(self):
        super(MaskDecoder, self).__init__()
        self.dec = nn.Sequential(nn.Conv2d(arch.z_mask_dim, 256, 1), nn.CELU(), nn.GroupNorm(16, 256), nn.Conv2d(256, 256 * 4 * 4, 1), nn.PixelShuffle(4), nn.CELU(), nn.GroupNorm(16, 256), nn.Conv2d(256, 256, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, 256), nn.Conv2d(256, 128 * 2 * 2, 1), nn.PixelShuffle(2), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 128, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 64 * 4 * 4, 1), nn.PixelShuffle(4), nn.CELU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 64, 3, 1, 1), nn.CELU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 16 * 4 * 4, 1), nn.PixelShuffle(4), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 16, 3, 1, 1), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 16, 3, 1, 1), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, z_mask):
        """
        Decode z_mask into mask
        
        :param z_mask: (B, D)
        :return: mask: (B, 1, H, W)
        """
        B = z_mask.size(0)
        z_mask = z_mask.view(B, -1, 1, 1)
        mask = torch.sigmoid(self.dec(z_mask))
        return mask


class PredictComp(nn.Module):
    """
    Predict component latents given mask latent
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(nn.Linear(arch.z_mask_dim, arch.predict_comp_hidden_dim), nn.ELU(), nn.Linear(arch.predict_comp_hidden_dim, arch.predict_comp_hidden_dim), nn.ELU(), nn.Linear(arch.predict_comp_hidden_dim, arch.z_comp_dim * 2))

    def forward(self, h):
        """
        :param h: (B, D) hidden state from rnn_mask
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.mlp(h)
        z_comp_loc = x[:, :arch.z_comp_dim]
        z_comp_scale = F.softplus(x[:, arch.z_comp_dim:]) + 0.0001
        return z_comp_loc, z_comp_scale


class PredictMask(nn.Module):
    """
    Predict z_mask given states from rnn. Used in inference
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(arch.rnn_mask_hidden_dim, arch.z_mask_dim * 2)

    def forward(self, h):
        """
        Predict z_mask given states from rnn. Used in inference
    
        :param h: hidden state from rnn_mask
        :return:
            z_mask_loc: (B, D)
            z_mask_scale: (B, D)
        
        """
        x = self.fc(h)
        z_mask_loc = x[:, :arch.z_mask_dim]
        z_mask_scale = F.softplus(x[:, arch.z_mask_dim:]) + 0.0001
        return z_mask_loc, z_mask_scale


class SpaceBg(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.image_enc = ImageEncoderBg()
        self.rnn_mask = nn.LSTMCell(arch.z_mask_dim + arch.img_enc_dim_bg, arch.rnn_mask_hidden_dim)
        self.rnn_mask_h = nn.Parameter(torch.zeros(arch.rnn_mask_hidden_dim))
        self.rnn_mask_c = nn.Parameter(torch.zeros(arch.rnn_mask_hidden_dim))
        self.z_mask_0 = nn.Parameter(torch.zeros(arch.z_mask_dim))
        self.predict_mask = PredictMask()
        self.mask_decoder = MaskDecoder()
        self.comp_encoder = CompEncoder()
        if arch.K > 1:
            self.comp_decoder = CompDecoder()
        else:
            self.comp_decoder = CompDecoderStrong()
        self.rnn_mask_prior = nn.LSTMCell(arch.z_mask_dim, arch.rnn_mask_prior_hidden_dim)
        self.rnn_mask_h_prior = nn.Parameter(torch.zeros(arch.rnn_mask_prior_hidden_dim))
        self.rnn_mask_c_prior = nn.Parameter(torch.zeros(arch.rnn_mask_prior_hidden_dim))
        self.predict_mask_prior = PredictMask()
        self.predict_comp_prior = PredictComp()
        self.bg_sigma = arch.bg_sigma

    def anneal(self, global_step):
        pass

    def forward(self, x, global_step):
        """
        Background inference backward pass
        
        :param x: shape (B, C, H, W)
        :param global_step: global training step
        :return:
            bg_likelihood: (B, 3, H, W)
            bg: (B, 3, H, W)
            kl_bg: (B,)
            log: a dictionary containing things for visualization
        """
        B = x.size(0)
        x_enc = self.image_enc(x)
        masks = []
        z_masks = []
        z_mask_posteriors = []
        z_comp_posteriors = []
        z_mask = self.z_mask_0.expand(B, arch.z_mask_dim)
        h = self.rnn_mask_h.expand(B, arch.rnn_mask_hidden_dim)
        c = self.rnn_mask_c.expand(B, arch.rnn_mask_hidden_dim)
        for i in range(arch.K):
            rnn_input = torch.cat((z_mask, x_enc), dim=1)
            h, c = self.rnn_mask(rnn_input, (h, c))
            z_mask_loc, z_mask_scale = self.predict_mask(h)
            z_mask_post = Normal(z_mask_loc, z_mask_scale)
            z_mask = z_mask_post.rsample()
            z_masks.append(z_mask)
            z_mask_posteriors.append(z_mask_post)
            mask = self.mask_decoder(z_mask)
            masks.append(mask)
        masks = torch.stack(masks, dim=1)
        masks = self.SBP(masks)
        B, K, _, H, W = masks.size()
        masks = masks.view(B * K, 1, H, W)
        comp_vae_input = torch.cat(((masks + 1e-05).log(), x[:, None].repeat(1, K, 1, 1, 1).view(B * K, 3, H, W)), dim=1)
        z_comp_loc, z_comp_scale = self.comp_encoder(comp_vae_input)
        z_comp_post = Normal(z_comp_loc, z_comp_scale)
        z_comp = z_comp_post.rsample()
        z_comp_loc_reshape = z_comp_loc.view(B, K, -1)
        z_comp_scale_reshape = z_comp_scale.view(B, K, -1)
        for i in range(arch.K):
            z_comp_post_this = Normal(z_comp_loc_reshape[:, i], z_comp_scale_reshape[:, i])
            z_comp_posteriors.append(z_comp_post_this)
        comps = self.comp_decoder(z_comp)
        comps = comps.view(B, K, 3, H, W)
        masks = masks.view(B, K, 1, H, W)
        comp_dist = Normal(comps, torch.full_like(comps, self.bg_sigma))
        log_likelihoods = comp_dist.log_prob(x[:, None].expand_as(comps))
        log_sum = log_likelihoods + (masks + 1e-05).log()
        bg_likelihood = torch.logsumexp(log_sum, dim=1)
        bg = (comps * masks).sum(dim=1)
        z_mask_total_kl = 0.0
        z_comp_total_kl = 0.0
        h = self.rnn_mask_h_prior.expand(B, arch.rnn_mask_prior_hidden_dim)
        c = self.rnn_mask_c_prior.expand(B, arch.rnn_mask_prior_hidden_dim)
        for i in range(arch.K):
            z_mask_loc_prior, z_mask_scale_prior = self.predict_mask_prior(h)
            z_mask_prior = Normal(z_mask_loc_prior, z_mask_scale_prior)
            z_comp_loc_prior, z_comp_scale_prior = self.predict_comp_prior(z_masks[i])
            z_comp_prior = Normal(z_comp_loc_prior, z_comp_scale_prior)
            z_mask_kl = kl_divergence(z_mask_posteriors[i], z_mask_prior).sum(dim=1)
            z_comp_kl = kl_divergence(z_comp_posteriors[i], z_comp_prior).sum(dim=1)
            z_mask_total_kl += z_mask_kl
            z_comp_total_kl += z_comp_kl
            h, c = self.rnn_mask_prior(z_masks[i], (h, c))
        kl_bg = z_mask_total_kl + z_comp_total_kl
        log = {'comps': comps, 'masks': masks, 'bg': bg, 'kl_bg': kl_bg}
        return bg_likelihood, bg, kl_bg, log

    @staticmethod
    def SBP(masks):
        """
        Stick breaking process to produce masks
        :param: masks (B, K, 1, H, W). In range (0, 1)
        :return: (B, K, 1, H, W)
        """
        B, K, _, H, W = masks.size()
        remained = torch.ones_like(masks[:, 0])
        new_masks = []
        for k in range(K):
            if k < K - 1:
                mask = masks[:, k] * remained
            else:
                mask = remained
            remained = remained - mask
            new_masks.append(mask)
        new_masks = torch.stack(new_masks, dim=1)
        return new_masks


class GlimpseDec(nn.Module):
    """Decoder z_what into reconstructed objects"""

    def __init__(self):
        super(GlimpseDec, self).__init__()
        self.dec = nn.Sequential(nn.Conv2d(arch.z_what_dim, 256, 1), nn.CELU(), nn.GroupNorm(16, 256), nn.Conv2d(256, 128 * 2 * 2, 1), nn.PixelShuffle(2), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 128, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 128 * 2 * 2, 1), nn.PixelShuffle(2), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 128, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 64 * 2 * 2, 1), nn.PixelShuffle(2), nn.CELU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 64, 3, 1, 1), nn.CELU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 32 * 2 * 2, 1), nn.PixelShuffle(2), nn.CELU(), nn.GroupNorm(8, 32), nn.Conv2d(32, 32, 3, 1, 1), nn.CELU(), nn.GroupNorm(8, 32), nn.Conv2d(32, 16 * 2 * 2, 1), nn.PixelShuffle(2), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 16, 3, 1, 1), nn.CELU(), nn.GroupNorm(4, 16))
        self.dec_o = nn.Conv2d(16, 3, 3, 1, 1)
        self.dec_alpha = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        """
        Decoder z_what into glimpse

        :param x: (B, D)
        :return:
            o_att: (B, 3, H, W)
            alpha_att: (B, 1, H, W)
        """
        x = self.dec(x.view(x.size(0), -1, 1, 1))
        o = torch.sigmoid(self.dec_o(x))
        alpha = torch.sigmoid(self.dec_alpha(x))
        return o, alpha


class NumericalRelaxedBernoulli(RelaxedBernoulli):
    """
    This is a bit weird. In essence it is just RelaxedBernoulli with logit as input.
    """

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)
        out = self.temperature.log() + diff - 2 * diff.exp().log1p()
        return out


class ImgEncoderFg(nn.Module):
    """
    Foreground image encoder.
    """

    def __init__(self):
        super(ImgEncoderFg, self).__init__()
        assert arch.G in [4, 8, 16]
        last_stride = 2 if arch.G in [8, 4] else 1
        second_to_last_stride = 2 if arch.G in [4] else 1
        self.enc = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 32, 4, 2, 1), nn.CELU(), nn.GroupNorm(8, 32), nn.Conv2d(32, 64, 4, 2, 1), nn.CELU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 128, 3, second_to_last_stride, 1), nn.CELU(), nn.GroupNorm(16, 128), nn.Conv2d(128, 256, 3, last_stride, 1), nn.CELU(), nn.GroupNorm(32, 256), nn.Conv2d(256, arch.img_enc_dim_fg, 1), nn.CELU(), nn.GroupNorm(16, arch.img_enc_dim_fg))
        self.enc_lat = nn.Sequential(nn.Conv2d(arch.img_enc_dim_fg, arch.img_enc_dim_fg, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, arch.img_enc_dim_fg), nn.Conv2d(arch.img_enc_dim_fg, arch.img_enc_dim_fg, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, arch.img_enc_dim_fg))
        self.enc_cat = nn.Sequential(nn.Conv2d(arch.img_enc_dim_fg * 2, 128, 3, 1, 1), nn.CELU(), nn.GroupNorm(16, 128))
        self.z_scale_net = nn.Conv2d(128, arch.z_where_scale_dim * 2, 1)
        self.z_shift_net = nn.Conv2d(128, arch.z_where_shift_dim * 2, 1)
        self.z_pres_net = nn.Conv2d(128, arch.z_pres_dim, 1)
        self.z_depth_net = nn.Conv2d(128, arch.z_depth_dim * 2, 1)
        offset_y, offset_x = torch.meshgrid([torch.arange(arch.G), torch.arange(arch.G)])
        self.register_buffer('offset', torch.stack((offset_x, offset_y), dim=0).float())

    def forward(self, x, tau):
        """
        Given image, infer z_pres, z_depth, z_where

        :param x: (B, 3, H, W)
        :param tau: temperature for the relaxed bernoulli
        :return
            z_pres: (B, G*G, 1)
            z_depth: (B, G*G, 1)
            z_scale: (B, G*G, 2)
            z_shift: (B, G*G, 2)
            z_where: (B, G*G, 4)
            z_pres_logits: (B, G*G, 1)
            z_depth_post: Normal, (B, G*G, 1)
            z_scale_post: Normal, (B, G*G, 2)
            z_shift_post: Normal, (B, G*G, 2)
        """
        B = x.size(0)
        img_enc = self.enc(x)
        lateral_enc = self.enc_lat(img_enc)
        cat_enc = self.enc_cat(torch.cat((img_enc, lateral_enc), dim=1))

        def reshape(*args):
            """(B, D, G, G) -> (B, G*G, D)"""
            out = []
            for x in args:
                B, D, G, G = x.size()
                y = x.permute(0, 2, 3, 1).view(B, G * G, D)
                out.append(y)
            return out[0] if len(args) == 1 else out
        z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(cat_enc))
        z_pres_logits = reshape(z_pres_logits)
        z_pres_post = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=tau)
        z_pres_y = z_pres_post.rsample()
        z_pres = torch.sigmoid(z_pres_y)
        z_depth_mean, z_depth_std = self.z_depth_net(cat_enc).chunk(2, 1)
        z_depth_mean, z_depth_std = reshape(z_depth_mean, z_depth_std)
        z_depth_std = F.softplus(z_depth_std)
        z_depth_post = Normal(z_depth_mean, z_depth_std)
        z_depth = z_depth_post.rsample()
        scale_std_bias = 1e-15
        z_scale_mean, _z_scale_std = self.z_scale_net(cat_enc).chunk(2, 1)
        z_scale_std = F.softplus(_z_scale_std) + scale_std_bias
        z_scale_mean, z_scale_std = reshape(z_scale_mean, z_scale_std)
        z_scale_post = Normal(z_scale_mean, z_scale_std)
        z_scale = z_scale_post.rsample()
        z_shift_mean, z_shift_std = self.z_shift_net(cat_enc).chunk(2, 1)
        z_shift_std = F.softplus(z_shift_std)
        z_shift_mean, z_shift_std = reshape(z_shift_mean, z_shift_std)
        z_shift_post = Normal(z_shift_mean, z_shift_std)
        z_shift = z_shift_post.rsample()
        z_scale = z_scale.sigmoid()
        offset = self.offset.permute(1, 2, 0).view(arch.G ** 2, 2)
        z_shift = 2.0 / arch.G * (offset + 0.5 + z_shift.tanh()) - 1
        z_where = torch.cat((z_scale, z_shift), dim=-1)
        assert z_pres.size() == (B, arch.G ** 2, 1) and z_depth.size() == (B, arch.G ** 2, 1) and z_shift.size() == (B, arch.G ** 2, 2) and z_scale.size() == (B, arch.G ** 2, 2) and z_where.size() == (B, arch.G ** 2, 4)
        return z_pres, z_depth, z_scale, z_shift, z_where, z_pres_logits, z_depth_post, z_scale_post, z_shift_post


class ZWhatEnc(nn.Module):

    def __init__(self):
        super(ZWhatEnc, self).__init__()
        self.enc_cnn = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.CELU(), nn.GroupNorm(4, 16), nn.Conv2d(16, 32, 4, 2, 1), nn.CELU(), nn.GroupNorm(8, 32), nn.Conv2d(32, 32, 3, 1, 1), nn.CELU(), nn.GroupNorm(4, 32), nn.Conv2d(32, 64, 4, 2, 1), nn.CELU(), nn.GroupNorm(8, 64), nn.Conv2d(64, 128, 4, 2, 1), nn.CELU(), nn.GroupNorm(8, 128), nn.Conv2d(128, 256, 4), nn.CELU(), nn.GroupNorm(16, 256))
        self.enc_what = nn.Linear(256, arch.z_what_dim * 2)

    def forward(self, x):
        """
        Encode a (32, 32) glimpse into z_what

        :param x: (B, C, H, W)
        :return:
            z_what: (B, D)
            z_what_post: (B, D)
        """
        x = self.enc_cnn(x)
        z_what_mean, z_what_std = self.enc_what(x.flatten(start_dim=1)).chunk(2, -1)
        z_what_std = F.softplus(z_what_std)
        z_what_post = Normal(z_what_mean, z_what_std)
        z_what = z_what_post.rsample()
        return z_what, z_what_post


def get_boundary_kernel(kernel_size=32, sigma=20, channels=1, beta=1.0):
    """
    TODO: This function is no longer used.
    """
    x_coord = torch.arange(kernel_size)
    boundary_kernel = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).float()
    part_sum = 1.0
    boundary = int((kernel_size - sigma) / 2)
    num_center = pow(kernel_size - boundary - boundary, 2)
    num_boundary = kernel_size * kernel_size - num_center
    boundary_kernel.data[:, :] = -part_sum / (num_boundary + num_center) * beta
    boundary_kernel.data[boundary:kernel_size - boundary, boundary:kernel_size - boundary] = part_sum / (num_boundary + num_center)
    boundary_kernel = boundary_kernel.view(1, 1, kernel_size, kernel_size)
    boundary_kernel = boundary_kernel.repeat(channels, 1, 1, 1)
    boundary_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels, bias=False)
    boundary_filter.weight.data = boundary_kernel
    boundary_filter.weight.requires_grad = False
    return boundary_filter


def get_boundary_kernel_new(kernel_size=32, boundary_width=6):
    """
    Will return something like this:
    ============
    =          =
    =          =
    ============
    size will be (kernel_size, kernel_size)
    """
    filter = torch.zeros(kernel_size, kernel_size)
    filter[:, :] = 1.0 / kernel_size ** 2
    filter[boundary_width:kernel_size - boundary_width, boundary_width:kernel_size - boundary_width] = 0.0
    return filter


def kl_divergence_bern_bern(z_pres_logits, prior_pres_prob, eps=1e-15):
    """
    Compute kl divergence of two Bernoulli distributions
    :param z_pres_logits: (B, ...)
    :param prior_pres_prob: float
    :return: kl divergence, (B, ...)
    """
    z_pres_probs = torch.sigmoid(z_pres_logits)
    kl = z_pres_probs * (torch.log(z_pres_probs + eps) - torch.log(prior_pres_prob + eps)) + (1 - z_pres_probs) * (torch.log(1 - z_pres_probs + eps) - torch.log(1 - prior_pres_prob + eps))
    return kl


def linear_annealing(device, step, start_step, end_step, start_value, end_value):
    """
    Linear annealing
    
    :param x: original value. Only for getting device
    :param step: current global step
    :param start_step: when to start changing value
    :param end_step: when to stop changing value
    :param start_value: initial value
    :param end_value: final value
    :return:
    """
    if step <= start_step:
        x = torch.tensor(start_value, device=device)
    elif start_step < step < end_step:
        slope = (end_value - start_value) / (end_step - start_step)
        x = torch.tensor(start_value + slope * (step - start_step), device=device)
    else:
        x = torch.tensor(end_value, device=device)
    return x


def spatial_transform(image, z_where, out_dims, inverse=False):
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """
    theta = torch.zeros(2, 3).repeat(image.shape[0], 1, 1)
    theta[:, 0, 0] = z_where[:, 0] if not inverse else 1 / (z_where[:, 0] + 1e-09)
    theta[:, 1, 1] = z_where[:, 1] if not inverse else 1 / (z_where[:, 1] + 1e-09)
    theta[:, 0, -1] = z_where[:, 2] if not inverse else -z_where[:, 2] / (z_where[:, 0] + 1e-09)
    theta[:, 1, -1] = z_where[:, 3] if not inverse else -z_where[:, 3] / (z_where[:, 1] + 1e-09)
    grid = F.affine_grid(theta, torch.Size(out_dims))
    return F.grid_sample(image, grid)


class SpaceFg(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.img_encoder = ImgEncoderFg()
        self.z_what_net = ZWhatEnc()
        self.glimpse_dec = GlimpseDec()
        self.boundary_kernel = get_boundary_kernel_new(kernel_size=32, boundary_width=6)
        self.fg_sigma = arch.fg_sigma
        self.register_buffer('tau', torch.tensor(arch.tau_start_value))
        self.register_buffer('prior_z_pres_prob', torch.tensor(arch.z_pres_start_value))
        self.register_buffer('prior_what_mean', torch.zeros(1))
        self.register_buffer('prior_what_std', torch.ones(1))
        self.register_buffer('prior_depth_mean', torch.zeros(1))
        self.register_buffer('prior_depth_std', torch.ones(1))
        self.prior_scale_mean_new = torch.tensor(arch.z_scale_mean_start_value)
        self.prior_scale_std_new = torch.tensor(arch.z_scale_std_value)
        self.prior_shift_mean_new = torch.tensor(0.0)
        self.prior_shift_std_new = torch.tensor(1.0)
        self.boundary_filter = get_boundary_kernel(sigma=20)
        self.register_buffer('prior_scale_mean', torch.tensor([arch.z_scale_mean_start_value] * 2).view(arch.z_where_scale_dim, 1, 1))
        self.register_buffer('prior_scale_std', torch.tensor([arch.z_scale_std_value] * 2).view(arch.z_where_scale_dim, 1, 1))
        self.register_buffer('prior_shift_mean', torch.tensor([0.0, 0.0]).view(arch.z_where_shift_dim, 1, 1))
        self.register_buffer('prior_shift_std', torch.tensor([1.0, 1.0]).view(arch.z_where_shift_dim, 1, 1))

    @property
    def z_what_prior(self):
        return Normal(self.prior_what_mean, self.prior_what_std)

    @property
    def z_depth_prior(self):
        return Normal(self.prior_depth_mean, self.prior_depth_std)

    @property
    def z_scale_prior(self):
        return Normal(self.prior_scale_mean_new, self.prior_scale_std_new)

    @property
    def z_shift_prior(self):
        return Normal(self.prior_shift_mean_new, self.prior_shift_std_new)

    def anneal(self, global_step):
        """
        Update everything

        :param global_step: global step (training)
        :return:
        """
        self.prior_z_pres_prob = linear_annealing(self.prior_z_pres_prob.device, global_step, arch.z_pres_start_step, arch.z_pres_end_step, arch.z_pres_start_value, arch.z_pres_end_value)
        self.prior_scale_mean_new = linear_annealing(self.prior_z_pres_prob.device, global_step, arch.z_scale_mean_start_step, arch.z_scale_mean_end_step, arch.z_scale_mean_start_value, arch.z_scale_mean_end_value)
        self.tau = linear_annealing(self.tau.device, global_step, arch.tau_start_step, arch.tau_end_step, arch.tau_start_value, arch.tau_end_value)

    def forward(self, x, globel_step):
        """
        Forward pass

        :param x: (B, 3, H, W)
        :param globel_step: global step (training)
        :return:
            fg_likelihood: (B, 3, H, W)
            y_nobg: (B, 3, H, W), foreground reconstruction
            alpha_map: (B, 1, H, W)
            kl: (B,) total foreground kl
            boundary_loss: (B,)
            log: a dictionary containing anything we need for visualization
        """
        B = x.size(0)
        self.anneal(globel_step)
        z_pres, z_depth, z_scale, z_shift, z_where, z_pres_logits, z_depth_post, z_scale_post, z_shift_post = self.img_encoder(x, self.tau)
        x_repeat = torch.repeat_interleave(x, arch.G ** 2, dim=0)
        x_att = spatial_transform(x_repeat, z_where.view(B * arch.G ** 2, 4), (B * arch.G ** 2, 3, arch.glimpse_size, arch.glimpse_size), inverse=False)
        z_what, z_what_post = self.z_what_net(x_att)
        o_att, alpha_att = self.glimpse_dec(z_what)
        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)
        y_att = alpha_att_hat * o_att
        importance_map = alpha_att_hat * 100.0 * torch.sigmoid(-z_depth.view(B * arch.G ** 2, 1, 1, 1))
        importance_map_full_res = spatial_transform(importance_map, z_where.view(B * arch.G ** 2, 4), (B * arch.G ** 2, 1, *arch.img_shape), inverse=True)
        importance_map_full_res = importance_map_full_res.view(B, arch.G ** 2, 1, *arch.img_shape)
        importance_map_full_res_norm = torch.softmax(importance_map_full_res, dim=1)
        y_each_cell = spatial_transform(y_att, z_where.view(B * arch.G ** 2, 4), (B * arch.G ** 2, 3, *arch.img_shape), inverse=True).view(B, arch.G ** 2, 3, *arch.img_shape)
        y_nobg = (y_each_cell * importance_map_full_res_norm).sum(dim=1)
        alpha_map = spatial_transform(alpha_att_hat, z_where.view(B * arch.G ** 2, 4), (B * arch.G ** 2, 1, *arch.img_shape), inverse=True).view(B, arch.G ** 2, 1, *arch.img_shape)
        alpha_map = (alpha_map * importance_map_full_res_norm).sum(dim=1)
        kl_z_pres = kl_divergence_bern_bern(z_pres_logits, self.prior_z_pres_prob)
        kl_z_depth = kl_divergence(z_depth_post, self.z_depth_prior)
        kl_z_scale = kl_divergence(z_scale_post, self.z_scale_prior)
        kl_z_shift = kl_divergence(z_shift_post, self.z_shift_prior)
        z_what = z_what.view(B, arch.G ** 2, arch.z_what_dim)
        z_what_post = Normal(*[x.view(B, arch.G ** 2, arch.z_what_dim) for x in [z_what_post.mean, z_what_post.stddev]])
        kl_z_what = kl_divergence(z_what_post, self.z_what_prior)
        assert kl_z_pres.size() == (B, arch.G ** 2, 1) and kl_z_depth.size() == (B, arch.G ** 2, 1) and kl_z_scale.size() == (B, arch.G ** 2, 2) and kl_z_shift.size() == (B, arch.G ** 2, 2) and kl_z_what.size() == (B, arch.G ** 2, arch.z_what_dim)
        kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what = [x.flatten(start_dim=1).sum(1) for x in [kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what]]
        kl_z_where = kl_z_scale + kl_z_shift
        boundary_kernel = self.boundary_kernel[None, None]
        boundary_kernel = boundary_kernel * z_pres.view(B * arch.G ** 2, 1, 1, 1)
        boundary_map = spatial_transform(boundary_kernel, z_where.view(B * arch.G ** 2, 4), (B * arch.G ** 2, 1, *arch.img_shape), inverse=True).view(B, arch.G ** 2, 1, *arch.img_shape)
        boundary_map = boundary_map.sum(dim=1)
        boundary_map = boundary_map * 1000
        overlap = boundary_map * alpha_map
        p_boundary = Normal(0, 0.7)
        boundary_loss = p_boundary.log_prob(overlap)
        boundary_loss = boundary_loss.flatten(start_dim=1).sum(1)
        boundary_loss = -boundary_loss
        fg_dist = Normal(y_nobg, self.fg_sigma)
        fg_likelihood = fg_dist.log_prob(x)
        kl = kl_z_what + kl_z_where + kl_z_pres + kl_z_depth
        if not arch.boundary_loss or globel_step > arch.bl_off_step:
            boundary_loss = boundary_loss * 0.0
        assert z_pres.size() == (B, arch.G ** 2, 1) and z_depth.size() == (B, arch.G ** 2, 1) and z_scale.size() == (B, arch.G ** 2, 2) and z_shift.size() == (B, arch.G ** 2, 2) and z_where.size() == (B, arch.G ** 2, 4) and z_what.size() == (B, arch.G ** 2, arch.z_what_dim)
        log = {'fg': y_nobg, 'z_what': z_what, 'z_where': z_where, 'z_pres': z_pres, 'z_scale': z_scale, 'z_shift': z_shift, 'z_depth': z_depth, 'z_pres_prob': torch.sigmoid(z_pres_logits), 'prior_z_pres_prob': self.prior_z_pres_prob.unsqueeze(0), 'o_att': o_att, 'alpha_att_hat': alpha_att_hat, 'alpha_att': alpha_att, 'alpha_map': alpha_map, 'boundary_loss': boundary_loss, 'boundary_map': boundary_map, 'importance_map_full_res_norm': importance_map_full_res_norm, 'kl_z_what': kl_z_what, 'kl_z_pres': kl_z_pres, 'kl_z_scale': kl_z_scale, 'kl_z_shift': kl_z_shift, 'kl_z_depth': kl_z_depth, 'kl_z_where': kl_z_where}
        return fg_likelihood, y_nobg, alpha_map, kl, boundary_loss, log


class Space(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.fg_module = SpaceFg()
        self.bg_module = SpaceBg()

    def forward(self, x, global_step):
        """
        Inference.
        
        :param x: (B, 3, H, W)
        :param global_step: global training step
        :return:
            loss: a scalor. Note it will be better to return (B,)
            log: a dictionary for visualization
        """
        bg_likelihood, bg, kl_bg, log_bg = self.bg_module(x, global_step)
        fg_likelihood, fg, alpha_map, kl_fg, loss_boundary, log_fg = self.fg_module(x, global_step)
        if global_step and global_step < arch.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, arch.fix_alpha_value)
        fg_likelihood = fg_likelihood + (alpha_map + 1e-05).log()
        bg_likelihood = bg_likelihood + (1 - alpha_map + 1e-05).log()
        log_like = torch.stack((fg_likelihood, bg_likelihood), dim=1)
        log_like = torch.logsumexp(log_like, dim=1)
        log_like = log_like.flatten(start_dim=1).sum(1)
        y = alpha_map * fg + (1.0 - alpha_map) * bg
        elbo = log_like - kl_bg - kl_fg
        loss = (-elbo + loss_boundary).mean()
        log = {'imgs': x, 'y': y, 'mse': ((y - x) ** 2).flatten(start_dim=1).sum(dim=1), 'log_like': log_like}
        log.update(log_fg)
        log.update(log_bg)
        return loss, log


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

