
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


from torch import nn


from torch.nn import functional as F


import numpy as np


from typing import Dict


import torchvision.transforms as T


from matplotlib import use


import torch.nn as nn


import torch.nn.functional as F


from math import pi


from torchvision.models import resnet34


import matplotlib.pyplot as plt


import random


from functools import partial


class CoarseMaskHead(nn.Module):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(self, cfg, input_shape: 'ShapeSpec'):
        """
        The following attributes are parsed from config:
            conv_dim: the output dimension of the conv layers
            fc_dim: the feature dimenstion of the FC layers
            num_fc: the number of FC layers
            output_side_resolution: side resolution of the output square mask prediction
        """
        super(CoarseMaskHead, self).__init__()
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.fc_dim = cfg.MODEL.ROI_MASK_HEAD.FC_DIM
        num_fc = cfg.MODEL.ROI_MASK_HEAD.NUM_FC
        self.output_side_resolution = cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION
        self.input_channels = input_shape.channels
        self.input_h = input_shape.height
        self.input_w = input_shape.width
        self.conv_layers = []
        if self.input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(self.input_channels, conv_dim, kernel_size=1, stride=1, padding=0, bias=True, activation=F.relu)
            self.conv_layers.append(self.reduce_channel_dim_conv)
        self.reduce_spatial_dim_conv = Conv2d(conv_dim, conv_dim, kernel_size=2, stride=2, padding=0, bias=True, activation=F.relu)
        self.conv_layers.append(self.reduce_spatial_dim_conv)
        input_dim = conv_dim * self.input_h * self.input_w
        input_dim //= 4
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(input_dim, self.fc_dim)
            self.add_module('coarse_mask_fc{}'.format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = self.fc_dim
        output_dim = self.num_classes * self.output_side_resolution * self.output_side_resolution
        self.prediction = nn.Linear(self.fc_dim, output_dim)
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)
        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, self.input_channels, self.input_h, self.input_w)
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        return self.prediction(x).view(N, self.num_classes, self.output_side_resolution, self.output_side_resolution)


class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, cfg, input_shape: 'ShapeSpec'):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardPointHead, self).__init__()
        num_classes = cfg.MODEL.POINT_HEAD.NUM_CLASSES
        fc_dim = cfg.MODEL.POINT_HEAD.FC_DIM
        num_fc = cfg.MODEL.POINT_HEAD.NUM_FC
        cls_agnostic_mask = cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK
        self.coarse_pred_each_layer = cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER
        input_channels = input_shape.channels
        fc_dim_in = input_channels + num_classes
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module('fc{}'.format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)
        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = cat((x, coarse_features), dim=1)
        return self.predictor(x)


def build_point_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.POINT_HEAD.NAME
    return POINT_HEAD_REGISTRY.get(head_name)(cfg, input_channels)


def calculate_uncertainty(sem_seg_logits):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncerainty as the
        difference between top first and top second predicted logits.

    Args:
        mask_logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.

    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)
    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + point_indices % W * w_step
    point_coords[:, :, 1] = h_step / 2.0 + point_indices // W * h_step
    return point_indices, point_coords


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = cat([point_coords, torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device)], dim=1)
    return point_coords


class PointRendSemSegHead(nn.Module):
    """
    A semantic segmentation head that combines a head set in `POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME`
    and a point head set in `MODEL.POINT_HEAD.NAME`.
    """

    def __init__(self, cfg, input_shape: 'Dict[str, ShapeSpec]'):
        super().__init__()
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.coarse_sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(cfg.MODEL.POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME)(cfg, input_shape)
        self._init_point_head(cfg, input_shape)

    def _init_point_head(self, cfg, input_shape: 'Dict[str, ShapeSpec]'):
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.in_features = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.train_num_points = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.oversample_ratio = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        self.subdivision_steps = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.subdivision_num_points = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        in_channels = np.sum([feature_channels[f] for f in self.in_features])
        self.point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

    def forward(self, features, targets=None):
        coarse_sem_seg_logits = self.coarse_sem_seg_head.layers(features)
        if self.training:
            losses = self.coarse_sem_seg_head.losses(coarse_sem_seg_logits, targets)
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(coarse_sem_seg_logits, calculate_uncertainty, self.train_num_points, self.oversample_ratio, self.importance_sample_ratio)
            coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
            fine_grained_features = cat([point_sample(features[in_feature], point_coords, align_corners=False) for in_feature in self.in_features], dim=1)
            point_logits = self.point_head(fine_grained_features, coarse_features)
            point_targets = point_sample(targets.unsqueeze(1).to(torch.float), point_coords, mode='nearest', align_corners=False).squeeze(1)
            losses['loss_sem_seg_point'] = F.cross_entropy(point_logits, point_targets, reduction='mean', ignore_index=self.ignore_value)
            return None, losses
        else:
            sem_seg_logits = coarse_sem_seg_logits.clone()
            for _ in range(self.subdivision_steps):
                sem_seg_logits = F.interpolate(sem_seg_logits, scale_factor=2, mode='bilinear', align_corners=False)
                uncertainty_map = calculate_uncertainty(sem_seg_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(uncertainty_map, self.subdivision_num_points)
                fine_grained_features = cat([point_sample(features[in_feature], point_coords, align_corners=False) for in_feature in self.in_features])
                coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
                point_logits = self.point_head(fine_grained_features, coarse_features)
                N, C, H, W = sem_seg_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                sem_seg_logits = sem_seg_logits.reshape(N, C, H * W).scatter_(2, point_indices, point_logits).view(N, C, H, W)
            return sem_seg_logits, {}


class AlphaLossNV2(torch.nn.Module):
    """
    Implement Neural Volumes alpha loss 2
    """

    def __init__(self, lambda_alpha, clamp_alpha, init_epoch, force_opaque=False):
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.clamp_alpha = clamp_alpha
        self.init_epoch = init_epoch
        self.force_opaque = force_opaque
        if force_opaque:
            self.bceloss = torch.nn.BCELoss()
        self.register_buffer('epoch', torch.tensor(0, dtype=torch.long), persistent=True)

    def sched_step(self, num=1):
        self.epoch += num

    def forward(self, alpha_fine):
        if self.lambda_alpha > 0.0 and self.epoch.item() >= self.init_epoch:
            alpha_fine = torch.clamp(alpha_fine, 0.01, 0.99)
            if self.force_opaque:
                alpha_loss = self.lambda_alpha * self.bceloss(alpha_fine, torch.ones_like(alpha_fine))
            else:
                alpha_loss = torch.log(alpha_fine) + torch.log(1.0 - alpha_fine)
                alpha_loss = torch.clamp_min(alpha_loss, -self.clamp_alpha)
                alpha_loss = self.lambda_alpha * alpha_loss.mean()
        else:
            alpha_loss = torch.zeros(1, device=alpha_fine.device)
        return alpha_loss


class RGBWithUncertainty(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = torch.nn.L1Loss(reduction='none') if conf.get_bool('use_l1') else torch.nn.MSELoss(reduction='none')

    def forward(self, outputs, targets, betas):
        """computes the error per output, weights each element by the log variance
        outputs is B x 3, targets is B x 3, betas is B"""
        weighted_element_err = torch.mean(self.element_loss(outputs, targets), -1) / betas
        return torch.mean(weighted_element_err) + torch.mean(torch.log(betas))


class RGBWithBackground(torch.nn.Module):
    """Implement the uncertainty loss from Kendall '17"""

    def __init__(self, conf):
        super().__init__()
        self.element_loss = torch.nn.L1Loss(reduction='none') if conf.get_bool('use_l1') else torch.nn.MSELoss(reduction='none')

    def forward(self, outputs, targets, lambda_bg):
        """If we're using background, then the color is color_fg + lambda_bg * color_bg.
        We want to weight the background rays less, while not putting all alpha on bg"""
        weighted_element_err = torch.mean(self.element_loss(outputs, targets), -1) / (1 + lambda_bg)
        return torch.mean(weighted_element_err) + torch.mean(torch.log(lambda_bg))


class ImageEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = resnet34(True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)
        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        feats3 = self.resnet.layer2(feats2)
        feats4 = self.resnet.layer3(feats3)
        latents = [feats1, feats2, feats3, feats4]
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(latents[i], latent_sz, mode='bilinear', align_corners=True)
        latents = torch.cat(latents, dim=1)
        return F.max_pool2d(latents, kernel_size=latents.size()[2:])[:, :, 0, 0]


class Decoder(nn.Module):

    def __init__(self, hidden_size=128, n_blocks=8, n_blocks_view=1, skips=[4], n_freq_posenc=10, n_freq_posenc_views=4, z_dim=128, rgb_out_dim=3):
        super().__init__()
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.z_dim = z_dim
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view
        dim_embed = 3 * self.n_freq_posenc * 2
        dim_embed_view = 3 * self.n_freq_posenc_views * 2
        self.fc_in = nn.Linear(dim_embed, hidden_size)
        if z_dim > 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.blocks = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])
        n_skips = sum([(i in skips) for i in range(n_blocks - 1)])
        if n_skips > 0:
            self.fc_z_skips = nn.ModuleList([nn.Linear(z_dim, hidden_size) for i in range(n_skips)])
            self.fc_p_skips = nn.ModuleList([nn.Linear(dim_embed, hidden_size) for i in range(n_skips)])
        self.sigma_out = nn.Linear(hidden_size, 1)
        self.fc_z_view = nn.Linear(z_dim, hidden_size)
        self.feat_view = nn.Linear(hidden_size, hidden_size)
        self.fc_view = nn.Linear(dim_embed_view, hidden_size)
        self.feat_out = nn.Linear(hidden_size, rgb_out_dim)
        self.blocks_view = nn.ModuleList([nn.Linear(dim_embed_view + hidden_size, hidden_size) for _ in range(n_blocks_view - 1)])
        self.fc_shape = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
        self.fc_app = nn.Sequential(nn.Linear(512, 128), nn.ReLU())

    def transform_points(self, p, views=False):
        L = self.n_freq_posenc_views if views else self.n_freq_posenc
        p_transformed = torch.cat([torch.cat([torch.sin(2 ** i * pi * p), torch.cos(2 ** i * pi * p)], dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, p_in, ray_d, latent=None):
        z_shape = self.fc_shape(latent)
        z_app = self.fc_app(latent)
        B, N, _ = p_in.shape
        z_shape = z_shape[:, None, :].repeat(1, N, 1)
        z_app = z_app[:, None, :].repeat(1, N, 1)
        p = self.transform_points(p_in)
        net = self.fc_in(p)
        if z_shape is not None:
            net = net + self.fc_z(z_shape)
        net = F.relu(net)
        skip_idx = 0
        for idx, layer in enumerate(self.blocks):
            net = F.relu(layer(net))
            if idx + 1 in self.skips and idx < len(self.blocks) - 1:
                net = net + self.fc_z_skips[skip_idx](z_shape)
                net = net + self.fc_p_skips[skip_idx](p)
                skip_idx += 1
        sigma_out = self.sigma_out(net)
        net = self.feat_view(net)
        net = net + self.fc_z_view(z_app)
        ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
        ray_d = self.transform_points(ray_d, views=True)
        net = net + self.fc_view(ray_d)
        net = F.relu(net)
        if self.n_blocks_view > 1:
            for layer in self.blocks_view:
                net = F.relu(layer(net))
        feat_out = self.feat_out(net)
        return feat_out, sigma_out


class PixelNeRFNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = Decoder()

    def encode(self, images):
        return self.encoder(images)

    def forward(self, xyz, viewdirs=None, latent=None):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        rgb, sigma = self.decoder(xyz, viewdirs, latent)
        output_list = [torch.sigmoid(rgb), F.softplus(sigma)]
        output = torch.cat(output_list, dim=-1)
        return output


class NeRFRenderer(torch.nn.Module):

    def __init__(self, n_coarse=64, noise_std=0.0, white_bkgd=False):
        super().__init__()
        self.n_coarse = n_coarse
        self.noise_std = noise_std
        self.white_bkgd = white_bkgd

    def sample_from_ray(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]
        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)
        z_steps += torch.rand_like(z_steps) * step
        return near * (1 - z_steps) + far * z_steps

    def nerf_predict(self, model, rays, z_samp, latent=None):
        B = latent.shape[0]
        NB, K = z_samp.shape
        points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
        viewdirs = rays[:, None, 3:6].expand(-1, K, -1).contiguous()
        split_points = points.view(-1, B, K, 3).permute(1, 0, 2, 3).reshape(B, -1, 3)
        split_viewdirs = viewdirs.view(-1, B, K, 3).permute(1, 0, 2, 3).reshape(B, -1, 3)
        out = model(split_points, viewdirs=split_viewdirs, latent=latent)
        C = out.shape[-1]
        out = out.view(B, -1, K, C).permute(1, 0, 2, 3).reshape(NB, K, C)
        out = out.view(NB, K, -1)
        rgbs = out[..., :3]
        sigmas = out[..., 3]
        if self.training and self.noise_std > 0.0:
            sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std
        return rgbs, sigmas

    def volume_render(self, rgbs, sigmas, z_samp):
        deltas = z_samp[:, 1:] - z_samp[:, :-1]
        delta_inf = torch.full_like(z_samp[..., :1], 0)
        deltas = torch.cat([deltas, delta_inf], -1)
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)
        T = torch.cumprod(alphas_shifted, -1)
        weights = alphas * T[:, :-1]
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)
        depth_final = torch.sum(weights * z_samp, -1)
        if self.white_bkgd:
            pix_alpha = weights.sum(dim=1)
            rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)
        return rgb_final, depth_final, weights

    def forward(self, model, rays, latent=None):
        assert len(rays.shape) == 3
        N, B, _ = rays.shape
        rays = rays.view(-1, 8)
        z_coarse = self.sample_from_ray(rays)
        rgbs, sigmas = self.nerf_predict(model, rays, z_coarse, latent)
        rgb, depth, weights = self.volume_render(rgbs, sigmas, z_coarse)
        outputs = {'rgb': rgb.view(N, B, -1), 'depth': depth.view(N, B), 'weights': weights.view(N, B, -1), 'intersect': (z_coarse[:, 0] != -1).view(N, B)}
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AlphaLossNV2,
     lambda: ([], {'lambda_alpha': 4, 'clamp_alpha': 4, 'init_epoch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ImageEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

