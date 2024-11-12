
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


import time


import numpy as np


import torch.utils.data


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import math


import torch.nn.init as init


from torch.autograd import Variable


import torch.nn


from collections import namedtuple


from torchvision import models as tv


def xaviermultiplier(m, gain):
    """ 
        Args:
            m (torch.nn.Module)
            gain (float)

        Returns:
            std (float): adjusted standard deviation
    """
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels
        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features
        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None
    return std


def xavier_uniform_(m, gain):
    """ Set module weight values with a uniform distribution.

        Args:
            m (torch.nn.Module)
            gain (float)
    """
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-(std * math.sqrt(3.0)), std * math.sqrt(3.0))


def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    """ Initialized module weights.

        Args:
            m (torch.nn.Module)
            gain (float)
            weightinitfunc (function)
    """
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias'):
            m.bias.data.zero_()
    if isinstance(m, nn.ConvTranspose2d):
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
    if isinstance(m, nn.ConvTranspose3d):
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]


def initseq(s):
    """ Initialized weights of all modules in a module sequence.

        Args:
            s (torch.nn.Sequential)
    """
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)
    initmod(s[-1])


class CanonicalMLP(nn.Module):

    def __init__(self, mlp_depth=8, mlp_width=256, input_ch=3, skips=None, **_):
        super(CanonicalMLP, self).__init__()
        if skips is None:
            skips = [4]
        self.mlp_depth = mlp_depth
        self.mlp_width = mlp_width
        self.input_ch = input_ch
        pts_block_mlps = [nn.Linear(input_ch, mlp_width), nn.ReLU()]
        layers_to_cat_input = []
        for i in range(mlp_depth - 1):
            if i in skips:
                layers_to_cat_input.append(len(pts_block_mlps))
                pts_block_mlps += [nn.Linear(mlp_width + input_ch, mlp_width), nn.ReLU()]
            else:
                pts_block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.layers_to_cat_input = layers_to_cat_input
        self.pts_linears = nn.ModuleList(pts_block_mlps)
        initseq(self.pts_linears)
        self.output_linear = nn.Sequential(nn.Linear(mlp_width, 4))
        initseq(self.output_linear)

    def forward(self, pos_embed, **_):
        h = pos_embed
        for i, _ in enumerate(self.pts_linears):
            if i in self.layers_to_cat_input:
                h = torch.cat([pos_embed, h], dim=-1)
            h = self.pts_linears[i](h)
        outputs = self.output_linear(h)
        return outputs


class ConvDecoder3D(nn.Module):
    """ Convolutional 3D volume decoder."""

    def __init__(self, embedding_size=256, volume_size=128, voxel_channels=4):
        """ 
            Args:
                embedding_size: integer
                volume_size: integer
                voxel_channels: integer
        """
        super(ConvDecoder3D, self).__init__()
        self.block_mlp = nn.Sequential(nn.Linear(embedding_size, 1024), nn.LeakyReLU(0.2))
        block_conv = []
        inchannels, outchannels = 1024, 512
        for _ in range(int(np.log2(volume_size)) - 1):
            block_conv.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
            block_conv.append(nn.LeakyReLU(0.2))
            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels
        block_conv.append(nn.ConvTranspose3d(inchannels, voxel_channels, 4, 2, 1))
        self.block_conv = nn.Sequential(*block_conv)
        for m in [self.block_mlp, self.block_conv]:
            initseq(m)

    def forward(self, embedding):
        """ 
            Args:
                embedding: Tensor (B, N)
        """
        return self.block_conv(self.block_mlp(embedding).view(-1, 1024, 1, 1, 1))


class MotionWeightVolumeDecoder(nn.Module):

    def __init__(self, embedding_size=256, volume_size=32, total_bones=24):
        super(MotionWeightVolumeDecoder, self).__init__()
        self.total_bones = total_bones
        self.volume_size = volume_size
        self.const_embedding = nn.Parameter(torch.randn(embedding_size), requires_grad=True)
        self.decoder = ConvDecoder3D(embedding_size=embedding_size, volume_size=volume_size, voxel_channels=total_bones + 1)

    def forward(self, motion_weights_priors, **_):
        embedding = self.const_embedding[None, ...]
        decoded_weights = F.softmax(self.decoder(embedding) + torch.log(motion_weights_priors), dim=1)
        return decoded_weights


SMPL_PARENT = {(1): 0, (2): 0, (3): 0, (4): 1, (5): 2, (6): 3, (7): 4, (8): 5, (9): 6, (10): 7, (11): 8, (12): 9, (13): 9, (14): 9, (15): 12, (16): 13, (17): 14, (18): 16, (19): 17, (20): 18, (21): 19, (22): 20, (23): 21}


class MotionBasisComputer(nn.Module):
    """Compute motion bases between the target pose and canonical pose."""

    def __init__(self, total_bones=24):
        super(MotionBasisComputer, self).__init__()
        self.total_bones = total_bones

    def _construct_G(self, R_mtx, T):
        """ Tile ration matrix and translation vector to build a 4x4 matrix.

        Args:
            R_mtx: Tensor (B, TOTAL_BONES, 3, 3)
            T:     Tensor (B, TOTAL_BONES, 3)

        Returns:
            G:     Tensor (B, TOTAL_BONES, 4, 4)
        """
        batch_size, total_bones = R_mtx.shape[:2]
        assert total_bones == self.total_bones
        G = torch.zeros(size=(batch_size, total_bones, 4, 4), dtype=R_mtx.dtype, device=R_mtx.device)
        G[:, :, :3, :3] = R_mtx
        G[:, :, :3, 3] = T
        G[:, :, 3, 3] = 1.0
        return G

    def forward(self, dst_Rs, dst_Ts, cnl_gtfms):
        """
        Args:
            dst_Rs:    Tensor (B, TOTAL_BONES, 3, 3)
            dst_Ts:    Tensor (B, TOTAL_BONES, 3)
            cnl_gtfms: Tensor (B, TOTAL_BONES, 4, 4)
                
        Returns:
            scale_Rs: Tensor (B, TOTAL_BONES, 3, 3)
            Ts:       Tensor (B, TOTAL_BONES, 3)
        """
        dst_gtfms = torch.zeros_like(cnl_gtfms)
        local_Gs = self._construct_G(dst_Rs, dst_Ts)
        dst_gtfms[:, 0, :, :] = local_Gs[:, 0, :, :]
        for i in range(1, self.total_bones):
            dst_gtfms[:, i, :, :] = torch.matmul(dst_gtfms[:, SMPL_PARENT[i], :, :].clone(), local_Gs[:, i, :, :])
        dst_gtfms = dst_gtfms.view(-1, 4, 4)
        inv_dst_gtfms = torch.inverse(dst_gtfms)
        cnl_gtfms = cnl_gtfms.view(-1, 4, 4)
        f_mtx = torch.matmul(cnl_gtfms, inv_dst_gtfms)
        f_mtx = f_mtx.view(-1, self.total_bones, 4, 4)
        scale_Rs = f_mtx[:, :, :3, :3]
        Ts = f_mtx[:, :, :3, 3]
        return scale_Rs, Ts


def determine_primary_secondary_gpus(cfg):
    None
    cfg.n_gpus = torch.cuda.device_count()
    if cfg.n_gpus > 0:
        all_gpus = list(range(cfg.n_gpus))
        cfg.primary_gpus = [0]
        if cfg.n_gpus > 1:
            cfg.secondary_gpus = [g for g in all_gpus if g not in cfg.primary_gpus]
        else:
            cfg.secondary_gpus = cfg.primary_gpus
    None
    None
    None


def get_cfg_defaults():
    return _C.clone()


def parse_cfg(cfg):
    cfg.logdir = os.path.join('experiments', cfg.category, cfg.task, cfg.subject, cfg.experiment)


def make_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/default.yaml')
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    parse_cfg(cfg)
    determine_primary_secondary_gpus(cfg)
    return cfg


def load_canonical_mlp(module_name):
    module = module_name
    module_path = module.replace('.', '/') + '.py'
    return imp.load_source(module, module_path).CanonicalMLP


def load_mweight_vol_decoder(module_name):
    module = module_name
    module_path = module.replace('.', '/') + '.py'
    return imp.load_source(module, module_path).MotionWeightVolumeDecoder


def load_non_rigid_motion_mlp(module_name):
    module = module_name
    module_path = module.replace('.', '/') + '.py'
    return imp.load_source(module, module_path).NonRigidMotionMLP


def load_pose_decoder(module_name):
    module = module_name
    module_path = module.replace('.', '/') + '.py'
    return imp.load_source(module, module_path).BodyPoseRefiner


def load_positional_embedder(module_name):
    module = module_name
    module_path = module.replace('.', '/') + '.py'
    return imp.load_source(module, module_path).get_embedder


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.motion_basis_computer = MotionBasisComputer(total_bones=cfg.total_bones)
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(embedding_size=cfg.mweight_volume.embedding_size, volume_size=cfg.mweight_volume.volume_size, total_bones=cfg.total_bones)
        self.get_non_rigid_embedder = load_positional_embedder(cfg.non_rigid_embedder.module)
        _, non_rigid_pos_embed_size = self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, cfg.non_rigid_motion_mlp.i_embed)
        self.non_rigid_mlp = load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(pos_embed_size=non_rigid_pos_embed_size, condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size, mlp_width=cfg.non_rigid_motion_mlp.mlp_width, mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth, skips=cfg.non_rigid_motion_mlp.skips)
        self.non_rigid_mlp = nn.DataParallel(self.non_rigid_mlp, device_ids=cfg.secondary_gpus, output_device=cfg.secondary_gpus[0])
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = get_embedder(cfg.canonical_mlp.multires, cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn
        skips = [4]
        self.cnl_mlp = load_canonical_mlp(cfg.canonical_mlp.module)(input_ch=cnl_pos_embed_size, mlp_depth=cfg.canonical_mlp.mlp_depth, mlp_width=cfg.canonical_mlp.mlp_width, skips=skips)
        self.cnl_mlp = nn.DataParallel(self.cnl_mlp, device_ids=cfg.secondary_gpus, output_device=cfg.primary_gpus[0])
        self.pose_decoder = load_pose_decoder(cfg.pose_decoder.module)(embedding_size=cfg.pose_decoder.embedding_size, mlp_width=cfg.pose_decoder.mlp_width, mlp_depth=cfg.pose_decoder.mlp_depth)

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp
        return self

    def _query_mlp(self, pos_xyz, pos_embed_fn, non_rigid_pos_embed_fn, non_rigid_mlp_input):
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = cfg.netchunk_per_gpu * len(cfg.secondary_gpus)
        result = self._apply_mlp_kernals(pos_flat=pos_flat, pos_embed_fn=pos_embed_fn, non_rigid_mlp_input=non_rigid_mlp_input, non_rigid_pos_embed_fn=non_rigid_pos_embed_fn, chunk=chunk)
        output = {}
        raws_flat = result['raws']
        output['raws'] = torch.reshape(raws_flat, list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])
        return output

    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))

    def _apply_mlp_kernals(self, pos_flat, pos_embed_fn, non_rigid_mlp_input, non_rigid_pos_embed_fn, chunk):
        raws = []
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start
            xyz = pos_flat[start:end]
            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(pos_embed=non_rigid_embed_xyz, pos_xyz=xyz, condition_code=self._expand_input(non_rigid_mlp_input, total_elem))
                xyz = result['xyz']
            xyz_embedded = pos_embed_fn(xyz)
            raws += [self.cnl_mlp(pos_embed=xyz_embedded)]
        output = {}
        output['raws'] = torch.cat(raws, dim=0)
        return output

    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i + cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):

        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw) * dists)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        infinity_dists = torch.Tensor([10000000000.0])
        infinity_dists = infinity_dists.expand(dists[..., :1].shape)
        dists = torch.cat([dists, infinity_dists], dim=-1)
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        rgb = torch.sigmoid(raw[..., :3])
        alpha = _raw2alpha(raw[..., 3], dists)
        alpha = alpha * raw_mask[:, :, 0]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.0 - acc_map[..., None]) * bgcolor[None, :] / 255.0
        return rgb_map, acc_map, weights, depth_map

    @staticmethod
    def _sample_motion_fields(pts, motion_scale_Rs, motion_Ts, motion_weights_vol, cnl_bbox_min_xyz, cnl_bbox_scale_xyz, output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3)
        motion_weights = motion_weights_vol[:-1]
        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) * cnl_bbox_scale_xyz[None, :] - 1.0
            weights = F.grid_sample(input=motion_weights[None, i:i + 1, :, :, :], grid=pos[None, None, None, :, :], padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None]
            weights_list.append(weights)
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]
        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i + 1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(torch.stack(weighted_motion_fields, dim=0), dim=0) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum
        x_skel = x_skel.reshape(orig_shape[:2] + [3])
        backwarp_motion_weights = backwarp_motion_weights.reshape(orig_shape[:2] + [total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2] + [1])
        results = {}
        if 'x_skel' in output_list:
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list:
            results['fg_likelihood_mask'] = fg_likelihood_mask
        return results

    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]
        return rays_o, rays_d, near, far

    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0.0, 1.0, steps=cfg.N_samples)
        z_vals = near * (1.0 - t_vals) + far * t_vals
        return z_vals.expand([N_rays, cfg.N_samples])

    @staticmethod
    def _stratified_sampling(z_vals):
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
        return z_vals

    def _render_rays(self, ray_batch, motion_scale_Rs, motion_Ts, motion_weights_vol, cnl_bbox_min_xyz, cnl_bbox_scale_xyz, pos_embed_fn, non_rigid_pos_embed_fn, non_rigid_mlp_input=None, bgcolor=None, **_):
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)
        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.0:
            z_vals = self._stratified_sampling(z_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        mv_output = self._sample_motion_fields(pts=pts, motion_scale_Rs=motion_scale_Rs[0], motion_Ts=motion_Ts[0], motion_weights_vol=motion_weights_vol, cnl_bbox_min_xyz=cnl_bbox_min_xyz, cnl_bbox_scale_xyz=cnl_bbox_scale_xyz, output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']
        query_result = self._query_mlp(pos_xyz=cnl_pts, non_rigid_mlp_input=non_rigid_mlp_input, pos_embed_fn=pos_embed_fn, non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        raw = query_result['raws']
        rgb_map, acc_map, _, depth_map = self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)
        return {'rgb': rgb_map, 'alpha': acc_map, 'depth': depth_map}

    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(dst_Rs, dst_Ts, cnl_gtfms)
        return motion_scale_Rs, motion_Ts

    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    def forward(self, rays, dst_Rs, dst_Ts, cnl_gtfms, motion_weights_priors, dst_posevec=None, near=None, far=None, iter_val=10000000.0, **kwargs):
        dst_Rs = dst_Rs[None, ...]
        dst_Ts = dst_Ts[None, ...]
        dst_posevec = dst_posevec[None, ...]
        cnl_gtfms = cnl_gtfms[None, ...]
        motion_weights_priors = motion_weights_priors[None, ...]
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
            pose_out = self.pose_decoder(dst_posevec)
            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(dst_Rs_no_root, refined_Rs)
            dst_Rs = torch.cat([dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)
            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts
        non_rigid_pos_embed_fn, _ = self.get_non_rigid_embedder(multires=cfg.non_rigid_motion_mlp.multires, is_identity=cfg.non_rigid_motion_mlp.i_embed, iter_val=iter_val)
        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec
        kwargs.update({'pos_embed_fn': self.pos_embed_fn, 'non_rigid_pos_embed_fn': non_rigid_pos_embed_fn, 'non_rigid_mlp_input': non_rigid_mlp_input})
        motion_scale_Rs, motion_Ts = self._get_motion_base(dst_Rs=dst_Rs, dst_Ts=dst_Ts, cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(motion_weights_priors=motion_weights_priors)
        motion_weights_vol = motion_weights_vol[0]
        kwargs.update({'motion_scale_Rs': motion_scale_Rs, 'motion_Ts': motion_Ts, 'motion_weights_vol': motion_weights_vol})
        rays_o, rays_d = rays
        rays_shape = rays_d.shape
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)
        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)
        return all_ret


class NonRigidMotionMLP(nn.Module):

    def __init__(self, pos_embed_size=3, condition_code_size=69, mlp_width=128, mlp_depth=6, skips=None):
        super(NonRigidMotionMLP, self).__init__()
        self.skips = [4] if skips is None else skips
        block_mlps = [nn.Linear(pos_embed_size + condition_code_size, mlp_width), nn.ReLU()]
        layers_to_cat_inputs = []
        for i in range(1, mlp_depth):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(mlp_width + pos_embed_size, mlp_width), nn.ReLU()]
            else:
                block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        block_mlps += [nn.Linear(mlp_width, 3)]
        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)
        self.layers_to_cat_inputs = layers_to_cat_inputs
        init_val = 1e-05
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    def forward(self, pos_embed, pos_xyz, condition_code, viewdirs=None, **_):
        h = torch.cat([condition_code, pos_embed], dim=-1)
        if viewdirs is not None:
            h = torch.cat([h, viewdirs], dim=-1)
        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)
        trans = h
        result = {'xyz': pos_xyz + trans, 'offsets': trans}
        return result


class RodriguesModule(nn.Module):

    def forward(self, rvec):
        """ Apply Rodriguez formula on a batch of rotation vectors.

            Args:
                rvec: Tensor (B, 3)
            
            Returns
                rmtx: Tensor (B, 3, 3)
        """
        theta = torch.sqrt(1e-05 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((rvec[:, 0] ** 2 + (1.0 - rvec[:, 0] ** 2) * costh, rvec[:, 0] * rvec[:, 1] * (1.0 - costh) - rvec[:, 2] * sinth, rvec[:, 0] * rvec[:, 2] * (1.0 - costh) + rvec[:, 1] * sinth, rvec[:, 0] * rvec[:, 1] * (1.0 - costh) + rvec[:, 2] * sinth, rvec[:, 1] ** 2 + (1.0 - rvec[:, 1] ** 2) * costh, rvec[:, 1] * rvec[:, 2] * (1.0 - costh) - rvec[:, 0] * sinth, rvec[:, 0] * rvec[:, 2] * (1.0 - costh) - rvec[:, 1] * sinth, rvec[:, 1] * rvec[:, 2] * (1.0 - costh) + rvec[:, 0] * sinth, rvec[:, 2] ** 2 + (1.0 - rvec[:, 2] ** 2) * costh), dim=1).view(-1, 3, 3)


class BodyPoseRefiner(nn.Module):

    def __init__(self, embedding_size=69, mlp_width=256, mlp_depth=4, **_):
        super(BodyPoseRefiner, self).__init__()
        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
        for _ in range(0, mlp_depth - 1):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        self.total_bones = cfg.total_bones - 1
        block_mlps += [nn.Linear(mlp_width, 3 * self.total_bones)]
        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)
        init_val = 1e-05
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()
        self.rodriguez = RodriguesModule()

    def forward(self, pose_input):
        rvec = self.block_mlps(pose_input).view(-1, 3)
        Rs = self.rodriguez(rvec).view(-1, self.total_bones, 3, 3)
        return {'Rs': Rs}


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ScalingLayer(nn.Module):

    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-0.03, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.45])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class LPIPS(nn.Module):

    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, pnet_rand=False, pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        super(LPIPS, self).__init__()
        if verbose:
            None
        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = pn.vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == 'alex':
            net_type = pn.alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == 'squeeze':
            net_type = pn.squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)
        self.net = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == 'squeeze':
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.ModuleList(self.lins)
            if pretrained:
                if model_path is None:
                    import inspect
                    model_path = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))
                if verbose:
                    None
                self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        if eval_mode:
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        if normalize:
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(outs0[kk]), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = res[0]
        for l in range(1, self.L):
            val += res[l]
        if retPerLayer:
            return val, res
        else:
            return val


class Dist2LogitLayer(nn.Module):
    """ takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) """

    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()
        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True)]
        layers += [nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True)]
        if use_sigmoid:
            layers += [nn.Sigmoid()]
        self.model = nn.Sequential(*layers)

    def forward(self, d0, d1, eps=0.1):
        return self.model.forward(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):

    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge + 1.0) / 2.0
        self.logit = self.net.forward(d0, d1)
        return self.loss(self.logit, per)


class FakeNet(nn.Module):

    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace = colorspace


class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            N, C, X, Y = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0 - in1) ** 2, dim=1).view(N, 1, X, Y), dim=2).view(N, 1, 1, Y), dim=3).view(N)
            return value
        elif self.colorspace == 'Lab':
            value = lpips.l2(lpips.tensor2np(lpips.tensor2tensorlab(in0.data, to_norm=False)), lpips.tensor2np(lpips.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = lpips.dssim(1.0 * lpips.tensor2im(in0.data), 1.0 * lpips.tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = lpips.dssim(lpips.tensor2np(lpips.tensor2tensorlab(in0.data, to_norm=False)), lpips.tensor2np(lpips.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var
        return ret_var


class squeezenet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = tv.squeezenet1_1(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple('SqueezeOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        return out


class alexnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple('AlexnetOutputs', ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class vgg16(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class resnet(torch.nn.Module):

    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = tv.resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = tv.resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = tv.resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = tv.resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = tv.resnet152(pretrained=pretrained)
        self.N_slices = 5
        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def forward(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h
        outputs = namedtuple('Outputs', ['relu1', 'conv2', 'conv3', 'conv4', 'conv5'])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Dist2LogitLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {})),
    (NetLinLayer,
     lambda: ([], {'chn_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RodriguesModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ScalingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {})),
    (alexnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (resnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (squeezenet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (vgg16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

