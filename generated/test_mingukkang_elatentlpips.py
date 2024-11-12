
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


import torch


import scipy.signal


import re


import uuid


import torch.utils.cpp_extension


from torch.utils.file_baton import FileBaton


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import warnings


from collections import namedtuple


from collections import OrderedDict


from torchvision import models as tv


from copy import deepcopy


from torchvision import transforms


import torch.optim as optim


import matplotlib.pyplot as plt


def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return misc.constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [(x if isinstance(x, torch.Tensor) else misc.constant(x, shape=ref[0].shape, device=ref[0].device)) for x in elems]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def rotate2d(theta, **kwargs):
    return matrix([torch.cos(theta), torch.sin(-theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1], **kwargs)


def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)


def scale2d(sx, sy, **kwargs):
    return matrix([sx, 0, 0], [0, sy, 0], [0, 0, 1], **kwargs)


def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)


def translate2d(tx, ty, **kwargs):
    return matrix([1, 0, tx], [0, 1, ty], [0, 0, 1], **kwargs)


def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)


def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(module_name='upfirdn2d_plugin', sources=['upfirdn2d.cpp', 'upfirdn2d.cu'], headers=['upfirdn2d.h'], source_dir=os.path.dirname(__file__), extra_cuda_cflags=['--use_fast_math'])
    return True


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
    misc.assert_shape(f, [fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


_plugin = None


_upfirdn2d_cuda_cache = dict()


def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.
    """
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    key = upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]


    class Upfirdn2dCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, f):
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0)
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn2d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = _plugin.upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy):
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [fw - padx0 - 1, iw * upx - ow * downx + padx0 - upx + 1, fh - pady0 - 1, ih * upy - oh * downy + pady0 - upy + 1]
            dx = None
            df = None
            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up, padding=p, flip_filter=not flip_filter, gain=gain).apply(dy, f)
            assert not ctx.needs_input_grad[1]
            return dx, df
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda


def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= f.shape[-1] and upH >= f.shape[0]
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0):x.shape[2] - max(-pady1, 0), max(-padx0, 0):x.shape[3] - max(-padx1, 0)]
    f = f * gain ** (f.ndim / 2)
    f = f
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d_gradfix.conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)
    x = x[:, :, ::downy, ::downx]
    return x


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda'):
    """Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain)


wavelets = {'haar': [0.7071067811865476, 0.7071067811865476], 'db1': [0.7071067811865476, 0.7071067811865476], 'db2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025], 'db3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569], 'db4': [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523], 'db5': [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125], 'db6': [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017], 'db7': [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236], 'db8': [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161], 'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025], 'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569], 'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427], 'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728], 'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148], 'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255], 'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609]}


class AdaAugment(torch.nn.Module):

    def __init__(self, xflip=0, rotate90=0, xint=0, xint_max=0.125, scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125, brightness=0, contrast=0, saturation=0, noise=0, cutout=0, noise_std=0.1, cutout_size=0.5):
        super().__init__()
        self.register_buffer('p', torch.ones([]))
        self.xflip = float(xflip)
        self.rotate90 = float(rotate90)
        self.xint = float(xint)
        self.xint_max = float(xint_max)
        self.scale = float(scale)
        self.rotate = float(rotate)
        self.aniso = float(aniso)
        self.xfrac = float(xfrac)
        self.scale_std = float(scale_std)
        self.rotate_max = float(rotate_max)
        self.aniso_std = float(aniso_std)
        self.xfrac_std = float(xfrac_std)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.noise = float(noise)
        self.cutout = float(cutout)
        self.noise_std = float(noise_std)
        self.cutout_size = float(cutout_size)
        self.register_buffer('Hz_geom', upfirdn2d.setup_filter(wavelets['sym6']))

    def forward(self, sources, targets, debug_percentile=None):
        assert isinstance(sources, torch.Tensor) and sources.ndim == 4
        assert isinstance(targets, torch.Tensor) and targets.ndim == 4
        assert sources.shape == targets.shape
        batch_size, num_channels, height, width = sources.shape
        device = sources.device
        if debug_percentile is not None:
            debug_percentile = torch.as_tensor(debug_percentile, dtype=torch.float32, device=device)
        I_3 = torch.eye(3, device=device)
        G_inv = I_3
        if self.xflip > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 2)
            i = torch.where(torch.rand([batch_size], device=device) < self.xflip * self.p, i, torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 2))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)
        if self.rotate90 > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 4)
            i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * self.p, i, torch.zeros_like(i))
            if debug_percentile is not None:
                i = torch.full_like(i, torch.floor(debug_percentile * 4))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)
        if self.xint > 0:
            t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
            if debug_percentile is not None:
                t = torch.full_like(t, (debug_percentile * 2 - 1) * self.xint_max)
            G_inv = G_inv @ translate2d_inv(torch.round(t[:, 0] * width), torch.round(t[:, 1] * height))
        if self.scale > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.scale * self.p, s, torch.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.scale_std))
            G_inv = G_inv @ scale2d_inv(s, s)
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1))
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.rotate_max)
            G_inv = G_inv @ rotate2d_inv(-theta)
        if self.aniso > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.aniso * self.p, s, torch.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.aniso_std))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            if debug_percentile is not None:
                theta = torch.zeros_like(theta)
            G_inv = G_inv @ rotate2d_inv(-theta)
        if self.xfrac > 0:
            t = torch.randn([batch_size, 2], device=device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xfrac * self.p, t, torch.zeros_like(t))
            if debug_percentile is not None:
                t = torch.full_like(t, torch.erfinv(debug_percentile * 2 - 1) * self.xfrac_std)
            G_inv = G_inv @ translate2d_inv(t[:, 0] * width, t[:, 1] * height)
        if G_inv is not I_3:
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device)
            cp = G_inv @ cp.t()
            Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1)
            margin = torch.cat([-margin, margin]).max(dim=1).values
            margin = margin + misc.constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(misc.constant([0, 0] * 2, device=device))
            margin = margin.min(misc.constant([width - 1, height - 1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil()
            sources = torch.nn.functional.pad(input=sources, pad=[mx0, mx1, my0, my1], mode='reflect')
            targets = torch.nn.functional.pad(input=targets, pad=[mx0, mx1, my0, my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv
            sources = upfirdn2d.upsample2d(x=sources, f=self.Hz_geom, up=2)
            targets = upfirdn2d.upsample2d(x=targets, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / sources.shape[3], 2 / sources.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:, :2, :], size=shape, align_corners=False)
            sources = grid_sample_gradfix.grid_sample(sources, grid)
            targets = grid_sample_gradfix.grid_sample(targets, grid)
            sources = upfirdn2d.downsample2d(x=sources, f=self.Hz_geom, down=2, padding=-Hz_pad * 2, flip_filter=True)
            targets = upfirdn2d.downsample2d(x=targets, f=self.Hz_geom, down=2, padding=-Hz_pad * 2, flip_filter=True)
        I_4 = torch.eye(4, device=device)
        if self.brightness > 0:
            brightness_random = torch.rand(sources.size(0), 1, 1, 1, dtype=sources.dtype, device=sources.device)
            sources = sources + (brightness_random - 0.5)
            targets = targets + (brightness_random - 0.5)
        if self.contrast > 0:
            source_mean = sources.mean(dim=[1, 2, 3], keepdim=True)
            target_mean = targets.mean(dim=[1, 2, 3], keepdim=True)
            contrast_random = torch.rand(sources.size(0), 1, 1, 1, dtype=sources.dtype, device=sources.device)
            sources = (sources - source_mean) * (contrast_random + 0.5) + source_mean
            targets = (targets - target_mean) * (contrast_random + 0.5) + target_mean
        if self.saturation > 0 and num_channels > 1:
            source_mean = sources.mean(dim=1, keepdim=True)
            target_mean = targets.mean(dim=1, keepdim=True)
            saturation_random = torch.rand(sources.size(0), 1, 1, 1, dtype=sources.dtype, device=sources.device)
            sources = (sources - source_mean) * (saturation_random * 2) + source_mean
            targets = (targets - target_mean) * (saturation_random * 2) + target_mean
        if self.noise > 0:
            sigma = torch.randn([batch_size, 1, 1, 1], device=device).abs() * self.noise_std
            sigma = torch.where(torch.rand([batch_size, 1, 1, 1], device=device) < self.noise * self.p, sigma, torch.zeros_like(sigma))
            if debug_percentile is not None:
                sigma = torch.full_like(sigma, torch.erfinv(debug_percentile) * self.noise_std)
            sources = sources + torch.randn([batch_size, num_channels, height, width], device=device) * sigma
            targets = targets + torch.randn([batch_size, num_channels, height, width], device=device) * sigma
        if self.cutout > 0:
            size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=device)
            size = torch.where(torch.rand([batch_size, 1, 1, 1, 1], device=device) < self.cutout * self.p, size, torch.zeros_like(size))
            center = torch.rand([batch_size, 2, 1, 1, 1], device=device)
            if debug_percentile is not None:
                size = torch.full_like(size, self.cutout_size)
                center = torch.full_like(center, debug_percentile)
            coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
            mask_x = ((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2
            mask_y = ((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2
            mask = torch.logical_or(mask_x, mask_y)
            sources = sources * mask
            targets = targets * mask
        return sources, targets


class LatentVGG16BN(torch.nn.Module):

    def __init__(self, num_latent_channels, requires_grad=True, pretrained=False):
        super(LatentVGG16BN, self).__init__()
        self.model = tv.vgg16_bn(pretrained=pretrained)
        self.model.features[0] = torch.nn.Conv2d(num_latent_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        delete_idx = [i for i, layer in enumerate(self.model.features) if isinstance(layer, torch.nn.MaxPool2d)][:3]
        for idx in delete_idx:
            self.model.features[idx] = torch.nn.Identity()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = self.model(X)
        return out


class CalibratedLatentVGG16BN(torch.nn.Module):

    def __init__(self, num_latent_channels, encoder, pretrained=False, requires_grad=True):
        super(CalibratedLatentVGG16BN, self).__init__()
        model = LatentVGG16BN(num_latent_channels, requires_grad=True)
        if pretrained:
            url_path = f'https://huggingface.co/Mingguksky/elatentlpips/resolve/main/elatentlpips_ckpt/{encoder}_latent_vgg16.pth.tar'
            if not os.path.exists('./ckpt'):
                os.makedirs('./ckpt', exist_ok=True)
            if not os.path.exists(f'./ckpt/{encoder}_latent_vgg16.pth.tar'):
                torch.hub.download_url_to_file(url_path, f'./ckpt/{encoder}_latent_vgg16.pth.tar')
            ckpt = torch.load(f'./ckpt/{encoder}_latent_vgg16.pth.tar', map_location=lambda storage, loc: storage)
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            None
            vgg_pretrained_features = deepcopy(model.model.features)
            del model
            del ckpt
        else:
            vgg_pretrained_features = tv.vgg16_bn(pretrained=False).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(7):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 14):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 24):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(24, 34):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(34, 44):
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


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


ada_augpipe = {'b': dict(xflip=1, rotate90=1, xint=1), 'g': dict(scale=1, rotate=1, aniso=1, xfrac=1), 'c': dict(brightness=1, contrast=1, saturation=1), 'o': dict(cutout=1), 'co': dict(brightness=1, contrast=1, saturation=1, cutout=1), 'bg': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1), 'bgc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, saturation=1), 'bgco': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, saturation=1, cutout=1)}


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_HW=(64, 64)):
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class ELatentLPIPS(nn.Module):

    def __init__(self, pretrained=True, net='vgg16', elatentlpips=True, spatial=False, pnet_rand=False, pnet_tune=False, use_dropout=True, eval_mode=True, verbose=True, encoder='sd3', augment='bg'):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters
        ---------------------------------
        pretrained : bool
            This flag controls the linear layers, which are only in effect when elatentlpips=True above
            [True] means linear layers are calibrated with human perceptual judgments
            [False] means linear layers are randomly initialized
        net : str
            ['vgg16'] is the base/trunk networks available
        elatentlpips : bool
            This flag activates ensembling of latent perceptual loss computation.
        pnet_rand : bool
            [False] means trunk loaded with ImageNet classification weights
            [True] means randomly initialized trunk
        encoder : str
            Specifies the type of latent space generated by the encoder.
            Available options: ['sd15', 'sd21', 'sdxl', 'sd3', 'flux']. Default is ['sd3'].
        augment : str
            Types of differentiable augmentations applied to the input.
            Available options: ['b', 'g', 'c', 'o', 'co', 'bg', 'bgc', 'bgco']. Default is ['bg'].

        The following parameters should only be changed if training the network

        pnet_tune
            [False] keep base/trunk frozen
            [True] tune the base/trunk network
        use_dropout : bool
            [True] to use dropout when training linear layers
            [False] for no dropout when training linear layers
        eval_mode : bool
            [True] is for test mode (default)
            [False] is for training mode
        """
        super(ELatentLPIPS, self).__init__()
        if verbose:
            None
        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.elatentlpips = elatentlpips
        self.encoder = encoder
        self.augment = augment
        if self.pnet_type in ['vgg', 'vgg16']:
            net_type = CalibratedLatentVGG16BN
            self.chns = [64, 128, 256, 512, 512]
        else:
            raise NotImplementedError('Network %s not implemented' % net)
        self.L = len(self.chns)
        if self.encoder == 'sd15':
            num_latent_channels = 4
        elif self.encoder == 'sd21':
            num_latent_channels = 4
        elif self.encoder == 'sdxl':
            num_latent_channels = 4
        elif self.encoder == 'sd3':
            num_latent_channels = 16
        elif self.encoder == 'flux':
            num_latent_channels = 16
        self.net = net_type(num_latent_channels, self.encoder, pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)
        if elatentlpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            self.lins = nn.ModuleList(self.lins)
            if pretrained:
                if self.encoder in ['sd15', 'sd21', 'sdxl', 'sd3', 'flux']:
                    url_path = f'https://huggingface.co/Mingguksky/elatentlpips/resolve/main/elatentlpips_ckpt/{self.encoder}_latest_{self.pnet_type}_tuned.pth'
                    if not os.path.exists('./ckpt'):
                        os.makedirs('./ckpt', exist_ok=True)
                    if not os.path.exists(f'./ckpt/{self.encoder}_latest_{self.pnet_type}_tuned.pth'):
                        torch.hub.download_url_to_file(url_path, f'./ckpt/{self.encoder}_latest_{self.pnet_type}_tuned.pth')
                    ckpt = torch.load(f'./ckpt/{self.encoder}_latest_{self.pnet_type}_tuned.pth', map_location=lambda storage, loc: storage)
                else:
                    raise NotImplementedError('Encoder %s not implemented' % self.encoder)
                if verbose:
                    None
                self.load_state_dict(ckpt, strict=True)
        if augment is not None:
            self.augment = AdaAugment(**ada_augpipe[augment]).train().requires_grad_(False)
            self.augment.p = torch.tensor(1.0)
        if eval_mode:
            self.eval()

    def forward(self, in0, in1, retPerLayer=False, normalize=False, ensembling=True, add_l1_loss=True):
        if normalize:
            if self.encoder in ['sd15', 'sd21']:
                in0 = in0 * 0.18215
                in1 = in1 * 0.18215
            elif self.encoder == 'sdxl':
                in0 = in0 * 0.13025
                in1 = in1 * 0.13025
            elif self.encoder == 'sd3':
                in0 = (in0 - 0.0609) * 1.5305
                in1 = (in1 - 0.0609) * 1.5305
            elif self.encoder == 'flux':
                in0 = (in0 - 0.1159) * 0.3611
                in1 = (in1 - 0.1159) * 0.3611
        if add_l1_loss:
            l1_loss = F.l1_loss(in0, in1, reduction='none').mean(dim=(1, 2, 3))[:, None, None, None]
        if ensembling and self.augment is not None:
            in0, in1 = self.augment(in0, in1)
        elif ensembling and self.augment is None:
            raise ValueError('Augmentation is not enabled.')
        outs0, outs1 = self.net.forward(in0), self.net.forward(in1)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = elatentlpips.normalize_tensor(outs0[kk]), elatentlpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        if self.elatentlpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        elif self.spatial:
            res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]
        val = 0
        for l in range(self.L):
            val += res[l]
        if add_l1_loss:
            val += l1_loss
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
        self.loss = nn.BCELoss()

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
            value = elatentlpips.l2(elatentlpips.tensor2np(elatentlpips.tensor2tensorlab(in0.data, to_norm=False)), elatentlpips.tensor2np(elatentlpips.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
            ret_var = Variable(torch.Tensor((value,)))
            if self.use_gpu:
                ret_var = ret_var
            return ret_var


class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert in0.size()[0] == 1
        if self.colorspace == 'RGB':
            value = elatentlpips.dssim(1.0 * elatentlpips.tensor2im(in0.data), 1.0 * elatentlpips.tensor2im(in1.data), range=255.0).astype('float')
        elif self.colorspace == 'Lab':
            value = elatentlpips.dssim(elatentlpips.tensor2np(elatentlpips.tensor2tensorlab(in0.data, to_norm=False)), elatentlpips.tensor2np(elatentlpips.tensor2tensorlab(in1.data, to_norm=False)), range=100.0).astype('float')
        ret_var = Variable(torch.Tensor((value,)))
        if self.use_gpu:
            ret_var = ret_var
        return ret_var


class VGG16BN(torch.nn.Module):

    def __init__(self, requires_grad=True, pretrained=False):
        super(VGG16BN, self).__init__()
        self.model = tv.vgg16_bn(pretrained=pretrained)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = self.model(X)
        return out


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (CalibratedLatentVGG16BN,
     lambda: ([], {'num_latent_channels': 4, 'encoder': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (Dist2LogitLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 1, 4, 4])], {})),
    (LatentVGG16BN,
     lambda: ([], {'num_latent_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NetLinLayer,
     lambda: ([], {'chn_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VGG16BN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
]

