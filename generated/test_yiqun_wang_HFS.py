import sys
_module = sys.modules[__name__]
del sys
exp_runner_high = _module
dataset = _module
embedder_high = _module
fields_high = _module
renderer_high = _module
pytorch_ssim = _module

from paritybench._paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import time


import logging


import numpy as np


import torch


import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter


import matplotlib


from scipy.spatial.transform import Rotation as Rot


from scipy.spatial.transform import Slerp


import torch.nn as nn


from torch.autograd import Variable


from math import exp


def coarse2fine(progress_data, inputs, L):
    barf_c2f = [0.1, 0.5]
    if barf_c2f is not None:
        start, end = barf_c2f
        alpha = (progress_data - start) / (end - start) * L
        k = torch.arange(L, dtype=torch.float32, device=inputs.device)
        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        shape = inputs.shape
        input_enc = (inputs.view(-1, L, int(shape[1] / L)) * weight.tile(int(shape[1] / L), 1).T).view(*shape)
    return input_enc, weight


class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
        out_dim += d
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {'include_input': False, 'input_dims': input_dims, 'max_freq_log2': multires - 1, 'num_freqs': multires, 'log_sampling': True, 'periodic_fns': [torch.sin, torch.cos]}
    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)
    return embed, embedder_obj.out_dim


class SDFNetwork(nn.Module):

    def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=(4,), multires=0, bias=0.5, scale=1, geometric_init=True, weight_norm=True, inside_outside=False):
        super(SDFNetwork, self).__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn_fine = None
        self.multires = multires
        self.progress = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            self.num_eoc = int((input_ch - d_in) / 2)
            dims[0] = d_in + self.num_eoc
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, 'lin' + str(l), lin)
        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            input_enc = self.embed_fn_fine(inputs)
            nfea_eachband = int(input_enc.shape[1] / self.multires)
            N = int(self.multires / 2)
            inputs_enc, weight = coarse2fine(0.5 * (self.progress.data - 0.1), input_enc, self.multires)
            inputs_enc = inputs_enc.view(-1, self.multires, nfea_eachband)[:, :N, :].view([-1, self.num_eoc])
            input_enc = input_enc.view(-1, self.multires, nfea_eachband)[:, :N, :].view([-1, self.num_eoc]).contiguous()
            input_enc = (input_enc.view(-1, N) * weight[:N]).view([-1, self.num_eoc])
            flag = weight[:N].tile(input_enc.shape[0], nfea_eachband, 1).transpose(1, 2).contiguous().view([-1, self.num_eoc])
            inputs_enc = torch.where(flag > 0.01, inputs_enc, input_enc)
            inputs = torch.cat([inputs, inputs_enc], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gradients.unsqueeze(1)


class SDFNetworkHigh(nn.Module):

    def __init__(self, d_in, d_out, d_hidden, n_layers, skip_in=(4,), multires=0, bias=0.5, scale=1, geometric_init=True, weight_norm=True, inside_outside=False):
        super(SDFNetworkHigh, self).__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.embed_fn_fine = None
        self.multires = multires
        self.progress = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            self.num_eoc = int((input_ch - d_in) / 2)
            dims[0] = d_in + self.num_eoc
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, 'lin' + str(l), lin)
        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs, sdf_network, gradient, s):
        inputs = inputs * self.scale
        pts = inputs
        if self.embed_fn_fine is not None:
            input_enc = self.embed_fn_fine(inputs)
            nfea_eachband = int(input_enc.shape[1] / self.multires)
            N = int(self.multires / 2)
            inputs_enc, weight = coarse2fine(0.2 + 0.5 * (self.progress.data - 0.1), input_enc, self.multires)
            inputs_enc = inputs_enc.view(-1, self.multires, nfea_eachband)[:, N:, :].view([-1, self.num_eoc])
            input_enc = input_enc.view(-1, self.multires, nfea_eachband)[:, N:, :].view([-1, self.num_eoc]).contiguous()
            input_enc = (input_enc.view(-1, N) * weight[N:]).view([-1, self.num_eoc])
            flag = weight[N:].tile(input_enc.shape[0], nfea_eachband, 1).transpose(1, 2).contiguous().view([-1, self.num_eoc])
            inputs_enc = torch.where(flag > 0.01, inputs_enc, input_enc)
            inputs = torch.cat([inputs, inputs_enc], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        sigmoid_sdf = torch.sigmoid(4 * 0.01 * s * sdf_network.sdf(pts.reshape(-1, 3)))
        weaken = 4 * torch.tanh(4 * 0.01 * s) * sigmoid_sdf * (1 - sigmoid_sdf)
        sdf_output = sdf_network(pts.reshape(-1, 3) - weaken * x[:, :1] * (gradient / torch.norm(gradient, dim=1, keepdim=True)))
        return torch.cat([sdf_output[:, :1] / self.scale, sdf_output[:, 1:] + x[:, 1:]], dim=-1)

    def sdf(self, x, sdf_network, gradient, s):
        return self.forward(x, sdf_network, gradient, s)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x, sdf_network, gradient, s):
        x.requires_grad_(True)
        y = self.sdf(x, sdf_network, gradient, s)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):

    def __init__(self, d_feature, mode, d_in, d_out, d_hidden, n_layers, weight_norm=True, multires_view=0, squeeze_out=True):
        super().__init__()
        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]
        self.multires_view = multires_view
        self.progress = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, 'lin' + str(l), lin)
        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs_enc = self.embedview_fn(view_dirs)
            view_dirs_enc, weight = coarse2fine(0.5 * self.progress.data, view_dirs_enc, self.multires_view)
            view_dirs = torch.cat([view_dirs, view_dirs_enc], dim=-1)
        rendering_input = None
        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class NeRF(nn.Module):

    def __init__(self, D=8, W=256, d_in=3, d_in_view=3, multires=0, multires_view=0, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None
        self.multires = multires
        self.multires_view = multires_view
        self.progress = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch
        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList([nn.Linear(self.input_ch, W)] + [(nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)) for i in range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts_enc = self.embed_fn(input_pts)
            input_pts_enc, weight = coarse2fine(self.progress.data, input_pts_enc, self.multires)
            input_pts = torch.cat([input_pts, input_pts_enc], dim=-1)
        if self.embed_fn_view is not None:
            input_views_enc = self.embed_fn_view(input_views)
            input_views_enc, weight = coarse2fine(self.progress.data, input_views_enc, self.multires_view)
            input_views = torch.cat([input_views, input_views_enc], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


def _ssim(img1, img2, window, window_size, channel, use_padding, size_average=True):
    if use_padding:
        padding_size = window_size // 2
    else:
        padding_size = 0
    mu1 = F.conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding_size, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):

    def __init__(self, window_size=11, use_padding=True, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.use_padding = use_padding
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.use_padding, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from paritybench._paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SDFNetwork,
     lambda: ([], {'d_in': 4, 'd_out': 4, 'd_hidden': 4, 'n_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_yiqun_wang_HFS(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

