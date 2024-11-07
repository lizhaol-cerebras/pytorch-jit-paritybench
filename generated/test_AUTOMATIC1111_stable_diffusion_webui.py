import sys
_module = sys.modules[__name__]
del sys
ldsr_model_arch = _module
preload = _module
ldsr_model = _module
sd_hijack_autoencoder = _module
sd_hijack_ddpm_v1 = _module
vqvae_quantize = _module
extra_networks_lora = _module
lora = _module
lora_logger = _module
lora_patches = _module
lyco_helpers = _module
network = _module
network_full = _module
network_glora = _module
network_hada = _module
network_ia3 = _module
network_lokr = _module
network_lora = _module
network_norm = _module
network_oft = _module
networks = _module
lora_script = _module
ui_edit_user_metadata = _module
ui_extra_networks_lora = _module
scunet_model = _module
swinir_model = _module
hotkey_config = _module
extra_options_section = _module
hypertile = _module
hypertile_script = _module
postprocessing_autosized_crop = _module
postprocessing_caption = _module
postprocessing_create_flipped_copies = _module
postprocessing_focal_crop = _module
postprocessing_split_oversized = _module
soft_inpainting = _module
launch = _module
api = _module
models = _module
cache = _module
call_queue = _module
cmd_args = _module
codeformer_model = _module
config_states = _module
dat_model = _module
deepbooru = _module
deepbooru_model = _module
devices = _module
errors = _module
esrgan_model = _module
extensions = _module
extra_networks = _module
extra_networks_hypernet = _module
extras = _module
face_restoration = _module
face_restoration_utils = _module
fifo_lock = _module
gfpgan_model = _module
gitpython_hack = _module
gradio_extensons = _module
hashes = _module
hat_model = _module
hypernetwork = _module
ui = _module
images = _module
img2img = _module
import_hook = _module
infotext_utils = _module
infotext_versions = _module
initialize = _module
initialize_util = _module
interrogate = _module
launch_utils = _module
localization = _module
logging_config = _module
lowvram = _module
mac_specific = _module
masking = _module
memmon = _module
modelloader = _module
ddpm_edit = _module
uni_pc = _module
sampler = _module
uni_pc = _module
mmdit = _module
other_impls = _module
sd3_cond = _module
sd3_impls = _module
sd3_model = _module
ngrok = _module
npu_specific = _module
options = _module
patches = _module
paths = _module
paths_internal = _module
postprocessing = _module
processing = _module
comments = _module
refiner = _module
seed = _module
profiling = _module
progress = _module
prompt_parser = _module
realesrgan_model = _module
restart = _module
rng = _module
rng_philox = _module
safe = _module
script_callbacks = _module
script_loading = _module
scripts = _module
scripts_auto_postprocessing = _module
scripts_postprocessing = _module
sd_disable_initialization = _module
sd_emphasis = _module
sd_hijack = _module
sd_hijack_checkpoint = _module
sd_hijack_clip = _module
sd_hijack_clip_old = _module
sd_hijack_ip2p = _module
sd_hijack_open_clip = _module
sd_hijack_optimizations = _module
sd_hijack_unet = _module
sd_hijack_utils = _module
sd_hijack_xlmr = _module
sd_models = _module
sd_models_config = _module
sd_models_types = _module
sd_models_xl = _module
sd_samplers = _module
sd_samplers_cfg_denoiser = _module
sd_samplers_common = _module
sd_samplers_compvis = _module
sd_samplers_extra = _module
sd_samplers_kdiffusion = _module
sd_samplers_lcm = _module
sd_samplers_timesteps = _module
sd_samplers_timesteps_impl = _module
sd_schedulers = _module
sd_unet = _module
sd_vae = _module
sd_vae_approx = _module
sd_vae_taesd = _module
shared = _module
shared_cmd_options = _module
shared_gradio_themes = _module
shared_init = _module
shared_items = _module
shared_options = _module
shared_state = _module
shared_total_tqdm = _module
styles = _module
sub_quadratic_attention = _module
sysinfo = _module
autocrop = _module
dataset = _module
image_embedding = _module
learn_schedule = _module
saving_settings = _module
textual_inversion = _module
timer = _module
torch_utils = _module
txt2img = _module
ui = _module
ui_checkpoint_merger = _module
ui_common = _module
ui_components = _module
ui_extensions = _module
ui_extra_networks = _module
ui_extra_networks_checkpoints = _module
ui_extra_networks_checkpoints_user_metadata = _module
ui_extra_networks_hypernets = _module
ui_extra_networks_textual_inversion = _module
ui_extra_networks_user_metadata = _module
ui_gradio_extensions = _module
ui_loadsave = _module
ui_postprocessing = _module
ui_prompt_styles = _module
ui_settings = _module
ui_tempdir = _module
ui_toprow = _module
upscaler = _module
upscaler_utils = _module
util = _module
xlmr = _module
xlmr_m18 = _module
xpu_specific = _module
custom_code = _module
img2imgalt = _module
loopback = _module
outpainting_mk_2 = _module
poor_mans_outpainting = _module
postprocessing_codeformer = _module
postprocessing_gfpgan = _module
postprocessing_upscale = _module
prompt_matrix = _module
prompts_from_file = _module
sd_upscale = _module
xyz_grid = _module
test = _module
conftest = _module
test_extras = _module
test_face_restorers = _module
test_img2img = _module
test_torch_utils = _module
test_txt2img = _module
test_utils = _module
webui = _module

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


import time


import numpy as np


import torch


import torchvision


import torch.nn.functional as F


from torch.optim.lr_scheduler import LambdaLR


import torch.nn as nn


from functools import partial


from torchvision.utils import make_grid


from collections import namedtuple


import enum


import logging


import re


from typing import Union


from typing import Callable


from functools import wraps


from functools import cache


import math


import random


from typing import Any


from functools import lru_cache


from functools import cached_property


from typing import TYPE_CHECKING


import inspect


from torch import einsum


from torch.nn.init import normal_


from torch.nn.init import xavier_normal_


from torch.nn.init import xavier_uniform_


from torch.nn.init import kaiming_normal_


from torch.nn.init import kaiming_uniform_


from torch.nn.init import zeros_


from collections import deque


import warnings


import torch.hub


from torchvision import transforms


from torchvision.transforms.functional import InterpolationMode


from torch import Tensor


from collections import defaultdict


from typing import Dict


from typing import Optional


from torch import nn


import typing


import collections


import numpy


from torch.nn.functional import silu


from types import MethodType


from torch.utils.checkpoint import checkpoint


from scipy import stats


import torch.nn


from typing import NamedTuple


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import Sampler


from random import shuffle


from random import choices


from functools import reduce


import types


class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index='random', sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer('used', torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == 'extra':
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            None
        else:
            self.re_embed = n_e
        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == 'random':
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, 'Only for interface compatible with Gumbel'
        assert rescale_logits is False, 'Only for interface compatible with Gumbel'
        assert return_logits is False, 'Only for interface compatible with Gumbel'
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class DeepDanbooruModel(nn.Module):

    def __init__(self):
        super(DeepDanbooruModel, self).__init__()
        self.tags = []
        self.n_Conv_0 = nn.Conv2d(kernel_size=(7, 7), in_channels=3, out_channels=64, stride=(2, 2))
        self.n_MaxPool_0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.n_Conv_1 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_2 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=64)
        self.n_Conv_3 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_4 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_5 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=64)
        self.n_Conv_6 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_7 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_8 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=64)
        self.n_Conv_9 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=64)
        self.n_Conv_10 = nn.Conv2d(kernel_size=(1, 1), in_channels=64, out_channels=256)
        self.n_Conv_11 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=512, stride=(2, 2))
        self.n_Conv_12 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=128)
        self.n_Conv_13 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, stride=(2, 2))
        self.n_Conv_14 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_15 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_16 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_17 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_18 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_19 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_20 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_21 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_22 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_23 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_24 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_25 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_26 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_27 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_28 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_29 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_30 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_31 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_32 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_33 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=128)
        self.n_Conv_34 = nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128)
        self.n_Conv_35 = nn.Conv2d(kernel_size=(1, 1), in_channels=128, out_channels=512)
        self.n_Conv_36 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=1024, stride=(2, 2))
        self.n_Conv_37 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=256)
        self.n_Conv_38 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(2, 2))
        self.n_Conv_39 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_40 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_41 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_42 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_43 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_44 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_45 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_46 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_47 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_48 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_49 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_50 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_51 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_52 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_53 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_54 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_55 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_56 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_57 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_58 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_59 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_60 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_61 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_62 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_63 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_64 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_65 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_66 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_67 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_68 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_69 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_70 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_71 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_72 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_73 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_74 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_75 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_76 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_77 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_78 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_79 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_80 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_81 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_82 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_83 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_84 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_85 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_86 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_87 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_88 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_89 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_90 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_91 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_92 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_93 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_94 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_95 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_96 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_97 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_98 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, stride=(2, 2))
        self.n_Conv_99 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_100 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=1024, stride=(2, 2))
        self.n_Conv_101 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_102 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_103 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_104 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_105 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_106 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_107 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_108 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_109 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_110 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_111 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_112 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_113 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_114 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_115 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_116 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_117 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_118 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_119 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_120 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_121 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_122 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_123 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_124 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_125 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_126 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_127 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_128 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_129 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_130 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_131 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_132 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_133 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_134 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_135 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_136 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_137 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_138 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_139 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_140 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_141 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_142 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_143 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_144 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_145 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_146 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_147 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_148 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_149 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_150 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_151 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_152 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_153 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_154 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_155 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=256)
        self.n_Conv_156 = nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256)
        self.n_Conv_157 = nn.Conv2d(kernel_size=(1, 1), in_channels=256, out_channels=1024)
        self.n_Conv_158 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=2048, stride=(2, 2))
        self.n_Conv_159 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=512)
        self.n_Conv_160 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512, stride=(2, 2))
        self.n_Conv_161 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_162 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=512)
        self.n_Conv_163 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512)
        self.n_Conv_164 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_165 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=512)
        self.n_Conv_166 = nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=512)
        self.n_Conv_167 = nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=2048)
        self.n_Conv_168 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=4096, stride=(2, 2))
        self.n_Conv_169 = nn.Conv2d(kernel_size=(1, 1), in_channels=2048, out_channels=1024)
        self.n_Conv_170 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, stride=(2, 2))
        self.n_Conv_171 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_172 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=1024)
        self.n_Conv_173 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024)
        self.n_Conv_174 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_175 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=1024)
        self.n_Conv_176 = nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024)
        self.n_Conv_177 = nn.Conv2d(kernel_size=(1, 1), in_channels=1024, out_channels=4096)
        self.n_Conv_178 = nn.Conv2d(kernel_size=(1, 1), in_channels=4096, out_channels=9176, bias=False)

    def forward(self, *inputs):
        t_358, = inputs
        t_359 = t_358.permute(*[0, 3, 1, 2])
        t_359_padded = F.pad(t_359, [2, 3, 2, 3], value=0)
        t_360 = self.n_Conv_0(t_359_padded if devices.unet_needs_upcast else t_359_padded)
        t_361 = F.relu(t_360)
        t_361 = F.pad(t_361, [0, 1, 0, 1], value=float('-inf'))
        t_362 = self.n_MaxPool_0(t_361)
        t_363 = self.n_Conv_1(t_362)
        t_364 = self.n_Conv_2(t_362)
        t_365 = F.relu(t_364)
        t_365_padded = F.pad(t_365, [1, 1, 1, 1], value=0)
        t_366 = self.n_Conv_3(t_365_padded)
        t_367 = F.relu(t_366)
        t_368 = self.n_Conv_4(t_367)
        t_369 = torch.add(t_368, t_363)
        t_370 = F.relu(t_369)
        t_371 = self.n_Conv_5(t_370)
        t_372 = F.relu(t_371)
        t_372_padded = F.pad(t_372, [1, 1, 1, 1], value=0)
        t_373 = self.n_Conv_6(t_372_padded)
        t_374 = F.relu(t_373)
        t_375 = self.n_Conv_7(t_374)
        t_376 = torch.add(t_375, t_370)
        t_377 = F.relu(t_376)
        t_378 = self.n_Conv_8(t_377)
        t_379 = F.relu(t_378)
        t_379_padded = F.pad(t_379, [1, 1, 1, 1], value=0)
        t_380 = self.n_Conv_9(t_379_padded)
        t_381 = F.relu(t_380)
        t_382 = self.n_Conv_10(t_381)
        t_383 = torch.add(t_382, t_377)
        t_384 = F.relu(t_383)
        t_385 = self.n_Conv_11(t_384)
        t_386 = self.n_Conv_12(t_384)
        t_387 = F.relu(t_386)
        t_387_padded = F.pad(t_387, [0, 1, 0, 1], value=0)
        t_388 = self.n_Conv_13(t_387_padded)
        t_389 = F.relu(t_388)
        t_390 = self.n_Conv_14(t_389)
        t_391 = torch.add(t_390, t_385)
        t_392 = F.relu(t_391)
        t_393 = self.n_Conv_15(t_392)
        t_394 = F.relu(t_393)
        t_394_padded = F.pad(t_394, [1, 1, 1, 1], value=0)
        t_395 = self.n_Conv_16(t_394_padded)
        t_396 = F.relu(t_395)
        t_397 = self.n_Conv_17(t_396)
        t_398 = torch.add(t_397, t_392)
        t_399 = F.relu(t_398)
        t_400 = self.n_Conv_18(t_399)
        t_401 = F.relu(t_400)
        t_401_padded = F.pad(t_401, [1, 1, 1, 1], value=0)
        t_402 = self.n_Conv_19(t_401_padded)
        t_403 = F.relu(t_402)
        t_404 = self.n_Conv_20(t_403)
        t_405 = torch.add(t_404, t_399)
        t_406 = F.relu(t_405)
        t_407 = self.n_Conv_21(t_406)
        t_408 = F.relu(t_407)
        t_408_padded = F.pad(t_408, [1, 1, 1, 1], value=0)
        t_409 = self.n_Conv_22(t_408_padded)
        t_410 = F.relu(t_409)
        t_411 = self.n_Conv_23(t_410)
        t_412 = torch.add(t_411, t_406)
        t_413 = F.relu(t_412)
        t_414 = self.n_Conv_24(t_413)
        t_415 = F.relu(t_414)
        t_415_padded = F.pad(t_415, [1, 1, 1, 1], value=0)
        t_416 = self.n_Conv_25(t_415_padded)
        t_417 = F.relu(t_416)
        t_418 = self.n_Conv_26(t_417)
        t_419 = torch.add(t_418, t_413)
        t_420 = F.relu(t_419)
        t_421 = self.n_Conv_27(t_420)
        t_422 = F.relu(t_421)
        t_422_padded = F.pad(t_422, [1, 1, 1, 1], value=0)
        t_423 = self.n_Conv_28(t_422_padded)
        t_424 = F.relu(t_423)
        t_425 = self.n_Conv_29(t_424)
        t_426 = torch.add(t_425, t_420)
        t_427 = F.relu(t_426)
        t_428 = self.n_Conv_30(t_427)
        t_429 = F.relu(t_428)
        t_429_padded = F.pad(t_429, [1, 1, 1, 1], value=0)
        t_430 = self.n_Conv_31(t_429_padded)
        t_431 = F.relu(t_430)
        t_432 = self.n_Conv_32(t_431)
        t_433 = torch.add(t_432, t_427)
        t_434 = F.relu(t_433)
        t_435 = self.n_Conv_33(t_434)
        t_436 = F.relu(t_435)
        t_436_padded = F.pad(t_436, [1, 1, 1, 1], value=0)
        t_437 = self.n_Conv_34(t_436_padded)
        t_438 = F.relu(t_437)
        t_439 = self.n_Conv_35(t_438)
        t_440 = torch.add(t_439, t_434)
        t_441 = F.relu(t_440)
        t_442 = self.n_Conv_36(t_441)
        t_443 = self.n_Conv_37(t_441)
        t_444 = F.relu(t_443)
        t_444_padded = F.pad(t_444, [0, 1, 0, 1], value=0)
        t_445 = self.n_Conv_38(t_444_padded)
        t_446 = F.relu(t_445)
        t_447 = self.n_Conv_39(t_446)
        t_448 = torch.add(t_447, t_442)
        t_449 = F.relu(t_448)
        t_450 = self.n_Conv_40(t_449)
        t_451 = F.relu(t_450)
        t_451_padded = F.pad(t_451, [1, 1, 1, 1], value=0)
        t_452 = self.n_Conv_41(t_451_padded)
        t_453 = F.relu(t_452)
        t_454 = self.n_Conv_42(t_453)
        t_455 = torch.add(t_454, t_449)
        t_456 = F.relu(t_455)
        t_457 = self.n_Conv_43(t_456)
        t_458 = F.relu(t_457)
        t_458_padded = F.pad(t_458, [1, 1, 1, 1], value=0)
        t_459 = self.n_Conv_44(t_458_padded)
        t_460 = F.relu(t_459)
        t_461 = self.n_Conv_45(t_460)
        t_462 = torch.add(t_461, t_456)
        t_463 = F.relu(t_462)
        t_464 = self.n_Conv_46(t_463)
        t_465 = F.relu(t_464)
        t_465_padded = F.pad(t_465, [1, 1, 1, 1], value=0)
        t_466 = self.n_Conv_47(t_465_padded)
        t_467 = F.relu(t_466)
        t_468 = self.n_Conv_48(t_467)
        t_469 = torch.add(t_468, t_463)
        t_470 = F.relu(t_469)
        t_471 = self.n_Conv_49(t_470)
        t_472 = F.relu(t_471)
        t_472_padded = F.pad(t_472, [1, 1, 1, 1], value=0)
        t_473 = self.n_Conv_50(t_472_padded)
        t_474 = F.relu(t_473)
        t_475 = self.n_Conv_51(t_474)
        t_476 = torch.add(t_475, t_470)
        t_477 = F.relu(t_476)
        t_478 = self.n_Conv_52(t_477)
        t_479 = F.relu(t_478)
        t_479_padded = F.pad(t_479, [1, 1, 1, 1], value=0)
        t_480 = self.n_Conv_53(t_479_padded)
        t_481 = F.relu(t_480)
        t_482 = self.n_Conv_54(t_481)
        t_483 = torch.add(t_482, t_477)
        t_484 = F.relu(t_483)
        t_485 = self.n_Conv_55(t_484)
        t_486 = F.relu(t_485)
        t_486_padded = F.pad(t_486, [1, 1, 1, 1], value=0)
        t_487 = self.n_Conv_56(t_486_padded)
        t_488 = F.relu(t_487)
        t_489 = self.n_Conv_57(t_488)
        t_490 = torch.add(t_489, t_484)
        t_491 = F.relu(t_490)
        t_492 = self.n_Conv_58(t_491)
        t_493 = F.relu(t_492)
        t_493_padded = F.pad(t_493, [1, 1, 1, 1], value=0)
        t_494 = self.n_Conv_59(t_493_padded)
        t_495 = F.relu(t_494)
        t_496 = self.n_Conv_60(t_495)
        t_497 = torch.add(t_496, t_491)
        t_498 = F.relu(t_497)
        t_499 = self.n_Conv_61(t_498)
        t_500 = F.relu(t_499)
        t_500_padded = F.pad(t_500, [1, 1, 1, 1], value=0)
        t_501 = self.n_Conv_62(t_500_padded)
        t_502 = F.relu(t_501)
        t_503 = self.n_Conv_63(t_502)
        t_504 = torch.add(t_503, t_498)
        t_505 = F.relu(t_504)
        t_506 = self.n_Conv_64(t_505)
        t_507 = F.relu(t_506)
        t_507_padded = F.pad(t_507, [1, 1, 1, 1], value=0)
        t_508 = self.n_Conv_65(t_507_padded)
        t_509 = F.relu(t_508)
        t_510 = self.n_Conv_66(t_509)
        t_511 = torch.add(t_510, t_505)
        t_512 = F.relu(t_511)
        t_513 = self.n_Conv_67(t_512)
        t_514 = F.relu(t_513)
        t_514_padded = F.pad(t_514, [1, 1, 1, 1], value=0)
        t_515 = self.n_Conv_68(t_514_padded)
        t_516 = F.relu(t_515)
        t_517 = self.n_Conv_69(t_516)
        t_518 = torch.add(t_517, t_512)
        t_519 = F.relu(t_518)
        t_520 = self.n_Conv_70(t_519)
        t_521 = F.relu(t_520)
        t_521_padded = F.pad(t_521, [1, 1, 1, 1], value=0)
        t_522 = self.n_Conv_71(t_521_padded)
        t_523 = F.relu(t_522)
        t_524 = self.n_Conv_72(t_523)
        t_525 = torch.add(t_524, t_519)
        t_526 = F.relu(t_525)
        t_527 = self.n_Conv_73(t_526)
        t_528 = F.relu(t_527)
        t_528_padded = F.pad(t_528, [1, 1, 1, 1], value=0)
        t_529 = self.n_Conv_74(t_528_padded)
        t_530 = F.relu(t_529)
        t_531 = self.n_Conv_75(t_530)
        t_532 = torch.add(t_531, t_526)
        t_533 = F.relu(t_532)
        t_534 = self.n_Conv_76(t_533)
        t_535 = F.relu(t_534)
        t_535_padded = F.pad(t_535, [1, 1, 1, 1], value=0)
        t_536 = self.n_Conv_77(t_535_padded)
        t_537 = F.relu(t_536)
        t_538 = self.n_Conv_78(t_537)
        t_539 = torch.add(t_538, t_533)
        t_540 = F.relu(t_539)
        t_541 = self.n_Conv_79(t_540)
        t_542 = F.relu(t_541)
        t_542_padded = F.pad(t_542, [1, 1, 1, 1], value=0)
        t_543 = self.n_Conv_80(t_542_padded)
        t_544 = F.relu(t_543)
        t_545 = self.n_Conv_81(t_544)
        t_546 = torch.add(t_545, t_540)
        t_547 = F.relu(t_546)
        t_548 = self.n_Conv_82(t_547)
        t_549 = F.relu(t_548)
        t_549_padded = F.pad(t_549, [1, 1, 1, 1], value=0)
        t_550 = self.n_Conv_83(t_549_padded)
        t_551 = F.relu(t_550)
        t_552 = self.n_Conv_84(t_551)
        t_553 = torch.add(t_552, t_547)
        t_554 = F.relu(t_553)
        t_555 = self.n_Conv_85(t_554)
        t_556 = F.relu(t_555)
        t_556_padded = F.pad(t_556, [1, 1, 1, 1], value=0)
        t_557 = self.n_Conv_86(t_556_padded)
        t_558 = F.relu(t_557)
        t_559 = self.n_Conv_87(t_558)
        t_560 = torch.add(t_559, t_554)
        t_561 = F.relu(t_560)
        t_562 = self.n_Conv_88(t_561)
        t_563 = F.relu(t_562)
        t_563_padded = F.pad(t_563, [1, 1, 1, 1], value=0)
        t_564 = self.n_Conv_89(t_563_padded)
        t_565 = F.relu(t_564)
        t_566 = self.n_Conv_90(t_565)
        t_567 = torch.add(t_566, t_561)
        t_568 = F.relu(t_567)
        t_569 = self.n_Conv_91(t_568)
        t_570 = F.relu(t_569)
        t_570_padded = F.pad(t_570, [1, 1, 1, 1], value=0)
        t_571 = self.n_Conv_92(t_570_padded)
        t_572 = F.relu(t_571)
        t_573 = self.n_Conv_93(t_572)
        t_574 = torch.add(t_573, t_568)
        t_575 = F.relu(t_574)
        t_576 = self.n_Conv_94(t_575)
        t_577 = F.relu(t_576)
        t_577_padded = F.pad(t_577, [1, 1, 1, 1], value=0)
        t_578 = self.n_Conv_95(t_577_padded)
        t_579 = F.relu(t_578)
        t_580 = self.n_Conv_96(t_579)
        t_581 = torch.add(t_580, t_575)
        t_582 = F.relu(t_581)
        t_583 = self.n_Conv_97(t_582)
        t_584 = F.relu(t_583)
        t_584_padded = F.pad(t_584, [0, 1, 0, 1], value=0)
        t_585 = self.n_Conv_98(t_584_padded)
        t_586 = F.relu(t_585)
        t_587 = self.n_Conv_99(t_586)
        t_588 = self.n_Conv_100(t_582)
        t_589 = torch.add(t_587, t_588)
        t_590 = F.relu(t_589)
        t_591 = self.n_Conv_101(t_590)
        t_592 = F.relu(t_591)
        t_592_padded = F.pad(t_592, [1, 1, 1, 1], value=0)
        t_593 = self.n_Conv_102(t_592_padded)
        t_594 = F.relu(t_593)
        t_595 = self.n_Conv_103(t_594)
        t_596 = torch.add(t_595, t_590)
        t_597 = F.relu(t_596)
        t_598 = self.n_Conv_104(t_597)
        t_599 = F.relu(t_598)
        t_599_padded = F.pad(t_599, [1, 1, 1, 1], value=0)
        t_600 = self.n_Conv_105(t_599_padded)
        t_601 = F.relu(t_600)
        t_602 = self.n_Conv_106(t_601)
        t_603 = torch.add(t_602, t_597)
        t_604 = F.relu(t_603)
        t_605 = self.n_Conv_107(t_604)
        t_606 = F.relu(t_605)
        t_606_padded = F.pad(t_606, [1, 1, 1, 1], value=0)
        t_607 = self.n_Conv_108(t_606_padded)
        t_608 = F.relu(t_607)
        t_609 = self.n_Conv_109(t_608)
        t_610 = torch.add(t_609, t_604)
        t_611 = F.relu(t_610)
        t_612 = self.n_Conv_110(t_611)
        t_613 = F.relu(t_612)
        t_613_padded = F.pad(t_613, [1, 1, 1, 1], value=0)
        t_614 = self.n_Conv_111(t_613_padded)
        t_615 = F.relu(t_614)
        t_616 = self.n_Conv_112(t_615)
        t_617 = torch.add(t_616, t_611)
        t_618 = F.relu(t_617)
        t_619 = self.n_Conv_113(t_618)
        t_620 = F.relu(t_619)
        t_620_padded = F.pad(t_620, [1, 1, 1, 1], value=0)
        t_621 = self.n_Conv_114(t_620_padded)
        t_622 = F.relu(t_621)
        t_623 = self.n_Conv_115(t_622)
        t_624 = torch.add(t_623, t_618)
        t_625 = F.relu(t_624)
        t_626 = self.n_Conv_116(t_625)
        t_627 = F.relu(t_626)
        t_627_padded = F.pad(t_627, [1, 1, 1, 1], value=0)
        t_628 = self.n_Conv_117(t_627_padded)
        t_629 = F.relu(t_628)
        t_630 = self.n_Conv_118(t_629)
        t_631 = torch.add(t_630, t_625)
        t_632 = F.relu(t_631)
        t_633 = self.n_Conv_119(t_632)
        t_634 = F.relu(t_633)
        t_634_padded = F.pad(t_634, [1, 1, 1, 1], value=0)
        t_635 = self.n_Conv_120(t_634_padded)
        t_636 = F.relu(t_635)
        t_637 = self.n_Conv_121(t_636)
        t_638 = torch.add(t_637, t_632)
        t_639 = F.relu(t_638)
        t_640 = self.n_Conv_122(t_639)
        t_641 = F.relu(t_640)
        t_641_padded = F.pad(t_641, [1, 1, 1, 1], value=0)
        t_642 = self.n_Conv_123(t_641_padded)
        t_643 = F.relu(t_642)
        t_644 = self.n_Conv_124(t_643)
        t_645 = torch.add(t_644, t_639)
        t_646 = F.relu(t_645)
        t_647 = self.n_Conv_125(t_646)
        t_648 = F.relu(t_647)
        t_648_padded = F.pad(t_648, [1, 1, 1, 1], value=0)
        t_649 = self.n_Conv_126(t_648_padded)
        t_650 = F.relu(t_649)
        t_651 = self.n_Conv_127(t_650)
        t_652 = torch.add(t_651, t_646)
        t_653 = F.relu(t_652)
        t_654 = self.n_Conv_128(t_653)
        t_655 = F.relu(t_654)
        t_655_padded = F.pad(t_655, [1, 1, 1, 1], value=0)
        t_656 = self.n_Conv_129(t_655_padded)
        t_657 = F.relu(t_656)
        t_658 = self.n_Conv_130(t_657)
        t_659 = torch.add(t_658, t_653)
        t_660 = F.relu(t_659)
        t_661 = self.n_Conv_131(t_660)
        t_662 = F.relu(t_661)
        t_662_padded = F.pad(t_662, [1, 1, 1, 1], value=0)
        t_663 = self.n_Conv_132(t_662_padded)
        t_664 = F.relu(t_663)
        t_665 = self.n_Conv_133(t_664)
        t_666 = torch.add(t_665, t_660)
        t_667 = F.relu(t_666)
        t_668 = self.n_Conv_134(t_667)
        t_669 = F.relu(t_668)
        t_669_padded = F.pad(t_669, [1, 1, 1, 1], value=0)
        t_670 = self.n_Conv_135(t_669_padded)
        t_671 = F.relu(t_670)
        t_672 = self.n_Conv_136(t_671)
        t_673 = torch.add(t_672, t_667)
        t_674 = F.relu(t_673)
        t_675 = self.n_Conv_137(t_674)
        t_676 = F.relu(t_675)
        t_676_padded = F.pad(t_676, [1, 1, 1, 1], value=0)
        t_677 = self.n_Conv_138(t_676_padded)
        t_678 = F.relu(t_677)
        t_679 = self.n_Conv_139(t_678)
        t_680 = torch.add(t_679, t_674)
        t_681 = F.relu(t_680)
        t_682 = self.n_Conv_140(t_681)
        t_683 = F.relu(t_682)
        t_683_padded = F.pad(t_683, [1, 1, 1, 1], value=0)
        t_684 = self.n_Conv_141(t_683_padded)
        t_685 = F.relu(t_684)
        t_686 = self.n_Conv_142(t_685)
        t_687 = torch.add(t_686, t_681)
        t_688 = F.relu(t_687)
        t_689 = self.n_Conv_143(t_688)
        t_690 = F.relu(t_689)
        t_690_padded = F.pad(t_690, [1, 1, 1, 1], value=0)
        t_691 = self.n_Conv_144(t_690_padded)
        t_692 = F.relu(t_691)
        t_693 = self.n_Conv_145(t_692)
        t_694 = torch.add(t_693, t_688)
        t_695 = F.relu(t_694)
        t_696 = self.n_Conv_146(t_695)
        t_697 = F.relu(t_696)
        t_697_padded = F.pad(t_697, [1, 1, 1, 1], value=0)
        t_698 = self.n_Conv_147(t_697_padded)
        t_699 = F.relu(t_698)
        t_700 = self.n_Conv_148(t_699)
        t_701 = torch.add(t_700, t_695)
        t_702 = F.relu(t_701)
        t_703 = self.n_Conv_149(t_702)
        t_704 = F.relu(t_703)
        t_704_padded = F.pad(t_704, [1, 1, 1, 1], value=0)
        t_705 = self.n_Conv_150(t_704_padded)
        t_706 = F.relu(t_705)
        t_707 = self.n_Conv_151(t_706)
        t_708 = torch.add(t_707, t_702)
        t_709 = F.relu(t_708)
        t_710 = self.n_Conv_152(t_709)
        t_711 = F.relu(t_710)
        t_711_padded = F.pad(t_711, [1, 1, 1, 1], value=0)
        t_712 = self.n_Conv_153(t_711_padded)
        t_713 = F.relu(t_712)
        t_714 = self.n_Conv_154(t_713)
        t_715 = torch.add(t_714, t_709)
        t_716 = F.relu(t_715)
        t_717 = self.n_Conv_155(t_716)
        t_718 = F.relu(t_717)
        t_718_padded = F.pad(t_718, [1, 1, 1, 1], value=0)
        t_719 = self.n_Conv_156(t_718_padded)
        t_720 = F.relu(t_719)
        t_721 = self.n_Conv_157(t_720)
        t_722 = torch.add(t_721, t_716)
        t_723 = F.relu(t_722)
        t_724 = self.n_Conv_158(t_723)
        t_725 = self.n_Conv_159(t_723)
        t_726 = F.relu(t_725)
        t_726_padded = F.pad(t_726, [0, 1, 0, 1], value=0)
        t_727 = self.n_Conv_160(t_726_padded)
        t_728 = F.relu(t_727)
        t_729 = self.n_Conv_161(t_728)
        t_730 = torch.add(t_729, t_724)
        t_731 = F.relu(t_730)
        t_732 = self.n_Conv_162(t_731)
        t_733 = F.relu(t_732)
        t_733_padded = F.pad(t_733, [1, 1, 1, 1], value=0)
        t_734 = self.n_Conv_163(t_733_padded)
        t_735 = F.relu(t_734)
        t_736 = self.n_Conv_164(t_735)
        t_737 = torch.add(t_736, t_731)
        t_738 = F.relu(t_737)
        t_739 = self.n_Conv_165(t_738)
        t_740 = F.relu(t_739)
        t_740_padded = F.pad(t_740, [1, 1, 1, 1], value=0)
        t_741 = self.n_Conv_166(t_740_padded)
        t_742 = F.relu(t_741)
        t_743 = self.n_Conv_167(t_742)
        t_744 = torch.add(t_743, t_738)
        t_745 = F.relu(t_744)
        t_746 = self.n_Conv_168(t_745)
        t_747 = self.n_Conv_169(t_745)
        t_748 = F.relu(t_747)
        t_748_padded = F.pad(t_748, [0, 1, 0, 1], value=0)
        t_749 = self.n_Conv_170(t_748_padded)
        t_750 = F.relu(t_749)
        t_751 = self.n_Conv_171(t_750)
        t_752 = torch.add(t_751, t_746)
        t_753 = F.relu(t_752)
        t_754 = self.n_Conv_172(t_753)
        t_755 = F.relu(t_754)
        t_755_padded = F.pad(t_755, [1, 1, 1, 1], value=0)
        t_756 = self.n_Conv_173(t_755_padded)
        t_757 = F.relu(t_756)
        t_758 = self.n_Conv_174(t_757)
        t_759 = torch.add(t_758, t_753)
        t_760 = F.relu(t_759)
        t_761 = self.n_Conv_175(t_760)
        t_762 = F.relu(t_761)
        t_762_padded = F.pad(t_762, [1, 1, 1, 1], value=0)
        t_763 = self.n_Conv_176(t_762_padded)
        t_764 = F.relu(t_763)
        t_765 = self.n_Conv_177(t_764)
        t_766 = torch.add(t_765, t_760)
        t_767 = F.relu(t_766)
        t_768 = self.n_Conv_178(t_767)
        t_769 = F.avg_pool2d(t_768, kernel_size=t_768.shape[-2:])
        t_770 = torch.squeeze(t_769, 3)
        t_770 = torch.squeeze(t_770, 2)
        t_771 = torch.sigmoid(t_770)
        return t_771

    def load_state_dict(self, state_dict, **kwargs):
        self.tags = state_dict.get('tags', [])
        super(DeepDanbooruModel, self).load_state_dict({k: v for k, v in state_dict.items() if k != 'tags'})


class HypernetworkModule(torch.nn.Module):
    activation_dict = {'linear': torch.nn.Identity, 'relu': torch.nn.ReLU, 'leakyrelu': torch.nn.LeakyReLU, 'elu': torch.nn.ELU, 'swish': torch.nn.Hardswish, 'tanh': torch.nn.Tanh, 'sigmoid': torch.nn.Sigmoid}
    activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'})

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal', add_layer_norm=False, activate_output=False, dropout_structure=None):
        super().__init__()
        self.multiplier = 1.0
        assert layer_structure is not None, 'layer_structure must not be None'
        assert layer_structure[0] == 1, 'Multiplier Sequence should start with size 1!'
        assert layer_structure[-1] == 1, 'Multiplier Sequence should end with size 1!'
        linears = []
        for i in range(len(layer_structure) - 1):
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i + 1])))
            if activation_func == 'linear' or activation_func is None or i >= len(layer_structure) - 2 and not activate_output:
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i + 1])))
            if dropout_structure is not None and dropout_structure[i + 1] > 0:
                assert 0 < dropout_structure[i + 1] < 1, 'Dropout probability should be 0 or float between 0 and 1!'
                linears.append(torch.nn.Dropout(p=dropout_structure[i + 1]))
        self.linear = torch.nn.Sequential(*linears)
        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    w, b = layer.weight.data, layer.bias.data
                    if weight_init == 'Normal' or type(layer) == torch.nn.LayerNorm:
                        normal_(w, mean=0.0, std=0.01)
                        normal_(b, mean=0.0, std=0)
                    elif weight_init == 'XavierUniform':
                        xavier_uniform_(w)
                        zeros_(b)
                    elif weight_init == 'XavierNormal':
                        xavier_normal_(w)
                        zeros_(b)
                    elif weight_init == 'KaimingUniform':
                        kaiming_uniform_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    elif weight_init == 'KaimingNormal':
                        kaiming_normal_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    else:
                        raise KeyError(f'Key {weight_init} is not defined as initialization!')
        devices.torch_npu_set_device()
        self

    def fix_old_state_dict(self, state_dict):
        changes = {'linear1.bias': 'linear.0.bias', 'linear1.weight': 'linear.0.weight', 'linear2.bias': 'linear.1.bias', 'linear2.weight': 'linear.1.weight'}
        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue
            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x):
        return x + self.linear(x) * (self.multiplier if not self.training else 1)

    def trainables(self):
        layer_structure = []
        for layer in self.linear:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                layer_structure += [layer.weight, layer.bias]
        return layer_structure


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding"""

    def __init__(self, img_size: 'Optional[int]'=224, patch_size: 'int'=16, in_chans: 'int'=3, embed_dim: 'int'=768, flatten: 'bool'=True, bias: 'bool'=True, strict_img_size: 'bool'=True, dynamic_img_pad: 'bool'=False, dtype=None, device=None):
        super().__init__()
        self.patch_size = patch_size, patch_size
        if img_size is not None:
            self.img_size = img_size, img_size
            self.grid_size = tuple([(s // p) for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding
        return embedding

    def forward(self, t, dtype, **kwargs):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class VectorEmbedder(nn.Module):
    """Embeds a flat vector of dimension input_dim"""

    def __init__(self, input_dim: 'int', hidden_size: 'int', dtype=None, device=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size, bias=True, dtype=dtype, device=device), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.mlp(x)


class QkvLinear(torch.nn.Linear):
    pass


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: 'int', elementwise_affine: 'bool'=False, eps: 'float'=1e-06, device=None, dtype=None):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.learnable_scale = elementwise_affine
        if self.learnable_scale:
            self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        x = self._norm(x)
        if self.learnable_scale:
            return x * self.weight
        else:
            return x


def attention(q, k, v, heads, mask=None):
    """Convenience wrapper around a basic attention operation"""
    b, _, dim_head = q.shape
    dim_head //= heads
    q, k, v = [t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]


class SelfAttention(nn.Module):
    ATTENTION_MODES = 'xformers', 'torch', 'torch-hb', 'math', 'debug'

    def __init__(self, dim: 'int', num_heads: 'int'=8, qkv_bias: 'bool'=False, qk_scale: 'Optional[float]'=None, attn_mode: 'str'='xformers', pre_only: 'bool'=False, qk_norm: 'Optional[str]'=None, rmsnorm: 'bool'=False, dtype=None, device=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = QkvLinear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        if not pre_only:
            self.proj = nn.Linear(dim, dim, dtype=dtype, device=device)
        assert attn_mode in self.ATTENTION_MODES
        self.attn_mode = attn_mode
        self.pre_only = pre_only
        if qk_norm == 'rms':
            self.ln_q = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
            self.ln_k = RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
        elif qk_norm == 'ln':
            self.ln_q = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
            self.ln_k = nn.LayerNorm(self.head_dim, elementwise_affine=True, eps=1e-06, dtype=dtype, device=device)
        elif qk_norm is None:
            self.ln_q = nn.Identity()
            self.ln_k = nn.Identity()
        else:
            raise ValueError(qk_norm)

    def pre_attention(self, x: 'torch.Tensor'):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = split_qkv(qkv, self.head_dim)
        q = self.ln_q(q).reshape(q.shape[0], q.shape[1], -1)
        k = self.ln_k(k).reshape(q.shape[0], q.shape[1], -1)
        return q, k, v

    def post_attention(self, x: 'torch.Tensor') ->torch.Tensor:
        assert not self.pre_only
        x = self.proj(x)
        return x

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        q, k, v = self.pre_attention(x)
        x = attention(q, k, v, self.num_heads)
        x = self.post_attention(x)
        return x


class SwiGLUFeedForward(nn.Module):

    def __init__(self, dim: 'int', hidden_dim: 'int', multiple_of: 'int', ffn_dim_multiplier: 'Optional[float]'=None):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, dtype=None, device=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, dtype=dtype, device=device)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def modulate(x, shift, scale):
    if shift is None:
        shift = torch.zeros_like(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DismantledBlock(nn.Module):
    """A DiT block with gated adaptive layer norm (adaLN) conditioning."""
    ATTENTION_MODES = 'xformers', 'torch', 'torch-hb', 'math', 'debug'

    def __init__(self, hidden_size: 'int', num_heads: 'int', mlp_ratio: 'float'=4.0, attn_mode: 'str'='xformers', qkv_bias: 'bool'=False, pre_only: 'bool'=False, rmsnorm: 'bool'=False, scale_mod_only: 'bool'=False, swiglu: 'bool'=False, qk_norm: 'Optional[str]'=None, dtype=None, device=None, **block_kwargs):
        super().__init__()
        assert attn_mode in self.ATTENTION_MODES
        if not rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        else:
            self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=pre_only, qk_norm=qk_norm, rmsnorm=rmsnorm, dtype=dtype, device=device)
        if not pre_only:
            if not rmsnorm:
                self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
            else:
                self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if not pre_only:
            if not swiglu:
                self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU(approximate='tanh'), dtype=dtype, device=device)
            else:
                self.mlp = SwiGLUFeedForward(dim=hidden_size, hidden_dim=mlp_hidden_dim, multiple_of=256)
        self.scale_mod_only = scale_mod_only
        if not scale_mod_only:
            n_mods = 6 if not pre_only else 2
        else:
            n_mods = 4 if not pre_only else 1
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, n_mods * hidden_size, bias=True, dtype=dtype, device=device))
        self.pre_only = pre_only

    def pre_attention(self, x: 'torch.Tensor', c: 'torch.Tensor'):
        assert x is not None, 'pre_attention called with None input'
        if not self.pre_only:
            if not self.scale_mod_only:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            else:
                shift_msa = None
                shift_mlp = None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, (x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        else:
            if not self.scale_mod_only:
                shift_msa, scale_msa = self.adaLN_modulation(c).chunk(2, dim=1)
            else:
                shift_msa = None
                scale_msa = self.adaLN_modulation(c)
            qkv = self.attn.pre_attention(modulate(self.norm1(x), shift_msa, scale_msa))
            return qkv, None

    def post_attention(self, attn, x, gate_msa, shift_mlp, scale_mlp, gate_mlp):
        assert not self.pre_only
        x = x + gate_msa.unsqueeze(1) * self.attn.post_attention(attn)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def forward(self, x: 'torch.Tensor', c: 'torch.Tensor') ->torch.Tensor:
        assert not self.pre_only
        (q, k, v), intermediates = self.pre_attention(x, c)
        attn = attention(q, k, v, self.attn.num_heads)
        return self.post_attention(attn, *intermediates)


def block_mixing(context, x, context_block, x_block, c):
    assert context is not None, 'block_mixing called with None context'
    context_qkv, context_intermediates = context_block.pre_attention(context, c)
    x_qkv, x_intermediates = x_block.pre_attention(x, c)
    o = []
    for t in range(3):
        o.append(torch.cat((context_qkv[t], x_qkv[t]), dim=1))
    q, k, v = tuple(o)
    attn = attention(q, k, v, x_block.attn.num_heads)
    context_attn, x_attn = attn[:, :context_qkv[0].shape[1]], attn[:, context_qkv[0].shape[1]:]
    if not context_block.pre_only:
        context = context_block.post_attention(context_attn, *context_intermediates)
    else:
        context = None
    x = x_block.post_attention(x_attn, *x_intermediates)
    return context, x


class JointBlock(nn.Module):
    """just a small wrapper to serve as a fsdp unit"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_only = kwargs.pop('pre_only')
        qk_norm = kwargs.pop('qk_norm', None)
        self.context_block = DismantledBlock(*args, pre_only=pre_only, qk_norm=qk_norm, **kwargs)
        self.x_block = DismantledBlock(*args, pre_only=False, qk_norm=qk_norm, **kwargs)

    def forward(self, *args, **kwargs):
        return block_mixing(*args, context_block=self.context_block, x_block=self.x_block, **kwargs)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size: 'int', patch_size: 'int', out_channels: 'int', total_out_channels: 'Optional[int]'=None, dtype=None, device=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06, dtype=dtype, device=device)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True, dtype=dtype, device=device) if total_out_channels is None else nn.Linear(hidden_size, total_out_channels, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device))

    def forward(self, x: 'torch.Tensor', c: 'torch.Tensor') ->torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MMDiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(self, input_size: 'int'=32, patch_size: 'int'=2, in_channels: 'int'=4, depth: 'int'=28, mlp_ratio: 'float'=4.0, learn_sigma: 'bool'=False, adm_in_channels: 'Optional[int]'=None, context_embedder_config: 'Optional[Dict]'=None, register_length: 'int'=0, attn_mode: 'str'='torch', rmsnorm: 'bool'=False, scale_mod_only: 'bool'=False, swiglu: 'bool'=False, out_channels: 'Optional[int]'=None, pos_embed_scaling_factor: 'Optional[float]'=None, pos_embed_offset: 'Optional[float]'=None, pos_embed_max_size: 'Optional[int]'=None, num_patches=None, qk_norm: 'Optional[str]'=None, qkv_bias: 'bool'=True, dtype=None, device=None):
        super().__init__()
        self.dtype = dtype
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        default_out_channels = in_channels * 2 if learn_sigma else in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.patch_size = patch_size
        self.pos_embed_scaling_factor = pos_embed_scaling_factor
        self.pos_embed_offset = pos_embed_offset
        self.pos_embed_max_size = pos_embed_max_size
        hidden_size = 64 * depth
        num_heads = depth
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=self.pos_embed_max_size is None, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype, device=device)
        if adm_in_channels is not None:
            assert isinstance(adm_in_channels, int)
            self.y_embedder = VectorEmbedder(adm_in_channels, hidden_size, dtype=dtype, device=device)
        self.context_embedder = nn.Identity()
        if context_embedder_config is not None:
            if context_embedder_config['target'] == 'torch.nn.Linear':
                self.context_embedder = nn.Linear(**context_embedder_config['params'], dtype=dtype, device=device)
        self.register_length = register_length
        if self.register_length > 0:
            self.register = nn.Parameter(torch.randn(1, register_length, hidden_size, dtype=dtype, device=device))
        if num_patches is not None:
            self.register_buffer('pos_embed', torch.zeros(1, num_patches, hidden_size, dtype=dtype, device=device))
        else:
            self.pos_embed = None
        self.joint_blocks = nn.ModuleList([JointBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attn_mode=attn_mode, pre_only=i == depth - 1, rmsnorm=rmsnorm, scale_mod_only=scale_mod_only, swiglu=swiglu, qk_norm=qk_norm, dtype=dtype, device=device) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=dtype, device=device)

    def cropped_pos_embed(self, hw):
        assert self.pos_embed_max_size is not None
        p = self.x_embedder.patch_size[0]
        h, w = hw
        h = h // p
        w = w // p
        assert h <= self.pos_embed_max_size, (h, self.pos_embed_max_size)
        assert w <= self.pos_embed_max_size, (w, self.pos_embed_max_size)
        top = (self.pos_embed_max_size - h) // 2
        left = (self.pos_embed_max_size - w) // 2
        spatial_pos_embed = rearrange(self.pos_embed, '1 (h w) c -> 1 h w c', h=self.pos_embed_max_size, w=self.pos_embed_max_size)
        spatial_pos_embed = spatial_pos_embed[:, top:top + h, left:left + w, :]
        spatial_pos_embed = rearrange(spatial_pos_embed, '1 h w c -> 1 (h w) c')
        return spatial_pos_embed

    def unpatchify(self, x, hw=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
            h = h // p
            w = w // p
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_core_with_concat(self, x: 'torch.Tensor', c_mod: 'torch.Tensor', context: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        if self.register_length > 0:
            context = torch.cat((repeat(self.register, '1 ... -> b ...', b=x.shape[0]), context if context is not None else torch.Tensor([]).type_as(x)), 1)
        for block in self.joint_blocks:
            context, x = block(context, x, c=c_mod)
        x = self.final_layer(x, c_mod)
        return x

    def forward(self, x: 'torch.Tensor', t: 'torch.Tensor', y: 'Optional[torch.Tensor]'=None, context: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        hw = x.shape[-2:]
        x = self.x_embedder(x) + self.cropped_pos_embed(hw)
        c = self.t_embedder(t, dtype=x.dtype)
        if y is not None:
            y = self.y_embedder(y)
            c = c + y
        context = self.context_embedder(context)
        x = self.forward_core_with_concat(x, c, context)
        x = self.unpatchify(x, hw=hw)
        return x


class AutocastLinear(nn.Linear):
    """Same as usual linear layer, but casts its weights to whatever the parameter type is.

    This is different from torch.autocast in a way that float16 layer processing float32 input
    will return float16 with autocast on, and float32 with this. T5 seems to be fucked
    if you do it in full float16 (returning almost all zeros in the final output).
    """

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias if self.bias is not None else None)


class CLIPAttention(torch.nn.Module):

    def __init__(self, embed_dim, heads, dtype, device):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = attention(q, k, v, self.heads, mask)
        return self.out_proj(out)


ACTIVATIONS = {'quick_gelu': lambda a: a * torch.sigmoid(1.702 * a), 'gelu': torch.nn.functional.gelu}


class CLIPLayer(torch.nn.Module):

    def __init__(self, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.self_attn = CLIPAttention(embed_dim, heads, dtype, device)
        self.layer_norm2 = nn.LayerNorm(embed_dim, dtype=dtype, device=device)
        self.mlp = Mlp(embed_dim, intermediate_size, embed_dim, act_layer=ACTIVATIONS[intermediate_activation], dtype=dtype, device=device)

    def forward(self, x, mask=None):
        x += self.self_attn(self.layer_norm1(x), mask)
        x += self.mlp(self.layer_norm2(x))
        return x


class CLIPEncoder(torch.nn.Module):

    def __init__(self, num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device):
        super().__init__()
        self.layers = torch.nn.ModuleList([CLIPLayer(embed_dim, heads, intermediate_size, intermediate_activation, dtype, device) for i in range(num_layers)])

    def forward(self, x, mask=None, intermediate_output=None):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output
        intermediate = None
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            if i == intermediate_output:
                intermediate = x.clone()
        return x, intermediate


class CLIPEmbeddings(torch.nn.Module):

    def __init__(self, embed_dim, vocab_size=49408, num_positions=77, dtype=None, device=None, textual_inversion_key='clip_l'):
        super().__init__()
        self.token_embedding = sd_hijack.TextualInversionEmbeddings(vocab_size, embed_dim, dtype=dtype, device=device, textual_inversion_key=textual_inversion_key)
        self.position_embedding = torch.nn.Embedding(num_positions, embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens):
        return self.token_embedding(input_tokens) + self.position_embedding.weight


class CLIPTextModel_(torch.nn.Module):

    def __init__(self, config_dict, dtype, device):
        num_layers = config_dict['num_hidden_layers']
        embed_dim = config_dict['hidden_size']
        heads = config_dict['num_attention_heads']
        intermediate_size = config_dict['intermediate_size']
        intermediate_activation = config_dict['hidden_act']
        super().__init__()
        self.embeddings = CLIPEmbeddings(embed_dim, dtype=torch.float32, device=device, textual_inversion_key=config_dict.get('textual_inversion_key', 'clip_l'))
        self.encoder = CLIPEncoder(num_layers, embed_dim, heads, intermediate_size, intermediate_activation, dtype, device)
        self.final_layer_norm = nn.LayerNorm(embed_dim, dtype=dtype, device=device)

    def forward(self, input_tokens, intermediate_output=None, final_layer_norm_intermediate=True):
        x = self.embeddings(input_tokens)
        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float('-inf')).triu_(1)
        x, i = self.encoder(x, mask=causal_mask, intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)
        pooled_output = x[torch.arange(x.shape[0], device=x.device), input_tokens.argmax(dim=-1)]
        return x, i, pooled_output


class CLIPTextModel(torch.nn.Module):

    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict['num_hidden_layers']
        self.text_model = CLIPTextModel_(config_dict, dtype, device)
        embed_dim = config_dict['hidden_size']
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
        self.text_projection.weight.copy_(torch.eye(embed_dim))
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding = embeddings

    def forward(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return x[0], x[1], out, x[2]


class ClipTokenWeightEncoder:

    def encode_token_weights(self, token_weight_pairs):
        tokens = [a[0] for a in token_weight_pairs[0]]
        out, pooled = self([tokens])
        if pooled is not None:
            first_pooled = pooled[0:1].cpu()
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return torch.cat(output, dim=-2).cpu(), first_pooled


class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = ['last', 'pooled', 'hidden']

    def __init__(self, device='cpu', max_length=77, layer='last', layer_idx=None, textmodel_json_config=None, dtype=None, model_class=CLIPTextModel, special_tokens=None, layer_norm_hidden_state=True, return_projected_pooled=True):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer = model_class(textmodel_json_config, dtype, device)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens if special_tokens is not None else {'start': 49406, 'end': 49407, 'pad': 49407}
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == 'hidden':
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({'layer': layer_idx})
        self.options_default = self.layer, self.layer_idx, self.return_projected_pooled

    def set_clip_options(self, options):
        layer_idx = options.get('layer', self.layer_idx)
        self.return_projected_pooled = options.get('projected_pooled', self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = 'last'
        else:
            self.layer = 'hidden'
            self.layer_idx = layer_idx

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        tokens = torch.asarray(tokens, dtype=torch.int64, device=backup_embeds.weight.device)
        outputs = self.transformer(tokens, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)
        self.transformer.set_input_embeddings(backup_embeds)
        if self.layer == 'last':
            z = outputs[0]
        else:
            z = outputs[1]
        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        return z.float(), pooled_output


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface"""

    def __init__(self, config, device='cpu', layer='penultimate', layer_idx=None, dtype=None):
        if layer == 'penultimate':
            layer = 'hidden'
            layer_idx = -2
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype, special_tokens={'start': 49406, 'end': 49407, 'pad': 0}, layer_norm_hidden_state=False)


class T5DenseGatedActDense(torch.nn.Module):

    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.wi_0 = AutocastLinear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = AutocastLinear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = AutocastLinear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        hidden_gelu = torch.nn.functional.gelu(self.wi_0(x), approximate='tanh')
        hidden_linear = self.wi_1(x)
        x = hidden_gelu * hidden_linear
        x = self.wo(x)
        return x


class T5LayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-06, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class T5LayerFF(torch.nn.Module):

    def __init__(self, model_dim, ff_dim, dtype, device):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(torch.nn.Module):

    def __init__(self, model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.q = AutocastLinear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = AutocastLinear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = AutocastLinear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = AutocastLinear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = torch.nn.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, bidirectional=True, num_buckets=self.relative_attention_num_buckets, max_distance=self.relative_attention_max_distance)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, x, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device)
        if past_bias is not None:
            mask = past_bias
        else:
            mask = None
        out = attention(q, k * (k.shape[-1] / self.num_heads) ** 0.5, v, self.num_heads, mask if mask is not None else None)
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):

    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype, device)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), past_bias=past_bias)
        x += output
        return x, past_bias


class T5Block(torch.nn.Module):

    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device):
        super().__init__()
        self.layer = torch.nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias, dtype, device))
        self.layer.append(T5LayerFF(model_dim, ff_dim, dtype, device))

    def forward(self, x, past_bias=None):
        x, past_bias = self.layer[0](x, past_bias)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(torch.nn.Module):

    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, num_heads, vocab_size, dtype, device):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, model_dim, device=device)
        self.block = torch.nn.ModuleList([T5Block(model_dim, inner_dim, ff_dim, num_heads, relative_attention_bias=i == 0, dtype=dtype, device=device) for i in range(num_layers)])
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, input_ids, intermediate_output=None, final_layer_norm_intermediate=True):
        intermediate = None
        x = self.embed_tokens(input_ids)
        past_bias = None
        for i, layer in enumerate(self.block):
            x, past_bias = layer(x, past_bias)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)
        return x, intermediate


class T5(torch.nn.Module):

    def __init__(self, config_dict, dtype, device):
        super().__init__()
        self.num_layers = config_dict['num_layers']
        self.encoder = T5Stack(self.num_layers, config_dict['d_model'], config_dict['d_model'], config_dict['d_ff'], config_dict['num_heads'], config_dict['vocab_size'], dtype, device)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.encoder.embed_tokens = embeddings

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)


class T5XXLModel(SDClipModel):
    """Wraps the T5-XXL model into the SD-CLIP-Model interface for convenience"""

    def __init__(self, config, device='cpu', layer='last', layer_idx=None, dtype=None):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=config, dtype=dtype, special_tokens={'end': 1, 'pad': 0}, model_class=T5)


class Sd3T5(torch.nn.Module):

    def __init__(self, t5xxl):
        super().__init__()
        self.t5xxl = t5xxl
        self.tokenizer = T5TokenizerFast.from_pretrained('google/t5-v1_1-xxl')
        empty = self.tokenizer('', padding='max_length', max_length=2)['input_ids']
        self.id_end = empty[0]
        self.id_pad = empty[1]

    def tokenize(self, texts):
        return self.tokenizer(texts, truncation=False, add_special_tokens=False)['input_ids']

    def tokenize_line(self, line, *, target_token_count=None):
        if shared.opts.emphasis != 'None':
            parsed = prompt_parser.parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]
        tokenized = self.tokenize([text for text, _ in parsed])
        tokens = []
        multipliers = []
        for text_tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                continue
            tokens += text_tokens
            multipliers += [weight] * len(text_tokens)
        tokens += [self.id_end]
        multipliers += [1.0]
        if target_token_count is not None:
            if len(tokens) < target_token_count:
                tokens += [self.id_pad] * (target_token_count - len(tokens))
                multipliers += [1.0] * (target_token_count - len(tokens))
            else:
                tokens = tokens[0:target_token_count]
                multipliers = multipliers[0:target_token_count]
        return tokens, multipliers

    def forward(self, texts, *, token_count):
        if not self.t5xxl or not shared.opts.sd3_enable_t5:
            return torch.zeros((len(texts), token_count, 4096), device=devices.device, dtype=devices.dtype)
        tokens_batch = []
        for text in texts:
            tokens, multipliers = self.tokenize_line(text, target_token_count=token_count)
            tokens_batch.append(tokens)
        t5_out, t5_pooled = self.t5xxl(tokens_batch)
        return t5_out

    def encode_embedding_init_text(self, init_text, nvpt):
        return torch.zeros((nvpt, 4096), device=devices.device)


CLIPG_CONFIG = {'hidden_act': 'gelu', 'hidden_size': 1280, 'intermediate_size': 5120, 'num_attention_heads': 20, 'num_hidden_layers': 32, 'textual_inversion_key': 'clip_g'}


CLIPG_URL = 'https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/clip_g.safetensors'


CLIPL_CONFIG = {'hidden_act': 'quick_gelu', 'hidden_size': 768, 'intermediate_size': 3072, 'num_attention_heads': 12, 'num_hidden_layers': 12}


CLIPL_URL = 'https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/clip_l.safetensors'


class SDTokenizer:

    def __init__(self, max_length=77, pad_with_end=True, tokenizer=None, has_start_token=True, pad_to_max_length=True, min_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        empty = self.tokenizer('')['input_ids']
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8

    def tokenize_with_weights(self, text: 'str'):
        """Tokenize the text, with weight values - presume 1.0 for all and ignore other features here. The details aren't relevant for a reference impl, and weights themselves has weak effect on SD3."""
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        to_tokenize = text.replace('\n', ' ').split(' ')
        to_tokenize = [x for x in to_tokenize if x != '']
        for word in to_tokenize:
            batch.extend([(t, 1) for t in self.tokenizer(word)['input_ids'][self.tokens_start:-1]])
        batch.append((self.end_token, 1.0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0)] * (self.min_length - len(batch)))
        return [batch]


class SDXLClipGTokenizer(SDTokenizer):

    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)


class T5XXLTokenizer(SDTokenizer):
    """Wraps the T5 Tokenizer from HF into the SDTokenizer interface"""

    def __init__(self):
        super().__init__(pad_with_end=False, tokenizer=T5TokenizerFast.from_pretrained('google/t5-v1_1-xxl'), has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=77)


class SD3Tokenizer:

    def __init__(self):
        clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        self.t5xxl = T5XXLTokenizer()

    def tokenize_with_weights(self, text: 'str'):
        out = {}
        out['g'] = self.clip_g.tokenize_with_weights(text)
        out['l'] = self.clip_l.tokenize_with_weights(text)
        out['t5xxl'] = self.t5xxl.tokenize_with_weights(text)
        return out


class SafetensorsMapping(typing.Mapping):

    def __init__(self, file):
        self.file = file

    def __len__(self):
        return len(self.file.keys())

    def __iter__(self):
        for key in self.file.keys():
            yield key

    def __getitem__(self, key):
        return self.file.get_tensor(key)


T5_CONFIG = {'d_ff': 10240, 'd_model': 4096, 'num_heads': 64, 'num_layers': 24, 'vocab_size': 32128}


T5_URL = 'https://huggingface.co/AUTOMATIC/stable-diffusion-3-medium-text-encoders/resolve/main/t5xxl_fp16.safetensors'


class SD3Cond(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = SD3Tokenizer()
        with torch.no_grad():
            self.clip_g = SDXLClipG(CLIPG_CONFIG, device='cpu', dtype=devices.dtype)
            self.clip_l = SDClipModel(layer='hidden', layer_idx=-2, device='cpu', dtype=devices.dtype, layer_norm_hidden_state=False, return_projected_pooled=False, textmodel_json_config=CLIPL_CONFIG)
            if shared.opts.sd3_enable_t5:
                self.t5xxl = T5XXLModel(T5_CONFIG, device='cpu', dtype=devices.dtype)
            else:
                self.t5xxl = None
            self.model_lg = Sd3ClipLG(self.clip_l, self.clip_g)
            self.model_t5 = Sd3T5(self.t5xxl)

    def forward(self, prompts: 'list[str]'):
        with devices.without_autocast():
            lg_out, vector_out = self.model_lg(prompts)
            t5_out = self.model_t5(prompts, token_count=lg_out.shape[1])
            lgt_out = torch.cat([lg_out, t5_out], dim=-2)
        return {'crossattn': lgt_out, 'vector': vector_out}

    def before_load_weights(self, state_dict):
        clip_path = os.path.join(shared.models_path, 'CLIP')
        if 'text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight' not in state_dict:
            clip_g_file = modelloader.load_file_from_url(CLIPG_URL, model_dir=clip_path, file_name='clip_g.safetensors')
            with safetensors.safe_open(clip_g_file, framework='pt') as file:
                self.clip_g.transformer.load_state_dict(SafetensorsMapping(file))
        if 'text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight' not in state_dict:
            clip_l_file = modelloader.load_file_from_url(CLIPL_URL, model_dir=clip_path, file_name='clip_l.safetensors')
            with safetensors.safe_open(clip_l_file, framework='pt') as file:
                self.clip_l.transformer.load_state_dict(SafetensorsMapping(file), strict=False)
        if self.t5xxl and 'text_encoders.t5xxl.transformer.encoder.embed_tokens.weight' not in state_dict:
            t5_file = modelloader.load_file_from_url(T5_URL, model_dir=clip_path, file_name='t5xxl_fp16.safetensors')
            with safetensors.safe_open(t5_file, framework='pt') as file:
                self.t5xxl.transformer.load_state_dict(SafetensorsMapping(file), strict=False)

    def encode_embedding_init_text(self, init_text, nvpt):
        return self.model_lg.encode_embedding_init_text(init_text, nvpt)

    def tokenize(self, texts):
        return self.model_lg.tokenize(texts)

    def medvram_modules(self):
        return [self.clip_g, self.clip_l, self.t5xxl]

    def get_token_count(self, text):
        _, token_count = self.model_lg.process_texts([text])
        return token_count

    def get_target_prompt_token_count(self, token_count):
        return self.model_lg.get_target_prompt_token_count(token_count)


class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: 'torch.Tensor'):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image


class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model"""

    def __init__(self, shift=1.0, device=None, dtype=torch.float32, state_dict=None, prefix=''):
        super().__init__()
        patch_size = state_dict[f'{prefix}x_embedder.proj.weight'].shape[2]
        depth = state_dict[f'{prefix}x_embedder.proj.weight'].shape[0] // 64
        num_patches = state_dict[f'{prefix}pos_embed'].shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = state_dict[f'{prefix}y_embedder.mlp.0.weight'].shape[1]
        context_shape = state_dict[f'{prefix}context_embedder.weight'].shape
        context_embedder_config = {'target': 'torch.nn.Linear', 'params': {'in_features': context_shape[1], 'out_features': context_shape[0]}}
        self.diffusion_model = MMDiT(input_size=None, pos_embed_scaling_factor=None, pos_embed_offset=None, pos_embed_max_size=pos_embed_max_size, patch_size=patch_size, in_channels=16, depth=depth, num_patches=num_patches, adm_in_channels=adm_in_channels, context_embedder_config=context_embedder_config, device=device, dtype=dtype)
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)
        self.depth = depth

    def apply_model(self, x, sigma, c_crossattn=None, y=None):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        model_output = self.diffusion_model(x, timestep, context=c_crossattn, y=y).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

    def get_dtype(self):
        return self.diffusion_model.dtype


class AfterCFGCallbackParams:

    def __init__(self, x, sampling_step, total_sampling_steps):
        self.x = x
        """Latent image representation in the process of being denoised"""
        self.sampling_step = sampling_step
        """Current Sampling step number"""
        self.total_sampling_steps = total_sampling_steps
        """Total number of sampling steps planned"""


class CFGDenoisedParams:

    def __init__(self, x, sampling_step, total_sampling_steps, inner_model):
        self.x = x
        """Latent image representation in the process of being denoised"""
        self.sampling_step = sampling_step
        """Current Sampling step number"""
        self.total_sampling_steps = total_sampling_steps
        """Total number of sampling steps planned"""
        self.inner_model = inner_model
        """Inner model reference used for denoising"""


class CFGDenoiserParams:

    def __init__(self, x, image_cond, sigma, sampling_step, total_sampling_steps, text_cond, text_uncond, denoiser=None):
        self.x = x
        """Latent image representation in the process of being denoised"""
        self.image_cond = image_cond
        """Conditioning image"""
        self.sigma = sigma
        """Current sigma noise step value"""
        self.sampling_step = sampling_step
        """Current Sampling step number"""
        self.total_sampling_steps = total_sampling_steps
        """Total number of sampling steps planned"""
        self.text_cond = text_cond
        """ Encoder hidden states of text conditioning from prompt"""
        self.text_uncond = text_uncond
        """ Encoder hidden states of text conditioning from negative prompt"""
        self.denoiser = denoiser
        """Current CFGDenoiser object with processing parameters"""


def catenate_conds(conds):
    if not isinstance(conds[0], dict):
        return torch.cat(conds)
    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}


callback_map = dict(callbacks_app_started=[], callbacks_model_loaded=[], callbacks_ui_tabs=[], callbacks_ui_train_tabs=[], callbacks_ui_settings=[], callbacks_before_image_saved=[], callbacks_image_saved=[], callbacks_extra_noise=[], callbacks_cfg_denoiser=[], callbacks_cfg_denoised=[], callbacks_cfg_after_cfg=[], callbacks_before_component=[], callbacks_after_component=[], callbacks_image_grid=[], callbacks_infotext_pasted=[], callbacks_script_unloaded=[], callbacks_before_ui=[], callbacks_on_reload=[], callbacks_list_optimizers=[], callbacks_list_unets=[], callbacks_before_token_counter=[])


ordered_callbacks_map = {}


def sort_callbacks(category, unordered_callbacks, *, enable_user_sort=True):
    callbacks = unordered_callbacks.copy()
    callback_lookup = {x.name: x for x in callbacks}
    dependencies = {}
    order_instructions = {}
    for extension in extensions.extensions:
        for order_instruction in extension.metadata.list_callback_order_instructions():
            if order_instruction.name in callback_lookup:
                if order_instruction.name not in order_instructions:
                    order_instructions[order_instruction.name] = []
                order_instructions[order_instruction.name].append(order_instruction)
    if order_instructions:
        for callback in callbacks:
            dependencies[callback.name] = []
        for callback in callbacks:
            for order_instruction in order_instructions.get(callback.name, []):
                for after in order_instruction.after:
                    if after not in callback_lookup:
                        continue
                    dependencies[callback.name].append(after)
                for before in order_instruction.before:
                    if before not in callback_lookup:
                        continue
                    dependencies[before].append(callback.name)
        sorted_names = util.topological_sort(dependencies)
        callbacks = [callback_lookup[x] for x in sorted_names]
    if enable_user_sort:
        for name in reversed(getattr(shared.opts, 'prioritized_callbacks_' + category, [])):
            index = next((i for i, callback in enumerate(callbacks) if callback.name == name), None)
            if index is not None:
                callbacks.insert(0, callbacks.pop(index))
    return callbacks


def ordered_callbacks(category, unordered_callbacks=None, *, enable_user_sort=True):
    if unordered_callbacks is None:
        unordered_callbacks = callback_map.get('callbacks_' + category, [])
    if not enable_user_sort:
        return sort_callbacks(category, unordered_callbacks, enable_user_sort=False)
    callbacks = ordered_callbacks_map.get(category)
    if callbacks is not None and len(callbacks) == len(unordered_callbacks):
        return callbacks
    callbacks = sort_callbacks(category, unordered_callbacks)
    ordered_callbacks_map[category] = callbacks
    return callbacks


def report_exception(c, job):
    errors.report(f'Error executing callback {job} for {c.script}', exc_info=True)


def cfg_after_cfg_callback(params: 'AfterCFGCallbackParams'):
    for c in ordered_callbacks('cfg_after_cfg'):
        try:
            c.callback(params)
        except Exception:
            report_exception(c, 'cfg_after_cfg_callback')


def cfg_denoised_callback(params: 'CFGDenoisedParams'):
    for c in ordered_callbacks('cfg_denoised'):
        try:
            c.callback(params)
        except Exception:
            report_exception(c, 'cfg_denoised_callback')


def cfg_denoiser_callback(params: 'CFGDenoiserParams'):
    for c in ordered_callbacks('cfg_denoiser'):
        try:
            c.callback(params)
        except Exception:
            report_exception(c, 'cfg_denoiser_callback')


def pad_cond(tensor, repeats, empty):
    if not isinstance(tensor, dict):
        return torch.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1))], axis=1)
    tensor['crossattn'] = pad_cond(tensor['crossattn'], repeats, empty)
    return tensor


def subscript_cond(cond, a, b):
    if not isinstance(cond, dict):
        return cond[a:b]
    return {key: vec[a:b] for key, vec in cond.items()}


class CFGDenoiser(torch.nn.Module):
    """
    Classifier free guidance denoiser. A wrapper for stable diffusion model (specifically for unet)
    that can take a noisy picture and produce a noise-free picture using two guidances (prompts)
    instead of one. Originally, the second prompt is just an empty string, but we use non-empty
    negative prompt.
    """

    def __init__(self, sampler):
        super().__init__()
        self.model_wrap = None
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.steps = None
        """number of steps as specified by user in UI"""
        self.total_steps = None
        """expected number of calls to denoiser calculated from self.steps and specifics of the selected sampler"""
        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        self.sampler = sampler
        self.model_wrap = None
        self.p = None
        self.cond_scale_miltiplier = 1.0
        self.need_last_noise_uncond = False
        self.last_noise_uncond = None
        self.mask_before_denoising = False

    @property
    def inner_model(self):
        raise NotImplementedError()

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0]:]
        denoised = torch.clone(denoised_uncond)
        for i, conds in enumerate(conds_list):
            for cond_index, weight in conds:
                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)
        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        denoised = out_uncond + cond_scale * (out_cond - out_img_cond) + self.image_cfg_scale * (out_img_cond - out_uncond)
        return denoised

    def get_pred_x0(self, x_in, x_out, sigma):
        return x_out

    def update_inner_model(self):
        self.model_wrap = None
        c, uc = self.p.get_conds()
        self.sampler.sampler_extra_args['cond'] = c
        self.sampler.sampler_extra_args['uncond'] = uc

    def pad_cond_uncond(self, cond, uncond):
        empty = shared.sd_model.cond_stage_model_empty_prompt
        num_repeats = (cond.shape[1] - uncond.shape[1]) // empty.shape[1]
        if num_repeats < 0:
            cond = pad_cond(cond, -num_repeats, empty)
            self.padded_cond_uncond = True
        elif num_repeats > 0:
            uncond = pad_cond(uncond, num_repeats, empty)
            self.padded_cond_uncond = True
        return cond, uncond

    def pad_cond_uncond_v0(self, cond, uncond):
        """
        Pads the 'uncond' tensor to match the shape of the 'cond' tensor.

        If 'uncond' is a dictionary, it is assumed that the 'crossattn' key holds the tensor to be padded.
        If 'uncond' is a tensor, it is padded directly.

        If the number of columns in 'uncond' is less than the number of columns in 'cond', the last column of 'uncond'
        is repeated to match the number of columns in 'cond'.

        If the number of columns in 'uncond' is greater than the number of columns in 'cond', 'uncond' is truncated
        to match the number of columns in 'cond'.

        Args:
            cond (torch.Tensor or DictWithShape): The condition tensor to match the shape of 'uncond'.
            uncond (torch.Tensor or DictWithShape): The tensor to be padded, or a dictionary containing the tensor to be padded.

        Returns:
            tuple: A tuple containing the 'cond' tensor and the padded 'uncond' tensor.

        Note:
            This is the padding that was always used in DDIM before version 1.6.0
        """
        is_dict_cond = isinstance(uncond, dict)
        uncond_vec = uncond['crossattn'] if is_dict_cond else uncond
        if uncond_vec.shape[1] < cond.shape[1]:
            last_vector = uncond_vec[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - uncond_vec.shape[1], 1])
            uncond_vec = torch.hstack([uncond_vec, last_vector_repeated])
            self.padded_cond_uncond_v0 = True
        elif uncond_vec.shape[1] > cond.shape[1]:
            uncond_vec = uncond_vec[:, :cond.shape[1]]
            self.padded_cond_uncond_v0 = True
        if is_dict_cond:
            uncond['crossattn'] = uncond_vec
        else:
            uncond = uncond_vec
        return cond, uncond

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        if state.interrupted or state.skipped:
            raise sd_samplers_common.InterruptedException
        if sd_samplers_common.apply_refiner(self, sigma):
            cond = self.sampler.sampler_extra_args['cond']
            uncond = self.sampler.sampler_extra_args['uncond']
        is_edit_model = shared.sd_model.cond_stage_key == 'edit' and self.image_cfg_scale is not None and self.image_cfg_scale != 1.0
        conds_list, tensor = prompt_parser.reconstruct_multicond_batch(cond, self.step)
        uncond = prompt_parser.reconstruct_cond_batch(uncond, self.step)
        assert not is_edit_model or all(len(conds) == 1 for conds in conds_list), 'AND is not supported for InstructPix2Pix checkpoint (unless using Image CFG scale = 1.0)'

        def apply_blend(current_latent):
            blended_latent = current_latent * self.nmask + self.init_latent * self.mask
            if self.p.scripts is not None:
                mba = scripts.MaskBlendArgs(current_latent, self.nmask, self.init_latent, self.mask, blended_latent, denoiser=self, sigma=sigma)
                self.p.scripts.on_mask_blend(self.p, mba)
                blended_latent = mba.blended_latent
            return blended_latent
        if self.mask_before_denoising and self.mask is not None:
            x = apply_blend(x)
        batch_size = len(conds_list)
        repeats = [len(conds_list[i]) for i in range(batch_size)]
        if shared.sd_model.model.conditioning_key == 'crossattn-adm':
            image_uncond = torch.zeros_like(image_cond)
            make_condition_dict = lambda c_crossattn, c_adm: {'c_crossattn': [c_crossattn], 'c_adm': c_adm}
        else:
            image_uncond = image_cond
            if isinstance(uncond, dict):
                make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, 'c_concat': [c_concat]}
            else:
                make_condition_dict = lambda c_crossattn, c_concat: {'c_crossattn': [c_crossattn], 'c_concat': [c_concat]}
        if not is_edit_model:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond])
        else:
            x_in = torch.cat([torch.stack([x[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [x] + [x])
            sigma_in = torch.cat([torch.stack([sigma[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [sigma] + [sigma])
            image_cond_in = torch.cat([torch.stack([image_cond[i] for _ in range(n)]) for i, n in enumerate(repeats)] + [image_uncond] + [torch.zeros_like(self.init_latent)])
        denoiser_params = CFGDenoiserParams(x_in, image_cond_in, sigma_in, state.sampling_step, state.sampling_steps, tensor, uncond, self)
        cfg_denoiser_callback(denoiser_params)
        x_in = denoiser_params.x
        image_cond_in = denoiser_params.image_cond
        sigma_in = denoiser_params.sigma
        tensor = denoiser_params.text_cond
        uncond = denoiser_params.text_uncond
        skip_uncond = False
        if shared.opts.skip_early_cond != 0.0 and self.step / self.total_steps <= shared.opts.skip_early_cond:
            skip_uncond = True
            self.p.extra_generation_params['Skip Early CFG'] = shared.opts.skip_early_cond
        elif (self.step % 2 or shared.opts.s_min_uncond_all) and s_min_uncond > 0 and sigma[0] < s_min_uncond and not is_edit_model:
            skip_uncond = True
            self.p.extra_generation_params['NGMS'] = s_min_uncond
            if shared.opts.s_min_uncond_all:
                self.p.extra_generation_params['NGMS all steps'] = shared.opts.s_min_uncond_all
        if skip_uncond:
            x_in = x_in[:-batch_size]
            sigma_in = sigma_in[:-batch_size]
        self.padded_cond_uncond = False
        self.padded_cond_uncond_v0 = False
        if shared.opts.pad_cond_uncond_v0 and tensor.shape[1] != uncond.shape[1]:
            tensor, uncond = self.pad_cond_uncond_v0(tensor, uncond)
        elif shared.opts.pad_cond_uncond and tensor.shape[1] != uncond.shape[1]:
            tensor, uncond = self.pad_cond_uncond(tensor, uncond)
        if tensor.shape[1] == uncond.shape[1] or skip_uncond:
            if is_edit_model:
                cond_in = catenate_conds([tensor, uncond, uncond])
            elif skip_uncond:
                cond_in = tensor
            else:
                cond_in = catenate_conds([tensor, uncond])
            if shared.opts.batch_cond_uncond:
                x_out = self.inner_model(x_in, sigma_in, cond=make_condition_dict(cond_in, image_cond_in))
            else:
                x_out = torch.zeros_like(x_in)
                for batch_offset in range(0, x_out.shape[0], batch_size):
                    a = batch_offset
                    b = a + batch_size
                    x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=make_condition_dict(subscript_cond(cond_in, a, b), image_cond_in[a:b]))
        else:
            x_out = torch.zeros_like(x_in)
            batch_size = batch_size * 2 if shared.opts.batch_cond_uncond else batch_size
            for batch_offset in range(0, tensor.shape[0], batch_size):
                a = batch_offset
                b = min(a + batch_size, tensor.shape[0])
                if not is_edit_model:
                    c_crossattn = subscript_cond(tensor, a, b)
                else:
                    c_crossattn = torch.cat([tensor[a:b]], uncond)
                x_out[a:b] = self.inner_model(x_in[a:b], sigma_in[a:b], cond=make_condition_dict(c_crossattn, image_cond_in[a:b]))
            if not skip_uncond:
                x_out[-uncond.shape[0]:] = self.inner_model(x_in[-uncond.shape[0]:], sigma_in[-uncond.shape[0]:], cond=make_condition_dict(uncond, image_cond_in[-uncond.shape[0]:]))
        denoised_image_indexes = [x[0][0] for x in conds_list]
        if skip_uncond:
            fake_uncond = torch.cat([x_out[i:i + 1] for i in denoised_image_indexes])
            x_out = torch.cat([x_out, fake_uncond])
        denoised_params = CFGDenoisedParams(x_out, state.sampling_step, state.sampling_steps, self.inner_model)
        cfg_denoised_callback(denoised_params)
        if self.need_last_noise_uncond:
            self.last_noise_uncond = torch.clone(x_out[-uncond.shape[0]:])
        if is_edit_model:
            denoised = self.combine_denoised_for_edit_model(x_out, cond_scale * self.cond_scale_miltiplier)
        elif skip_uncond:
            denoised = self.combine_denoised(x_out, conds_list, uncond, 1.0)
        else:
            denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale * self.cond_scale_miltiplier)
        if not self.mask_before_denoising and self.mask is not None:
            denoised = apply_blend(denoised)
        self.sampler.last_latent = self.get_pred_x0(torch.cat([x_in[i:i + 1] for i in denoised_image_indexes]), torch.cat([x_out[i:i + 1] for i in denoised_image_indexes]), sigma)
        if opts.live_preview_content == 'Prompt':
            preview = self.sampler.last_latent
        elif opts.live_preview_content == 'Negative prompt':
            preview = self.get_pred_x0(x_in[-uncond.shape[0]:], x_out[-uncond.shape[0]:], sigma)
        else:
            preview = self.get_pred_x0(torch.cat([x_in[i:i + 1] for i in denoised_image_indexes]), torch.cat([denoised[i:i + 1] for i in denoised_image_indexes]), sigma)
        sd_samplers_common.store_latent(preview)
        after_cfg_callback_params = AfterCFGCallbackParams(denoised, state.sampling_step, state.sampling_steps)
        cfg_after_cfg_callback(after_cfg_callback_params)
        denoised = after_cfg_callback_params.x
        self.step += 1
        return denoised


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-06, affine=True, dtype=dtype, device=device)


class ResnetBlock(torch.nn.Module):

    def __init__(self, *, in_channels, out_channels=None, dtype=torch.float32, device=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):

    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = [einops.rearrange(x, 'b c h w -> b 1 (h w) c').contiguous() for x in (q, k, v)]
        hidden = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        hidden = einops.rearrange(hidden, 'b 1 (h w) c -> b c h w', h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Downsample(torch.nn.Module):

    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0, dtype=dtype, device=device)

    def forward(self, x):
        pad = 0, 1, 0, 1
        x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
        x = self.conv(x)
        return x


class Upsample(torch.nn.Module):

    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv(x)
        return x


class VAEEncoder(torch.nn.Module):

    def __init__(self, ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2, in_channels=3, z_channels=16, dtype=torch.float32, device=None):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = torch.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, dtype=dtype, device=device)
            self.down.append(down)
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(torch.nn.Module):

    def __init__(self, ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2, resolution=256, z_channels=16, dtype=torch.float32, device=None):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dtype=dtype, device=device)
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dtype=dtype, device=device))
                block_in = block_out
            up = torch.nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, dtype=dtype, device=device)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, z):
        hidden = self.conv_in(z)
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        hidden = self.norm_out(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden


class SDVAE(torch.nn.Module):

    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.encoder = VAEEncoder(dtype=dtype, device=device)
        self.decoder = VAEDecoder(dtype=dtype, device=device)

    @torch.autocast('cuda', dtype=torch.float16)
    def decode(self, latent):
        return self.decoder(latent)

    @torch.autocast('cuda', dtype=torch.float16)
    def encode(self, image):
        hidden = self.encoder(image)
        mean, logvar = torch.chunk(hidden, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)


class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return latent / self.scale_factor + self.shift_factor

    def decode_latent_to_preview(self, x0):
        """Quick RGB approximate preview of sd3 latents"""
        factors = torch.tensor([[-0.0645, 0.0177, 0.1052], [0.0028, 0.0312, 0.065], [0.1848, 0.0762, 0.036], [0.0944, 0.036, 0.0889], [0.0897, 0.0506, -0.0364], [-0.002, 0.1203, 0.0284], [0.0855, 0.0118, 0.0283], [-0.0539, 0.0658, 0.1047], [-0.0057, 0.0116, 0.07], [-0.0412, 0.0281, -0.0039], [0.1106, 0.1171, 0.122], [-0.0248, 0.0682, -0.0481], [0.0815, 0.0846, 0.1207], [-0.012, -0.0055, -0.0867], [-0.0749, -0.0634, -0.0456], [-0.1418, -0.1457, -0.1259]], device='cpu')
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors
        latents_ubyte = ((latent_image + 1) / 2).clamp(0, 1).mul(255).byte().cpu()
        return Image.fromarray(latents_ubyte.numpy())


class SD3Inferencer(torch.nn.Module):

    def __init__(self, state_dict, shift=3, use_ema=False):
        super().__init__()
        self.shift = shift
        with torch.no_grad():
            self.model = BaseModel(shift=shift, state_dict=state_dict, prefix='model.diffusion_model.', device='cpu', dtype=devices.dtype)
            self.first_stage_model = SDVAE(device='cpu', dtype=devices.dtype_vae)
            self.first_stage_model.dtype = self.model.diffusion_model.dtype
        self.alphas_cumprod = 1 / (self.model.model_sampling.sigmas ** 2 + 1)
        self.text_encoders = SD3Cond()
        self.cond_stage_key = 'txt'
        self.parameterization = 'eps'
        self.model.conditioning_key = 'crossattn'
        self.latent_format = SD3LatentFormat()
        self.latent_channels = 16

    @property
    def cond_stage_model(self):
        return self.text_encoders

    def before_load_weights(self, state_dict):
        self.cond_stage_model.before_load_weights(state_dict)

    def ema_scope(self):
        return contextlib.nullcontext()

    def get_learned_conditioning(self, batch: 'list[str]'):
        return self.cond_stage_model(batch)

    def apply_model(self, x, t, cond):
        return self.model(x, t, c_crossattn=cond['crossattn'], y=cond['vector'])

    def decode_first_stage(self, latent):
        latent = self.latent_format.process_out(latent)
        return self.first_stage_model.decode(latent)

    def encode_first_stage(self, image):
        latent = self.first_stage_model.encode(image)
        return self.latent_format.process_in(latent)

    def get_first_stage_encoding(self, x):
        return x

    def create_denoiser(self):
        return SD3Denoiser(self, self.model.model_sampling.sigmas)

    def medvram_fields(self):
        return [(self, 'first_stage_model'), (self, 'text_encoders'), (self, 'model')]

    def add_noise_to_latent(self, x, noise, amount):
        return x * (1 - amount) + noise * amount

    def fix_dimensions(self, width, height):
        return width // 16 * 16, height // 16 * 16

    def diffusers_weight_mapping(self):
        for i in range(self.model.depth):
            yield f'transformer.transformer_blocks.{i}.attn.to_q', f'diffusion_model_joint_blocks_{i}_x_block_attn_qkv_q_proj'
            yield f'transformer.transformer_blocks.{i}.attn.to_k', f'diffusion_model_joint_blocks_{i}_x_block_attn_qkv_k_proj'
            yield f'transformer.transformer_blocks.{i}.attn.to_v', f'diffusion_model_joint_blocks_{i}_x_block_attn_qkv_v_proj'
            yield f'transformer.transformer_blocks.{i}.attn.to_out.0', f'diffusion_model_joint_blocks_{i}_x_block_attn_proj'
            yield f'transformer.transformer_blocks.{i}.attn.add_q_proj', f'diffusion_model_joint_blocks_{i}_context_block.attn_qkv_q_proj'
            yield f'transformer.transformer_blocks.{i}.attn.add_k_proj', f'diffusion_model_joint_blocks_{i}_context_block.attn_qkv_k_proj'
            yield f'transformer.transformer_blocks.{i}.attn.add_v_proj', f'diffusion_model_joint_blocks_{i}_context_block.attn_qkv_v_proj'
            yield f'transformer.transformer_blocks.{i}.attn.add_out_proj.0', f'diffusion_model_joint_blocks_{i}_context_block_attn_proj'


class EmbeddingsWithFixes(torch.nn.Module):

    def __init__(self, wrapped, embeddings, textual_inversion_key='clip_l'):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings
        self.textual_inversion_key = textual_inversion_key

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None
        inputs_embeds = self.wrapped(input_ids)
        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds
        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                vec = embedding.vec[self.textual_inversion_key] if isinstance(embedding.vec, dict) else embedding.vec
                emb = devices.cond_cast_unet(vec)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])
            vecs.append(tensor)
        return torch.stack(vecs)


def weighted_loss(sd_model, pred, target, mean=True):
    loss = sd_model._old_get_loss(pred, target, mean=False)
    weight = getattr(sd_model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight
    return loss.mean() if mean else loss


def weighted_forward(sd_model, x, c, w, *args, **kwargs):
    try:
        sd_model._custom_loss_weight = w
        if not hasattr(sd_model, '_old_get_loss'):
            sd_model._old_get_loss = sd_model.get_loss
        sd_model.get_loss = MethodType(weighted_loss, sd_model)
        return sd_model.forward(x, c, *args, **kwargs)
    finally:
        try:
            del sd_model._custom_loss_weight
        except AttributeError:
            pass
        if hasattr(sd_model, '_old_get_loss'):
            sd_model.get_loss = sd_model._old_get_loss
            del sd_model._old_get_loss


def apply_weighted_forward(sd_model):
    sd_model.weighted_forward = MethodType(weighted_forward, sd_model)


def undo_optimizations():
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward
    sgm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    sgm.modules.attention.CrossAttention.forward = hypernetwork.attention_CrossAttention_forward
    sgm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def undo_weighted_forward(sd_model):
    try:
        del sd_model.weighted_forward
    except AttributeError:
        pass


class StableDiffusionModelHijack:
    fixes = None
    layers = None
    circular_enabled = False
    clip = None
    optimization_method = None

    def __init__(self):
        self.extra_generation_params = {}
        self.comments = []
        self.embedding_db = modules.textual_inversion.textual_inversion.EmbeddingDatabase()
        self.embedding_db.add_embedding_dir(cmd_opts.embeddings_dir)

    def apply_optimizations(self, option=None):
        try:
            self.optimization_method = apply_optimizations(option)
        except Exception as e:
            errors.display(e, 'applying cross attention optimization')
            undo_optimizations()

    def convert_sdxl_to_ssd(self, m):
        """Converts an SDXL model to a Segmind Stable Diffusion model (see https://huggingface.co/segmind/SSD-1B)"""
        delattr(m.model.diffusion_model.middle_block, '1')
        delattr(m.model.diffusion_model.middle_block, '2')
        for i in ['9', '8', '7', '6', '5', '4']:
            delattr(m.model.diffusion_model.input_blocks[7][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.input_blocks[8][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.output_blocks[0][1].transformer_blocks, i)
            delattr(m.model.diffusion_model.output_blocks[1][1].transformer_blocks, i)
        delattr(m.model.diffusion_model.output_blocks[4][1].transformer_blocks, '1')
        delattr(m.model.diffusion_model.output_blocks[5][1].transformer_blocks, '1')
        devices.torch_gc()

    def hijack(self, m):
        conditioner = getattr(m, 'conditioner', None)
        if conditioner:
            text_cond_models = []
            for i in range(len(conditioner.embedders)):
                embedder = conditioner.embedders[i]
                typename = type(embedder).__name__
                if typename == 'FrozenOpenCLIPEmbedder':
                    embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self)
                    conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
                if typename == 'FrozenCLIPEmbedder':
                    model_embeddings = embedder.transformer.text_model.embeddings
                    model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
                    conditioner.embedders[i] = sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
                if typename == 'FrozenOpenCLIPEmbedder2':
                    embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self, textual_inversion_key='clip_g')
                    conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords(embedder, self)
                    text_cond_models.append(conditioner.embedders[i])
            if len(text_cond_models) == 1:
                m.cond_stage_model = text_cond_models[0]
            else:
                m.cond_stage_model = conditioner
        if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation or type(m.cond_stage_model) == xlmr_m18.BertSeriesModelWithTransformation:
            model_embeddings = m.cond_stage_model.roberta.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.word_embeddings, self)
            m.cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(m.cond_stage_model, self)
        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            m.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)
        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
            m.cond_stage_model.model.token_embedding = EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding, self)
            m.cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)
        apply_weighted_forward(m)
        if m.cond_stage_key == 'edit':
            sd_hijack_unet.hijack_ddpm_edit()
        self.apply_optimizations()
        self.clip = m.cond_stage_model

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res
        self.layers = flatten(m)
        if isinstance(m, ldm.models.diffusion.ddpm.LatentDiffusion):
            sd_unet.original_forward = ldm_original_forward
        elif isinstance(m, modules.models.diffusion.ddpm_edit.LatentDiffusion):
            sd_unet.original_forward = ldm_original_forward
        elif isinstance(m, sgm.models.diffusion.DiffusionEngine):
            sd_unet.original_forward = sgm_original_forward
        else:
            sd_unet.original_forward = None

    def undo_hijack(self, m):
        conditioner = getattr(m, 'conditioner', None)
        if conditioner:
            for i in range(len(conditioner.embedders)):
                embedder = conditioner.embedders[i]
                if isinstance(embedder, (sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords, sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords)):
                    embedder.wrapped.model.token_embedding = embedder.wrapped.model.token_embedding.wrapped
                    conditioner.embedders[i] = embedder.wrapped
                if isinstance(embedder, sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords):
                    embedder.wrapped.transformer.text_model.embeddings.token_embedding = embedder.wrapped.transformer.text_model.embeddings.token_embedding.wrapped
                    conditioner.embedders[i] = embedder.wrapped
            if hasattr(m, 'cond_stage_model'):
                delattr(m, 'cond_stage_model')
        elif type(m.cond_stage_model) == sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped
        elif type(m.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        elif type(m.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
            m.cond_stage_model = m.cond_stage_model.wrapped
        undo_optimizations()
        undo_weighted_forward(m)
        self.apply_circular(False)
        self.layers = None
        self.clip = None

    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return
        self.circular_enabled = enable
        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def clear_comments(self):
        self.comments = []
        self.extra_generation_params = {}

    def get_prompt_lengths(self, text):
        if self.clip is None:
            return '-', '-'
        if hasattr(self.clip, 'get_token_count'):
            token_count = self.clip.get_token_count(text)
        else:
            _, token_count = self.clip.process_texts([text])
        return token_count, self.clip.get_target_prompt_token_count(token_count)

    def redo_hijack(self, m):
        self.undo_hijack(m)
        self.hijack(m)


class TextualInversionEmbeddings(torch.nn.Embedding):

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', textual_inversion_key='clip_l', **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.embeddings = model_hijack
        self.textual_inversion_key = textual_inversion_key

    @property
    def wrapped(self):
        return super().forward

    def forward(self, input_ids):
        return EmbeddingsWithFixes.forward(self, input_ids)


class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])


class TextConditionalModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.hijack = sd_hijack.model_hijack
        self.chunk_length = 75
        self.is_trainable = False
        self.input_key = 'txt'
        self.return_pooled = False
        self.comma_token = None
        self.id_start = None
        self.id_end = None
        self.id_pad = None

    def empty_chunk(self):
        """creates an empty PromptChunk and returns it"""
        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        """returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented"""
        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        """Converts a batch of texts into a batch of token ids"""
        raise NotImplementedError

    def encode_with_transformers(self, tokens):
        """
        converts a batch of token ids (in python lists) into a single tensor with numeric representation of those tokens;
        All python lists with tokens are assumed to have same length, usually 77.
        if input is a list with B elements and each element has T tokens, expected output shape is (B, T, C), where C depends on
        model - can be 768 and 1024.
        Among other things, this call will read self.hijack.fixes, apply it to its inputs, and clear it (setting it to None).
        """
        raise NotImplementedError

    def encode_embedding_init_text(self, init_text, nvpt):
        """Converts text into a tensor with this text's tokens' embeddings. Note that those are embeddings before they are passed through
        transformers. nvpt is used as a maximum length in tokens. If text produces less teokens than nvpt, only this many is returned."""
        raise NotImplementedError

    def tokenize_line(self, line):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        """
        if opts.emphasis != 'None':
            parsed = prompt_parser.parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]
        tokenized = self.tokenize([text for text, _ in parsed])
        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk
            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length
            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add
            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]
            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()
        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue
            position = 0
            while position < len(tokens):
                token = tokens[position]
                if token == self.comma_token:
                    last_comma = len(chunk.tokens)
                elif opts.comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= opts.comma_padding_backtrack:
                    break_location = last_comma + 1
                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]
                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]
                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults
                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()
                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue
                emb_len = int(embedding.vectors)
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()
                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))
                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens
        if chunk.tokens or not chunks:
            next_chunk(is_last=True)
        return chunks, token_count

    def process_texts(self, texts):
        """
        Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum
        length, in tokens, of all texts.
        """
        token_count = 0
        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)
                cache[line] = chunks
            batch_chunks.append(chunks)
        return batch_chunks, token_count

    def forward(self, texts):
        """
        Accepts an array of texts; Passes texts through transformers network to create a tensor with numerical representation of those texts.
        Returns a tensor with shape of (B, T, C), where B is length of the array; T is length, in tokens, of texts (including padding) - T will
        be a multiple of 77; and C is dimensionality of each token - for SD1 it's 768, for SD2 it's 1024, and for SDXL it's 1280.
        An example shape returned by this function can be: (2, 77, 768).
        For SDXL, instead of returning one tensor avobe, it returns a tuple with two: the other one with shape (B, 1280) with pooled values.
        Webui usually sends just one text at a time through this function - the only time when texts is an array with more than one element
        is when you do prompt editing: "a picture of a [cat:dog:0.4] eating ice cream"
        """
        batch_chunks, token_count = self.process_texts(texts)
        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])
        zs = []
        for i in range(chunk_count):
            batch_chunk = [(chunks[i] if i < len(chunks) else self.empty_chunk()) for chunks in batch_chunks]
            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.hijack.fixes = [x.fixes for x in batch_chunk]
            for fixes in self.hijack.fixes:
                for _position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding
            devices.torch_npu_set_device()
            z = self.process_tokens(tokens, multipliers)
            zs.append(z)
        if opts.textual_inversion_add_hashes_to_infotext and used_embeddings:
            hashes = []
            for name, embedding in used_embeddings.items():
                shorthash = embedding.shorthash
                if not shorthash:
                    continue
                name = name.replace(':', '').replace(',', '')
                hashes.append(f'{name}: {shorthash}')
            if hashes:
                if self.hijack.extra_generation_params.get('TI hashes'):
                    hashes.append(self.hijack.extra_generation_params.get('TI hashes'))
                self.hijack.extra_generation_params['TI hashes'] = ', '.join(hashes)
        if any(x for x in texts if '(' in x or '[' in x) and opts.emphasis != 'Original':
            self.hijack.extra_generation_params['Emphasis'] = opts.emphasis
        if self.return_pooled:
            return torch.hstack(zs), zs[0].pooled
        else:
            return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        """
        sends one single prompt chunk to be encoded by transformers neural network.
        remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
        there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
        Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
        corresponds to one token.
        """
        tokens = torch.asarray(remade_batch_tokens)
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1:tokens.shape[1]] = self.id_pad
        z = self.encode_with_transformers(tokens)
        pooled = getattr(z, 'pooled', None)
        emphasis = sd_emphasis.get_current_option(opts.emphasis)()
        emphasis.tokens = remade_batch_tokens
        emphasis.multipliers = torch.asarray(batch_multipliers)
        emphasis.z = z
        emphasis.after_transformers()
        z = emphasis.z
        if pooled is not None:
            z.pooled = pooled
        return z


class FrozenCLIPEmbedderWithCustomWordsBase(TextConditionalModel):
    """A pytorch module that is a wrapper for FrozenCLIPEmbedder module. it enhances FrozenCLIPEmbedder, making it possible to
    have unlimited prompt length and assign weights to tokens in prompt.
    """

    def __init__(self, wrapped, hijack):
        super().__init__()
        self.hijack = hijack
        self.wrapped = wrapped
        """Original FrozenCLIPEmbedder module; can also be FrozenOpenCLIPEmbedder or xlmr.BertSeriesModelWithTransformation,
        depending on model."""
        self.is_trainable = getattr(wrapped, 'is_trainable', False)
        self.input_key = getattr(wrapped, 'input_key', 'txt')
        self.return_pooled = getattr(self.wrapped, 'return_pooled', False)
        self.legacy_ucg_val = None

    def forward(self, texts):
        if opts.use_old_emphasis_implementation:
            return modules.sd_hijack_clip_old.forward_old(self, texts)
        return super().forward(texts)


class FrozenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):

    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.tokenizer = wrapped.tokenizer
        vocab = self.tokenizer.get_vocab()
        self.comma_token = vocab.get(',</w>', None)
        self.token_mults = {}
        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1
            if mult != 1.0:
                self.token_mults[ident] = mult
        self.id_start = self.wrapped.tokenizer.bos_token_id
        self.id_end = self.wrapped.tokenizer.eos_token_id
        self.id_pad = self.id_end

    def tokenize(self, texts):
        tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)['input_ids']
        return tokenized

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=-opts.CLIP_stop_at_last_layers)
        if opts.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state
        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.transformer.text_model.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors='pt', add_special_tokens=False)['input_ids']
        embedded = embedding_layer.token_embedding.wrapped(ids).squeeze(0)
        return embedded


class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):

    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == 'hidden')
        if opts.sdxl_clip_l_skip is True:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
        elif self.wrapped.layer == 'last':
            z = outputs.last_hidden_state
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]
        return z


class FrozenOpenCLIPEmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):

    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.comma_token = [v for k, v in tokenizer.encoder.items() if k == ',</w>'][0]
        self.id_start = tokenizer.encoder['<start_of_text>']
        self.id_end = tokenizer.encoder['<end_of_text>']
        self.id_pad = 0

    def tokenize(self, texts):
        assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'
        tokenized = [tokenizer.encode(text) for text in texts]
        return tokenized

    def encode_with_transformers(self, tokens):
        z = self.wrapped.encode_with_transformer(tokens)
        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        ids = tokenizer.encode(init_text)
        ids = torch.asarray([ids], device=devices.device, dtype=torch.int)
        embedded = self.wrapped.model.token_embedding.wrapped(ids).squeeze(0)
        return embedded


class FrozenOpenCLIPEmbedder2WithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):

    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.comma_token = [v for k, v in tokenizer.encoder.items() if k == ',</w>'][0]
        self.id_start = tokenizer.encoder['<start_of_text>']
        self.id_end = tokenizer.encoder['<end_of_text>']
        self.id_pad = 0

    def tokenize(self, texts):
        assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'
        tokenized = [tokenizer.encode(text) for text in texts]
        return tokenized

    def encode_with_transformers(self, tokens):
        d = self.wrapped.encode_with_transformer(tokens)
        z = d[self.wrapped.layer]
        pooled = d.get('pooled')
        if pooled is not None:
            z.pooled = pooled
        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        ids = tokenizer.encode(init_text)
        ids = torch.asarray([ids], device=devices.device, dtype=torch.int)
        embedded = self.wrapped.model.token_embedding.wrapped(ids).squeeze(0)
        return embedded


class GELUHijack(torch.nn.GELU, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        torch.nn.GELU.__init__(self, *args, **kwargs)

    def forward(self, x):
        if devices.unet_needs_upcast:
            return torch.nn.GELU.forward(self.float(), x.float())
        else:
            return torch.nn.GELU.forward(self, x)


class FrozenXLMREmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords):

    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.id_start = wrapped.config.bos_token_id
        self.id_end = wrapped.config.eos_token_id
        self.id_pad = wrapped.config.pad_token_id
        self.comma_token = self.tokenizer.get_vocab().get(',', None)

    def encode_with_transformers(self, tokens):
        attention_mask = tokens != self.id_pad
        features = self.wrapped(input_ids=tokens, attention_mask=attention_mask)
        z = features['projection_state']
        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.roberta.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors='pt', add_special_tokens=False)['input_ids']
        embedded = embedding_layer.token_embedding.wrapped(ids).squeeze(0)
        return embedded


class CFGDenoiserKDiffusion(sd_samplers_cfg_denoiser.CFGDenoiser):

    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser_constructor = getattr(shared.sd_model, 'create_denoiser', None)
            if denoiser_constructor is not None:
                self.model_wrap = denoiser_constructor()
            else:
                denoiser = k_diffusion.external.CompVisVDenoiser if shared.sd_model.parameterization == 'v' else k_diffusion.external.CompVisDenoiser
                self.model_wrap = denoiser(shared.sd_model, quantize=shared.opts.enable_quantization)
        return self.model_wrap


class CFGDenoiserLCM(sd_samplers_cfg_denoiser.CFGDenoiser):

    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser = LCMCompVisDenoiser
            self.model_wrap = denoiser(shared.sd_model)
        return self.model_wrap


class CompVisTimestepsDenoiser(torch.nn.Module):

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def forward(self, input, timesteps, **kwargs):
        return self.inner_model.apply_model(input, timesteps, **kwargs)


class CompVisTimestepsVDenoiser(torch.nn.Module):

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return torch.sqrt(self.inner_model.alphas_cumprod)[t, None, None, None] * v + torch.sqrt(1 - self.inner_model.alphas_cumprod)[t, None, None, None] * x_t

    def forward(self, input, timesteps, **kwargs):
        model_output = self.inner_model.apply_model(input, timesteps, **kwargs)
        e_t = self.predict_eps_from_z_and_v(input, timesteps, model_output)
        return e_t


class CFGDenoiserTimesteps(CFGDenoiser):

    def __init__(self, sampler):
        super().__init__(sampler)
        self.alphas = shared.sd_model.alphas_cumprod
        self.mask_before_denoising = True

    def get_pred_x0(self, x_in, x_out, sigma):
        ts = sigma
        a_t = self.alphas[ts][:, None, None, None]
        sqrt_one_minus_at = (1 - a_t).sqrt()
        pred_x0 = (x_in - sqrt_one_minus_at * x_out) / a_t.sqrt()
        return pred_x0

    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser = CompVisTimestepsVDenoiser if shared.sd_model.parameterization == 'v' else CompVisTimestepsDenoiser
            self.model_wrap = denoiser(shared.sd_model)
        return self.model_wrap


class SdUnet(torch.nn.Module):

    def forward(self, x, timesteps, context, *args, **kwargs):
        raise NotImplementedError()

    def activate(self):
        pass

    def deactivate(self):
        pass


class VAEApprox(nn.Module):

    def __init__(self, latent_channels=4):
        super(VAEApprox, self).__init__()
        self.conv1 = nn.Conv2d(latent_channels, 8, (7, 7))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 16, (3, 3))
        self.conv7 = nn.Conv2d(16, 8, (3, 3))
        self.conv8 = nn.Conv2d(8, 3, (3, 3))

    def forward(self, x):
        extra = 11
        x = nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = nn.functional.leaky_relu(x, 0.1)
        return x


class Clamp(nn.Module):

    @staticmethod
    def forward(x):
        return torch.tanh(x / 3) * 3


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Block(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def decoder(latent_channels=4):
    return nn.Sequential(Clamp(), conv(latent_channels, 64), nn.ReLU(), Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False), Block(64, 64), conv(64, 3))


class TAESDDecoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, decoder_path='taesd_decoder.pth', latent_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        if latent_channels is None:
            latent_channels = 16 if 'taesd3' in str(decoder_path) else 4
        self.decoder = decoder(latent_channels)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location='cpu' if devices.device.type != 'cuda' else None))


def encoder(latent_channels=4):
    return nn.Sequential(conv(3, 64), Block(64, 64), conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64), conv(64, latent_channels))


class TAESDEncoder(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path='taesd_encoder.pth', latent_channels=None):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        if latent_channels is None:
            latent_channels = 16 if 'taesd3' in str(encoder_path) else 4
        self.encoder = encoder(latent_channels)
        self.encoder.load_state_dict(torch.load(encoder_path, map_location='cpu' if devices.device.type != 'cuda' else None))


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AutocastLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'n_in': 4, 'n_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Clamp,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DismantledBlock,
     lambda: ([], {'hidden_size': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (Downsample,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (FinalLayer,
     lambda: ([], {'hidden_size': 4, 'patch_size': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {})),
    (QkvLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SwiGLUFeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4, 'multiple_of': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (T5LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Upsample,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VAEApprox,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (VectorEmbedder,
     lambda: ([], {'input_dim': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

