import sys
_module = sys.modules[__name__]
del sys
setup = _module
timesformer_pytorch = _module
rotary = _module
timesformer_pytorch = _module

from _paritybench_helpers import _mock_config, patch_functional
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


from math import log


from math import pi


import torch


from torch import nn


from torch import einsum


import torch.nn.functional as F


class AxialRotaryEmbedding(nn.Module):

    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0.0, log(max_freq / 2) / log(2), self.dim // 4, base=2)
        self.register_buffer('scales', scales)

    def forward(self, h, w, device):
        scales = rearrange(self.scales, '... -> () ...')
        scales = scales
        h_seq = torch.linspace(-1.0, 1.0, steps=h, device=device)
        h_seq = h_seq.unsqueeze(-1)
        w_seq = torch.linspace(-1.0, 1.0, steps=w, device=device)
        w_seq = w_seq.unsqueeze(-1)
        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi
        x_sinu = repeat(h_seq, 'i d -> i j d', j=w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i=h)
        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)
        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j=2), (sin, cos))
        return sin, cos


class RotaryEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        inv_freqs = 1.0 / 10000 ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, n, device):
        seq = torch.arange(n, device=device)
        freqs = einsum('i, j -> i j', seq, self.inv_freqs)
        freqs = torch.cat((freqs, freqs), dim=-1)
        freqs = rearrange(freqs, 'n d -> () n d')
        return freqs.sin(), freqs.cos()


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


def shift(t, amt):
    if amt is 0:
        return t
    return F.pad(t, (0, 0, 0, 0, amt, -amt))


class PreTokenShift(nn.Module):

    def __init__(self, frames, fn):
        super().__init__()
        self.frames = frames
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        f, dim = self.frames, x.shape[-1]
        cls_x, x = x[:, :1], x[:, 1:]
        x = rearrange(x, 'b (f n) d -> b f n d', f=f)
        dim_chunk = dim // 3
        chunks = x.split(dim_chunk, dim=-1)
        chunks_to_shift, rest = chunks[:3], chunks[3:]
        shifted_chunks = tuple(map(lambda args: shift(*args), zip(chunks_to_shift, (-1, 0, 1))))
        x = torch.cat((*shifted_chunks, *rest), dim=-1)
        x = rearrange(x, 'b f n d -> b (f n) d')
        x = torch.cat((cls_x, x), dim=1)
        return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rot_emb(q, k, rot_emb):
    sin, cos = rot_emb
    rot_dim = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :rot_dim], t[..., rot_dim:]), (q, k))
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
    return q, k


def exists(val):
    return val is not None


def attn(q, k, v, mask=None):
    sim = einsum('b i d, b j d -> b i j', q, k)
    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Attention(nn.Module):

    def __init__(self, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, einops_from, einops_to, mask=None, cls_mask=None, rot_emb=None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, :1], t[:, 1:]), (q, k, v))
        cls_out = attn(cls_q, k, v, mask=cls_mask)
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))
        if exists(rot_emb):
            q_, k_ = apply_rot_emb(q_, k_, rot_emb)
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r), (cls_k, cls_v))
        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)
        out = attn(q_, k_, v_, mask=mask)
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)
        out = torch.cat((cls_out, out), dim=1)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class TimeSformer(nn.Module):

    def __init__(self, *, dim, num_frames, num_classes, image_size=224, patch_size=16, channels=3, depth=12, heads=8, dim_head=64, attn_dropout=0.0, ff_dropout=0.0, rotary_emb=True, shift_tokens=False):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2
        self.heads = heads
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.use_rotary_emb = rotary_emb
        if rotary_emb:
            self.frame_rot_emb = RotaryEmbedding(dim_head)
            self.image_rot_emb = AxialRotaryEmbedding(dim_head)
        else:
            self.pos_emb = nn.Embedding(num_positions + 1, dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout=ff_dropout)
            time_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
            spatial_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
            if shift_tokens:
                time_attn, spatial_attn, ff = map(lambda t: PreTokenShift(num_frames, t), (time_attn, spatial_attn, ff))
            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))
            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))
        self.to_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, video, mask=None):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        hp, wp = h // p, w // p
        n = hp * wp
        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)
        tokens = self.to_patch_embedding(video)
        cls_token = repeat(self.cls_token, 'n d -> b n d', b=b)
        x = torch.cat((cls_token, tokens), dim=1)
        frame_pos_emb = None
        image_pos_emb = None
        if not self.use_rotary_emb:
            x += self.pos_emb(torch.arange(x.shape[1], device=device))
        else:
            frame_pos_emb = self.frame_rot_emb(f, device=device)
            image_pos_emb = self.image_rot_emb(hp, wp, device=device)
        frame_mask = None
        cls_attn_mask = None
        if exists(mask):
            mask_with_cls = F.pad(mask, (1, 0), value=True)
            frame_mask = repeat(mask_with_cls, 'b f -> (b h n) () f', n=n, h=self.heads)
            cls_attn_mask = repeat(mask, 'b f -> (b h) () (f n)', n=n, h=self.heads)
            cls_attn_mask = F.pad(cls_attn_mask, (1, 0), value=True)
        for time_attn, spatial_attn, ff in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n=n, mask=frame_mask, cls_mask=cls_attn_mask, rot_emb=frame_pos_emb) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f=f, cls_mask=cls_attn_mask, rot_emb=image_pos_emb) + x
            x = ff(x) + x
        cls_token = x[:, 0]
        return self.to_out(cls_token)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForward,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_lucidrains_TimeSformer_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

