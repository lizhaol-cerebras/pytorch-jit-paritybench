
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


import copy


import time


import matplotlib.pyplot as plt


import torch


import numpy as np


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


import math


import torch.nn as nn


from torch.nn import functional as F


import random


import torchvision


import torchvision.transforms as transforms


from torch import nn


batch_size, seq_len = 128, 16


dim_in, dim_hidden, dim_out = 2, 30, 1


class LSTM_net(torch.nn.Module):

    def __init__(self):
        super(LSTM_net, self).__init__()
        W1 = 0.1 * torch.randn(dim_in + 2 * dim_hidden + 1, 4 * dim_hidden)
        W1[-1, dim_hidden:2 * dim_hidden] += 1.0
        W1[:, 2 * dim_hidden:3 * dim_hidden] *= 2.0
        self.W1 = torch.nn.Parameter(W1)
        self.W2 = torch.nn.Parameter(0.1 * torch.randn(dim_hidden + 1, dim_out))

    def forward(self, xs):
        h, c = torch.zeros(batch_size, dim_hidden), torch.zeros(batch_size, dim_hidden)
        for x in torch.unbind(xs):
            ifgo = torch.cat([x, h, c], dim=1) @ self.W1[:-1] + self.W1[-1]
            i, f, g, o = torch.chunk(torch.sigmoid(ifgo), 4, dim=1)
            c = f * c + i * (2.0 * g - 1.0)
            h = o * torch.tanh(c)
        return h @ self.W2[:-1] + self.W2[-1]


class AffineConv2d(torch.nn.Module):
    """
    Let's wrap function
        torch.nn.functional.conv2d
    as a class. The affine transform is
        [vectorized(image patch), 1] @ W
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, has_bias=True):
        super(AffineConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.has_bias = has_bias
        self.out_in_height_width = out_channels, in_channels, kernel_size, kernel_size
        std = (in_channels * kernel_size ** 2) ** -0.5
        w = torch.empty(out_channels, in_channels * kernel_size ** 2).normal_(std=std)
        if has_bias:
            b = torch.zeros(out_channels, 1)
            self.weight = torch.nn.Parameter(torch.cat([w, b], dim=1))
        else:
            self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        if self.has_bias:
            return F.conv2d(x, self.weight[:, :-1].view(self.out_in_height_width), bias=self.weight[:, -1], stride=self.stride, padding=self.padding)
        else:
            return F.conv2d(x, self.weight.view(self.out_in_height_width), stride=self.stride, padding=self.padding)


class AffineLinear(torch.nn.Module):
    """
    A linear layer clearly is an affine transform
    """

    def __init__(self, in_features, out_features, has_bias=True):
        super(AffineLinear, self).__init__()
        self.has_bias = has_bias
        w = torch.empty(in_features, out_features).normal_(std=in_features ** -0.5)
        if has_bias:
            b = torch.zeros(1, out_features)
            self.weight = torch.nn.Parameter(torch.cat([w, b]))
        else:
            self.weight = torch.nn.Parameter(w)

    def forward(self, x):
        if self.has_bias:
            return x @ self.weight[:-1] + self.weight[-1]
        else:
            return x @ self.weight


class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.W1 = torch.nn.Parameter(0.1 * torch.randn(6, 1 * 5 * 5 + 1))
        self.W2 = torch.nn.Parameter(0.1 * torch.randn(16, 6 * 5 * 5 + 1))
        self.W3 = torch.nn.Parameter(0.1 * torch.randn(16 * 4 * 4 + 1, 120))
        self.W4 = torch.nn.Parameter(0.1 * torch.randn(120 + 1, 84))
        self.W5 = torch.nn.Parameter(0.1 * torch.randn(84 + 1, 10))

    def forward(self, x):
        x = F.conv2d(x, self.W1[:, :-1].view(6, 1, 5, 5), bias=self.W1[:, -1])
        x = F.relu(F.max_pool2d(x, 2))
        x = F.conv2d(x, self.W2[:, :-1].view(16, 6, 5, 5), bias=self.W2[:, -1])
        x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(x.view(-1, 16 * 4 * 4).mm(self.W3[:-1]) + self.W3[-1])
        x = F.relu(x.mm(self.W4[:-1]) + self.W4[-1])
        return x.mm(self.W5[:-1]) + self.W5[-1]


class AffineRNN(torch.nn.Module):
    """
    Class AffineRNN wraps function
        torch._VF.rnn_tanh 
    into a class.
    I only consider the case with:
        has_biases=True, bidirectional=False, batch_first=True, nonl=tanh
        
    The affine transform of each layer is 
    
                                        [    w_ih,  ]
       [inputs, hidden states, 1]   @   [    w_hh,  ]
                                        [    bias,  ]
                                        
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(AffineRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        def rnn_weight_initialization(input_size, hidden_size):
            """
            the weight matrix is store as
            [   w_ih,
                w_hh,
                bias, ]
            
            The normal_ initialization might be better than uniform_ (keep the same variance)? 
            """
            w = torch.empty(input_size + hidden_size, hidden_size).normal_(std=(3 * (input_size + hidden_size)) ** -0.5)
            w[input_size:] = torch.linalg.qr(w[input_size:])[0]
            b = torch.zeros(1, hidden_size)
            return torch.cat([w, b])
        self.param0 = torch.nn.Parameter(rnn_weight_initialization(input_size, hidden_size))
        for i in range(num_layers - 1):
            self.register_parameter(f'param{i + 1}', torch.nn.Parameter(rnn_weight_initialization(hidden_size, hidden_size)))

    def forward(self, inputs, hx):
        weights = []
        for i in range(self.num_layers):
            p = self.get_parameter(f'param{i}')
            weights.extend([p[:-self.hidden_size - 1].t().contiguous(), p[-self.hidden_size - 1:-1].t().contiguous(), p[-1], torch.zeros_like(p[-1])])
        return torch._VF.rnn_tanh(inputs, hx, weights, has_biases=True, num_layers=self.num_layers, dropout=self.dropout, train=self.training, bidirectional=False, batch_first=True)


device = torch.device('cpu')


def get_rand_orth(dim):
    temp = np.random.normal(size=[dim, dim])
    q, _ = np.linalg.qr(temp)
    return torch.tensor(q, dtype=torch.float32)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.W1x = torch.nn.Parameter(0.1 * torch.randn(dim_in, dim_hidden))
        self.W1h = torch.nn.Parameter(get_rand_orth(dim_hidden))
        self.b1 = torch.nn.Parameter(torch.zeros(dim_hidden))
        self.W2 = torch.nn.Parameter(0.1 * torch.randn(dim_hidden, dim_out))
        self.b2 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, xs):
        h = torch.zeros(batch_size, dim_hidden, device=device)
        for x in torch.unbind(xs):
            h = torch.tanh(x @ self.W1x + h @ self.W1h + self.b1)
        return h @ self.W2 + self.b2


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-05)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            None
            self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(wte=nn.Embedding(config.vocab_size, config.n_embd), wpe=nn.Embedding(config.block_size, config.n_embd), drop=nn.Dropout(config.dropout), h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ln_f=LayerNorm(config.n_embd, bias=config.bias)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        None

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f'Cannot forward sequence of length {t}, block size is only {self.config.block_size}'
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss


class Logistic(torch.nn.Module):

    def __init__(self):
        super(Logistic, self).__init__()
        self.W = torch.nn.Parameter(torch.zeros(28 ** 2 + (28 ** 2) ** 2, 10))
        self.b = torch.nn.Parameter(torch.zeros(10))

    def forward(self, x):
        x1 = x.view(-1, 28 ** 2)
        x2 = torch.linalg.matmul(x1[:, :, None], x1[:, None, :])
        return torch.cat([x1, x2.view(-1, (28 ** 2) ** 2)], 1).mm(self.W) + self.b

    def reset(self):
        with torch.no_grad():
            self.W *= 0
            self.b *= 0


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout), FeedForward(dim, mlp_dim, dropout=dropout)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = image_height // patch_height * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AffineConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AffineLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Block,
     lambda: ([], {'config': SimpleNamespace(n_embd=4, bias=4, n_head=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (CausalSelfAttention,
     lambda: ([], {'config': SimpleNamespace(n_embd=4, n_head=4, bias=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'ndim': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Logistic,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 784])], {})),
    (MLP,
     lambda: ([], {'config': SimpleNamespace(n_embd=4, bias=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

