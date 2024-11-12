
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


from torchvision import datasets


from torchvision import transforms


import torch.nn as nn


from torch import optim


import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix


from sklearn.metrics import accuracy_score


class EmbedLayer(nn.Module):
    """
    Class for Embedding an Image.
    It breaks image into patches and embeds patches using a Conv2D Operation (Works same as the Linear layer).
    Next, a learnable positional embedding vector is added to all the patch embeddings to provide spatial position.
    Finally, a classification token is added which is used to classify the image.

    Parameters:
        n_channels (int) : Number of channels of the input image
        embed_dim  (int) : Embedding dimension
        image_size (int) : Image size
        patch_size (int) : Patch size
        dropout  (float) : dropout value

    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH
    
    Returns:
        Tensor: Embedding of the image of shape B, S, E
    """

    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim), requires_grad=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)
        x = x.reshape([B, x.shape[1], -1])
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """
    Class for computing self attention Self-Attention

    Parameters:
        embed_dim (int)        : Embedding dimension
        n_attention_heads (int): Number of attention heads to use for performing MultiHeadAttention
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output after Self-Attention Module of shape B, S, E
    """

    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads
        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.out_projection = nn.Linear(self.head_embed_dim * self.n_attention_heads, self.embed_dim)

    def forward(self, x):
        b, s, e = x.shape
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xq = xq.permute(0, 2, 1, 3)
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xk = xk.permute(0, 2, 1, 3)
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim)
        xv = xv.permute(0, 2, 1, 3)
        xk = xk.permute(0, 1, 3, 2)
        x_attention = torch.matmul(xq, xk)
        x_attention /= float(self.head_embed_dim) ** 0.5
        x_attention = torch.softmax(x_attention, dim=-1)
        x = torch.matmul(x_attention, xv)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, s, e)
        x = self.out_projection(x)
        return x


class Encoder(nn.Module):
    """
    Class for creating an encoder layer

    Parameters:
        embed_dim (int)         : Embedding dimension
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        dropout (float)         : Dropout parameter
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Output of the encoder block of shape B, S, E
    """

    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))
        return x


class Classifier(nn.Module):
    """
    Classification module of the Vision Transformer. Uses the embedding of the classification token to generate logits.

    Parameters:
        embed_dim (int) : Embedding dimension
        n_classes (int) : Number of classes
    
    Input:
        x (tensor): Tensor of shape B, S, E

    Returns:
        Tensor: Logits of shape B, CL
    """

    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def vit_init_weights(m):
    """
    function for initializing the weights of the Vision Transformer.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)


class VisionTransformer(nn.Module):
    """
    Vision Transformer Class.

    Parameters:
        n_channels (int)        : Number of channels of the input image
        embed_dim  (int)        : Embedding dimension
        n_layers   (int)        : Number of encoder blocks to use
        n_attention_heads (int) : Number of attention heads to use for performing MultiHeadAttention
        forward_mul (float)     : Used to calculate dimension of the hidden fc layer = embed_dim * forward_mul
        image_size (int)        : Image size
        patch_size (int)        : Patch size
        n_classes (int)         : Number of classes
        dropout  (float)        : dropout value
    
    Input:
        x (tensor): Image Tensor of shape B, C, IW, IH

    Returns:
        Tensor: Logits of shape B, CL
    """

    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = Classifier(embed_dim, n_classes)
        self.apply(vit_init_weights)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Classifier,
     lambda: ([], {'embed_dim': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (EmbedLayer,
     lambda: ([], {'n_channels': 4, 'embed_dim': 4, 'image_size': 4, 'patch_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Encoder,
     lambda: ([], {'embed_dim': 4, 'n_attention_heads': 4, 'forward_mul': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SelfAttention,
     lambda: ([], {'embed_dim': 4, 'n_attention_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (VisionTransformer,
     lambda: ([], {'n_channels': 4, 'embed_dim': 4, 'n_layers': 1, 'n_attention_heads': 4, 'forward_mul': 4, 'image_size': 4, 'patch_size': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

