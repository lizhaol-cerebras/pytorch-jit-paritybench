
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


import logging


from string import Template


from typing import List


from typing import Union


from torch.nn import functional as F


from torch.utils.data import Dataset


from typing import Tuple


from torch.nn.utils.rnn import pad_sequence


from torchvision import transforms


import abc


from torch import nn


from torch import optim


import string


from typing import Optional


import torch.nn as nn


import types


from typing import Dict


import torch.nn.functional as F


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import _LRScheduler


from torch.utils.data import DataLoader


from torch.utils.data import random_split


logger = logging.getLogger(__name__)


class Kw_BatchNorm(nn.Module):
    """Kw_BatchNorm

    BatchNorm Layer for keywords

    """

    def __init__(self, kw_num: 'int', kw_dim: 'int', batchnorm_type: 'str', init_bias: 'torch.Tensor', init_scale: 'torch.Tensor', std_scale: 'int'=1, learnable: 'bool'=True, parallel: 'bool'=False) ->None:
        """init

        Args:
            kw_num (int): number of keywords
            kw_dim (int): dimension of keywords
            batchnorm_type (str): type for BatchNorm (`eachKw`: each kw has it's own params or `same`: all kw shared the same params)
            init_bias (torch.Tensor): initialized bias for BatchNorm
            init_scale (torch.Tensor): initialized scale for BatchNorm
            std_scale (int, optional): scale for init scale. Defaults to 1.
            learnable (bool, optional): if gamma and beta is learnable in BatchNoem. Defaults to True.
            parallel (bool, optional): (in eachKw mode) if each kw is BatchNorm parallelly. Defaults to False.

        """
        super().__init__()
        self.batchnorm_type = batchnorm_type
        self.kw_num = kw_num
        self.kw_dim = kw_dim
        self.std_scale = std_scale
        self.learnable = learnable
        self.parallel = parallel
        if self.batchnorm_type == 'eachKw':
            if self.parallel:
                self.bn_layer = nn.BatchNorm1d(kw_dim * self.kw_num)
            else:
                self.bn_layers = nn.ModuleList([nn.BatchNorm1d(kw_dim) for _ in range(self.kw_num)])
        elif self.batchnorm_type == 'same':
            self.bn_layer = nn.BatchNorm1d(kw_dim)
        else:
            raise NotImplementedError()
        if not isinstance(self.std_scale, list):
            self.std_scale = [self.std_scale] * self.kw_num
        self.init_bn(init_bias, init_scale)
        logger.info('Initialize BatchNorm({}) weight and bias learnable=({}) with token embeddings w/ scale={}, parallel=({})'.format(self.batchnorm_type, self.learnable, self.std_scale, self.parallel))

    def init_bn(self, init_bias: 'torch.Tensor', init_scale: 'torch.Tensor') ->None:
        """init_bn
        Initialize batchnorm's params
        Args:
            init_bias (torch.Tensor): bias
            init_scale (torch.Tensor): scale
        """
        if self.batchnorm_type == 'eachKw':
            if self.parallel:
                self.bn_layer.weight.data.copy_((init_scale * self.std_scale[0]).repeat(self.kw_num))
                self.bn_layer.bias.data.copy_(init_bias.repeat(self.kw_num))
                self.bn_layer.weight.requires_grad = self.learnable
                self.bn_layer.bias.requires_grad = self.learnable
            else:
                for i, _bn_layer in enumerate(self.bn_layers):
                    _bn_layer.weight.data.copy_(init_scale * self.std_scale[i])
                    _bn_layer.bias.data.copy_(init_bias)
                    _bn_layer.weight.requires_grad = self.learnable
                    _bn_layer.bias.requires_grad = self.learnable
        elif self.batchnorm_type == 'same':
            self.bn_layer.weight.data.copy_(init_scale * self.std_scale[0])
            self.bn_layer.bias.data.copy_(init_bias)
            self.bn_layer.weight.requires_grad = self.learnable
            self.bn_layer.bias.requires_grad = self.learnable

    def forward(self, keywords: 'torch.Tensor', seq_lens: 'torch.Tensor'=None) ->torch.Tensor:
        """forward

        Args:
            keywords (torch.Tensor): the input keywords embeddings
            seq_lens (torch.Tensor, optional): lengths for input keywords tensor. Defaults to None.

        Returns:
            torch.Tensor: batchnormed output
        """
        assert keywords.dim() == 3
        assert keywords.shape[2] == self.kw_dim
        if seq_lens is None:
            assert keywords.shape[1] == self.kw_num
        bsz = keywords.shape[0]
        if self.batchnorm_type == 'eachKw':
            if self.parallel:
                keywords = keywords.permute(0, 2, 1)
                keywords = keywords.reshape(bsz, -1)
                keywords = self.bn_layer(keywords)
                keywords = keywords.reshape(bsz, self.kw_dim, self.kw_num)
                keywords = keywords.permute(0, 2, 1)
            else:
                keywords_bns = []
                for i in range(self.kw_num):
                    keywords_bns.append(self.bn_layers[i](keywords[:, i]))
                keywords = torch.stack(keywords_bns, dim=1)
                del keywords_bns
        elif self.batchnorm_type == 'same':
            if seq_lens is None:
                keywords = self.bn_layer(keywords.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                kw_flatten = []
                assert seq_lens.dim() == 1
                seq_lens = seq_lens.tolist()
                offsets = [0]
                for b_i in range(keywords.size(0)):
                    kw_flatten.append(keywords[b_i, :seq_lens[b_i]].view(seq_lens[b_i], -1))
                    offsets.append(offsets[-1] + seq_lens[b_i])
                kw_flatten = torch.cat(kw_flatten, dim=0)
                assert kw_flatten.size(0) == sum(seq_lens)
                kw_flatten = self.bn_layer(kw_flatten)
                for b_i, (st_i, ed_i) in enumerate(zip(offsets[:-1], offsets[1:])):
                    assert seq_lens[b_i] == ed_i - st_i
                    keywords[b_i, :seq_lens[b_i]] = kw_flatten[st_i:ed_i]
        else:
            raise NotImplementedError()
        return keywords


class MLPLayers(nn.Module):
    """MLPLayers

    MLP Layers

    """

    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout
        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


def get_keypadding_mask(max_length: 'int', data_lens: 'torch.Tensor') ->torch.Tensor:
    """Create keypadding mask for attention layers

    Args:
        max_length (int): the max sequence length of the batch
        audio_len (torch.Tensor): the lens for each data in the batch, shape = (bsz,)

    Returns:
        torch.Tensor: key_padding_mask, bool Tensor, True for padding
    """
    bsz = data_lens.size(0)
    key_padding_mask = torch.ones([bsz, max_length])
    for mask, len in zip(key_padding_mask, data_lens):
        mask[:len] = 0.0
    key_padding_mask = key_padding_mask.type_as(data_lens).bool()
    return key_padding_mask


class KW_CascadedBranch(nn.Module):
    """KW_CascadedBranch

    Cascaded Branch for SpeechCLIP

    """

    def __init__(self, config: 'OrderedNamespace', audio_dim: 'int', text_dim: 'int', clip: 'ClipModel') ->None:
        """init

        Args:
            config (OrderedNamespace): config of the model
            audio_dim (int): dimension for audio features
            text_dim (int): dimension for subword embeddings
            clip (ClipModel): the CLIP model

        """
        super().__init__()
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.clip = clip
        self.config = config
        self.kw_projection_config = self.config.model_settings.cascaded_branch.keyword.get('kw_projection', None)
        logger.info('Using KW_CascadedBranch')
        self.keyword_num = config.model_settings.cascaded_branch.keyword.number
        self.cls = self._create_cls()
        logger.info('Start init [CLS] {}'.format(self.cls.shape))
        assert hasattr(TransformerModels, config.model_settings.cascaded_branch.transformer_type), "transformer structure '{}' not supported".format(config.model_settings.cascaded_branch.transformer_type)
        logger.info(f'Using {config.model_settings.cascaded_branch.transformer_type} as KW_CascadedBranch')
        self.self_att = getattr(TransformerModels, config.model_settings.cascaded_branch.transformer_type)(**config.model_settings.cascaded_branch.transformer_args)
        if self.kw_projection_config is None:
            logger.info('kw_projection not specified, using single linear layer as default')
            self.linear_proj = nn.Linear(self.config.model_settings.cascaded_branch.transformer_args.d_model, self.text_dim)
        else:
            logger.info(f'kw_projection dims:{self.kw_projection_config.dimensions} droupout:{self.kw_projection_config.dropout}')
            assert self.kw_projection_config.dimensions[0] == self.config.model_settings.cascaded_branch.transformer_args.d_model, f'first dim({self.kw_projection_config.dimensions[0]}) should match the audio encoder dim({self.config.model_settings.cascaded_branch.transformer_args.d_model})'
            assert self.kw_projection_config.dimensions[-1] == self.text_dim, f'last dim({self.kw_projection_config.dimensions[-1]}) should match the text encoder dim({self.text_dim})'
            self.linear_proj = MLPLayers(units=self.kw_projection_config.dimensions, dropout=self.kw_projection_config.dropout)
        self.vector_quantizer = None
        self.vq_type = config.model_settings.cascaded_branch.vq.type
        if not hasattr(vector_quantizers, config.model_settings.cascaded_branch.vq.type):
            raise NotImplementedError('Vq ({}) not implemented'.format(config.model_settings.cascaded_branch.vq.type))
        self.vector_quantizer = getattr(vector_quantizers, self.vq_type)(**config.model_settings.cascaded_branch.vq.args)
        if hasattr(config.model_settings.cascaded_branch.keyword, 'batchnorms'):
            self.bn_layer = Kw_BatchNorm(kw_num=self.keyword_num, kw_dim=self.text_dim, batchnorm_type=config.model_settings.cascaded_branch.keyword.batchnorms.type, init_bias=torch.mean(self.clip.model.token_embedding.weight, dim=0), init_scale=torch.std(self.clip.model.token_embedding.weight, dim=0), std_scale=config.model_settings.cascaded_branch.keyword.batchnorms.std_scale, learnable=config.model_settings.cascaded_branch.keyword.batchnorms.learnable if hasattr(config.model_settings.cascaded_branch.keyword.batchnorms, 'learnable') else True, parallel=config.model_settings.cascaded_branch.keyword.batchnorms.parallel if hasattr(config.model_settings.cascaded_branch.keyword.batchnorms, 'parallel') else False)

    def _create_cls(self) ->torch.nn.Parameter:
        """Create CLS

        Returns:
            torch.nn.Parameter: the params for CLS(s)
        """
        return torch.nn.Parameter(torch.randn([1, self.keyword_num, self.config.model_settings.cascaded_branch.transformer_args.d_model]))

    def extract_hidden_states(self, audio_feat: 'torch.Tensor', audio_len: 'torch.Tensor') ->Tuple:
        """extract_hidden_states
        Extracting hidden representation of each layers

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: tuples of hiddenstates
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)
        key_padding_mask = get_keypadding_mask(max_length=total_max_len, data_lens=audio_len + self.keyword_num)
        hidden_states = self.self_att.extract_hidden_states(src=src, key_padding_mask=key_padding_mask)
        hidden_states = [x[:, self.keyword_num:, ...] for x in hidden_states]
        return tuple(hidden_states)

    def forward(self, audio_feat: 'torch.Tensor', audio_len: 'torch.Tensor') ->Tuple[torch.Tensor, dict, torch.Tensor]:
        """forward

        Args:
            audio_feat (torch.Tensor)
            audio_len (torch.Tensor)

        Returns:
            Tuple: (audio_feat, vq_results, keywords)
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)
        key_padding_mask = get_keypadding_mask(max_length=total_max_len, data_lens=audio_len + self.keyword_num)
        keywords = self.self_att(src=src, key_padding_mask=key_padding_mask)
        keywords = keywords[:, :self.keyword_num].reshape(-1, self.keyword_num, self.audio_dim)
        keywords = self.linear_proj(keywords)
        if hasattr(self, 'bn_layer'):
            keywords = self.bn_layer(keywords)
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(F.cosine_similarity(keywords[:, i, :].view(bsz, self.text_dim, 1), self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0), dim=1))
        cos_score = torch.stack(cos_score, dim=1)
        assert cos_score.shape == (bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings), f'{cos_score.shape}, {bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings}'
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results['subword_prob'] @ self.clip.model.token_embedding.weight
        audio_feat = self.clip.encode_keywords(keywords, self.keyword_num)
        return audio_feat, vq_results, keywords

    def getAttentionMap(self, audio_feat: 'torch.Tensor', audio_len: 'torch.Tensor'):
        """getAttentionMap

        return attention maps for visualization

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: cls_weights, topk_kw, None
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)
        key_padding_mask = get_keypadding_mask(max_length=total_max_len, data_lens=audio_len + self.keyword_num)
        _, attn_output_weights = self.self_att.extract_attention_map(src=src, key_padding_mask=key_padding_mask)
        cls_weights = []
        for i in range(attn_output_weights.shape[0]):
            cls_weights.append(attn_output_weights[i, :, :self.keyword_num, :audio_len[i] + self.keyword_num])
        keywords = self.self_att(src=src, key_padding_mask=key_padding_mask)
        keywords = keywords[:, :self.keyword_num].reshape(-1, self.keyword_num, self.audio_dim)
        keywords = self.linear_proj(keywords)
        if hasattr(self, 'bn_layer'):
            keywords = self.bn_layer(keywords)
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(F.cosine_similarity(keywords[:, i, :].view(bsz, self.text_dim, 1), self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0), dim=1))
        cos_score = torch.stack(cos_score, dim=1)
        cos_score[..., 0] -= 100
        cos_score[..., 2] -= 100
        cos_score[..., 3] -= 100
        assert cos_score.shape == (bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings), f'{cos_score.shape}, {bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings}'
        topk_kw = [[[] for _ in range(self.keyword_num)] for _ in range(bsz)]
        _, topk_kw_ids = torch.topk(cos_score, dim=-1, k=10)
        for bsz_i in range(bsz):
            for kw_i in range(self.keyword_num):
                topk_kw[bsz_i][kw_i] = [self.clip.tokenizer.decoder[self.clip.reducedl2Original[x.item()]].replace('</w>', '') for x in topk_kw_ids[bsz_i, kw_i]]
        return cls_weights, topk_kw, None


class KW_ParallelBranch(nn.Module):
    """KW_ParallelBranch

    The parallel branch of SpeechCLIP

    """

    def __init__(self, config: 'OrderedNamespace', audio_dim: 'int', out_dim: 'int') ->None:
        super().__init__()
        self.config = config
        self.audio_dim = audio_dim
        self.out_dim = out_dim
        self.need_projection = self.config.model_settings.parallel_branch.get('need_projection', True)
        assert hasattr(TransformerModels, config.model_settings.parallel_branch.transformer_type)
        logger.info(f'Using {config.model_settings.parallel_branch.transformer_type} as KW_ParallelBranch (projection={self.need_projection})')
        self.self_att = getattr(TransformerModels, config.model_settings.parallel_branch.transformer_type)(**config.model_settings.parallel_branch.transformer_args)
        self.cls = self._create_cls()
        logger.info('Start init [CLS] {}'.format(self.cls.shape))
        if self.need_projection:
            self.linear_proj = nn.Linear(self.audio_dim, self.out_dim)

    def _create_cls(self):
        return torch.nn.Parameter(torch.randn([1, 1, self.config.model_settings.parallel_branch.transformer_args.d_model]))

    def extract_hidden_states(self, audio_feat: 'torch.Tensor', audio_len: 'torch.Tensor') ->Tuple:
        """extract_hidden_states
        Extract hiddenstates of parallel branch
        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: hidden representation of each layers
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + 1
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)
        key_padding_mask = get_keypadding_mask(max_length=total_max_len, data_lens=audio_len + 1)
        hidden_states = self.self_att.extract_hidden_states(src=src, key_padding_mask=key_padding_mask)
        hidden_states = [x[:, 1:, ...] for x in hidden_states]
        return tuple(hidden_states)

    def forward(self, audio_feat: 'torch.Tensor', audio_len: 'torch.Tensor') ->torch.Tensor:
        """forward

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            torch.Tensor: output
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + 1
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)
        key_padding_mask = get_keypadding_mask(max_length=total_max_len, data_lens=audio_len + 1)
        out = self.self_att(src=src, key_padding_mask=key_padding_mask)
        out = out[:, :1].reshape(-1, self.audio_dim)
        if hasattr(self, 'linear_proj'):
            out = self.linear_proj(out)
        return out


_clip_models = {'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'}


class ClipModel(nn.Module):

    def __init__(self, name: 'str', device: 'str'='cpu', image_encoder_trainable: 'bool'=False, text_encoder_trainable: 'bool'=False, reduce_subword_embbedding: 'str'=None, **kwargs):
        """Official CLIP model.

        Args:
            name (str): Name of CLIP model.
            device (str, optional): Device. Defaults to "cpu".
            image_encoder_trainable (bool, optional): Whether to train the image encoder. Defaults to False.
            text_encoder_trainable (bool, optional): Whether to train the text encoder. Defaults to False.
            reduce_subword_embbedding (str, optional): The reduced vocabulary. Defaults to False
        """
        super().__init__()
        assert name in _clip_models
        self.name = name
        self.device = device
        self.model, self.image_preprocess = clip.load(name, device)
        self.image_encoder_trainable = image_encoder_trainable
        self.text_encoder_trainable = text_encoder_trainable
        self.out_dim = self.model.transformer.width
        self.tokenizer = SimpleTokenizer()
        self.freeze_models()
        self.selected_text_emb_ids = None
        if reduce_subword_embbedding is not None:
            if not os.path.exists(reduce_subword_embbedding):
                logger.error(f'File not found {reduce_subword_embbedding}')
                exit(1)
            _data = np.load(reduce_subword_embbedding)
            self.selected_text_emb_ids = _data[:, 0]
            self.selected_text_emb_ids_dist = _data[:, 1]
            self.selected_text_emb_ids_dist = torch.from_numpy(self.selected_text_emb_ids_dist / np.sum(self.selected_text_emb_ids_dist))
            del _data
            logger.warning('Reduce text embedding to size of {}'.format(len(self.selected_text_emb_ids)))
            self.original_text_emb_weight = self.model.token_embedding.weight
            reduced_embedding_weight = self.model.token_embedding.weight[self.selected_text_emb_ids]
            self.model.token_embedding = nn.Embedding.from_pretrained(reduced_embedding_weight)
            if not self.text_encoder_trainable:
                self.model.token_embedding.weight.requires_grad = False
            self.original2Reduced = {old_id: _new_id for _new_id, old_id in enumerate(self.selected_text_emb_ids)}
            self.reducedl2Original = {_new_id: old_id for _new_id, old_id in enumerate(self.selected_text_emb_ids)}
            self.startOfTxt_reduced = self.original2Reduced[self.tokenizer.encoder['<|startoftext|>']]
            self.endOfTxt_reduced = self.original2Reduced[self.tokenizer.encoder['<|endoftext|>']]
        else:
            pass

    def freeze_models(self):
        """Freeze Models if required"""
        if not self.image_encoder_trainable:
            for p in self.model.visual.parameters():
                p.requires_grad = False
        if not self.text_encoder_trainable:
            for p in self.model.token_embedding.parameters():
                p.requires_grad = False
            self.model.positional_embedding.requires_grad = False
            for p in self.model.transformer.parameters():
                p.requires_grad = False
            for p in self.model.ln_final.parameters():
                p.requires_grad = False
            self.model.text_projection.requires_grad = False
            self.model.logit_scale.requires_grad = False

    def trainable_params(self) ->list:
        params = []
        if self.image_encoder_trainable:
            params += list(self.model.visual.parameters())
        if self.text_encoder_trainable:
            params += list(self.model.token_embedding.parameters())
            params += [self.model.positional_embedding]
            params += list(self.model.transformer.parameters())
            params += list(self.model.ln_final.parameters())
            params += [self.model.text_projection]
        return params

    def update_device(self, device):
        self.device = device

    def prep_image(self, paths: 'list') ->torch.Tensor:
        """Prepare image tensor

        Args:
            paths (list): Paths to multiple images

        Returns:
            torch.Tensor: Preprocessed image tensor (B, 3, H, W)
        """
        image_list = []
        for p in paths:
            img = Image.open(p)
            image_list.append(self.image_preprocess(img))
        return torch.stack(image_list, dim=0)

    def prep_text(self, sents: 'list') ->torch.Tensor:
        """Tokenize text

        Args:
            sents (list): Sentences

        Returns:
            torch.Tensor: _description_
        """
        res = clip.tokenize(sents)
        if self.selected_text_emb_ids is not None:
            for sent in res:
                for i in range(len(sent)):
                    sent[i] = self.original2Reduced[sent[i].item()]
        return res

    def deTokenize(self, sents):
        if isinstance(sents, torch.Tensor):
            sents = sents.view(*sents.shape[:2]).tolist()
        res = []
        for sent in sents:
            if self.selected_text_emb_ids is not None:
                for i in range(len(sent)):
                    sent[i] = self.reducedl2Original[sent[i]]
            res.append(self.tokenizer.decode(sent).replace('<|startoftext|>', '').replace('<|endoftext|>', '').strip())
        return res

    def encode_image(self, image: 'torch.Tensor') ->torch.Tensor:
        """Encode a batch of images.

        Args:
            image (torch.Tensor): Images. (B, 3, H, W)

        Returns:
            torch.Tensor: Image features. (B, D)
        """
        return self.model.encode_image(image)

    def encode_text(self, text: 'torch.Tensor') ->torch.Tensor:
        """Encode a batch of sentences.
        Args:
            text (torch.Tensor): Sentences. (B, L)
        Returns:
            torch.Tensor: Text features. (B, D)
        """
        return self.model.encode_text(text)

    def encode_keywords(self, keywords: 'torch.Tensor', keyword_num: 'int') ->torch.Tensor:
        """encode_keywords

        Args:
            keywords (torch.Tensor): keywords input
            keyword_num (int): number of keywords

        Returns:
            torch.Tensor: output of CLIP Text Encoder
        """
        if isinstance(keywords, torch.Tensor):
            bsz = keywords.size(0)
        else:
            raise TypeError(f'Unknown keywords type {type(keywords)}')
        text = torch.zeros([bsz, 77], device=self.device, dtype=int)
        if self.selected_text_emb_ids is None:
            sot_token, eot_token = self.tokenizer.encoder['<|startoftext|>'], self.tokenizer.encoder['<|endoftext|>']
        else:
            sot_token, eot_token = self.startOfTxt_reduced, self.endOfTxt_reduced
        text[:, 0] = torch.full(text[:, 0].size(), sot_token, device=self.device)
        text[:, keyword_num + 1] = torch.full(text[:, keyword_num + 1].size(), eot_token, device=self.device)
        x = self.model.token_embedding(text)
        x[:, 1:1 + keyword_num] = keywords
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)
        x = x[:, 1 + keyword_num] @ self.model.text_projection
        return x

    def encode_subword(self, prob: 'torch.Tensor', audio_len: 'torch.Tensor', vq_type: 'string') ->torch.Tensor:
        """Encode a batch of subwords.

        Args:
            text (torch.Tensor): Sentences. (B, L)

        Returns:
            torch.Tensor: Text features. (B, D)
        """
        return self.encode_subword_prob(prob, audio_len, vq_type)

    def get_scores(self, image: 'torch.Tensor', text: 'torch.Tensor') ->tuple:
        """Get logit scores between the images and text sentences.

        Args:
            image (torch.Tensor): Images. (B_image, 3, H, W)
            text (torch.Tensor): Sentences. (B_text, L)

        Returns:
            tuple: (logits_per_image, logits_per_text) ((B_image, B_text), (B_text, B_image))
        """
        return self.model(image, text)

    def to(self, *args, **kwargs):
        super()
        self.device = self.model.token_embedding.weight.device
        return self


class nnTransformerEncoder(nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)

    def extract_hidden_states(self, src: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None, src_key_padding_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """Pass the input through the encoder layers in turn. (Return all hidden_states)

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        hidden_states = []
        for mod in self.layers:
            hidden_states.append(output)
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        hidden_states.append(output)
        if self.norm is not None:
            output = self.norm(output)
        return output, tuple(hidden_states)


class TransformerEncoder(nn.Module):

    def __init__(self, n_layers: 'int'=1, d_model: 'int'=768, nhead: 'int'=8, dim_feedforward: 'int'=3072, dropout: 'float'=0.1, activation: 'str'='gelu', layer_norm_eps: 'float'=1e-05, batch_first: 'bool'=True, norm_first: 'bool'=False) ->None:
        super().__init__()
        logger.info(f'Using {n_layers} layer transformer encoder')
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        encoder_norm = nn.LayerNorm(d_model, eps=1e-05)
        self.model = nnTransformerEncoder(encoder_layer, n_layers, encoder_norm)

    def forward(self, src: 'torch.Tensor', key_padding_mask: 'torch.Tensor'):
        return self.model(src=src, src_key_padding_mask=key_padding_mask)

    def extract_hidden_states(self, src: 'torch.Tensor', key_padding_mask: 'torch.Tensor'):
        """Extract all hidden states

        Args:
            src (torch.Tensor): src
            key_padding_mask (torch.Tensor): key padding mask

        Returns:
            tuple: (output,hidden_states)
        """
        return self.model.extract_hidden_states(src=src, src_key_padding_mask=key_padding_mask)[1]


class MultiheadAttentionAndNorm(nn.Module):

    def __init__(self, d_model: 'int'=768, nhead: 'int'=8, dropout: 'float'=0.1, layer_norm_eps: 'float'=1e-05, batch_first: 'bool'=True, **kwargs) ->None:
        super().__init__()
        self.multihead_attn_layer = nn.MultiheadAttention(d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first)
        self.attentionBlock_Norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src: 'torch.Tensor', key_padding_mask: 'torch.Tensor'):
        return self.attentionBlock_Norm(self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask)[0] + src)

    def extract_hidden_states(self, src: 'torch.Tensor', key_padding_mask: 'torch.Tensor'):
        return tuple([src, self.forward(src, key_padding_mask)])

    def extract_attention_map(self, src: 'torch.Tensor', key_padding_mask: 'torch.Tensor'):
        _out, _att_weight = self.multihead_attn_layer(src, src, src, key_padding_mask=key_padding_mask, average_attn_weights=False)
        _out = self.attentionBlock_Norm(_out + src)
        return _out, _att_weight


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Author: Yonglong Tian (yonglong@mit.edu)
    Date: May 07, 2020
    """

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, learnable_temperature=True):
        super(SupConLoss, self).__init__()
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            self.temperature = torch.nn.parameter.Parameter(torch.FloatTensor([temperature]))
        else:
            self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    @property
    def current_temperature(self):
        if self.learnable_temperature:
            return self.temperature.item()
        else:
            return self.temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(1 / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


MAX_EYE = 256


class MaskedContrastiveLoss(nn.Module):

    def __init__(self, temperature: 'float'=0.07, temperature_trainable: 'bool'=False, margin: 'float'=0.0, dcl: 'bool'=False, a2b: 'bool'=True, b2a: 'bool'=True):
        """Masked Contrastive Loss

        Args:
            temperature (float, optional): Temperature for scaling logits. Defaults to 0.07.
            temperature_trainable (bool, optional): Trains temperature. Defaults to False.
            margin (float, optional): Margin. Defaults to 0.0.
            dcl (bool, optional): Decoupled contrastive learning (https://arxiv.org/abs/2110.06848). Defaults to False.
            a2b (bool, optional): Computes A to B classification loss. Defaults to True.
            b2a (bool, optional): Computes B to A classification loss. Defaults to True.
        """
        super().__init__()
        assert a2b or b2a, 'Cannot set both `a2b` and `b2a` to False.'
        self.temperature_trainable = temperature_trainable
        self.margin = margin
        self.dcl = dcl
        self.a2b = a2b
        self.b2a = b2a
        if temperature_trainable:
            self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            self.temperature = 1 / temperature
        eye_mat = torch.eye(MAX_EYE, dtype=torch.bool)
        self.register_buffer('eye_mat', eye_mat)
        self.register_buffer('neg_eye_mat', ~eye_mat)
        self.register_buffer('eye_mat_fl', eye_mat.type(torch.float))

    @property
    def current_temperature(self) ->float:
        """Current Temperature

        Returns:
            float: Temperature
        """
        if self.temperature_trainable:
            temp = self.temperature.data.cpu().detach().float().exp().item()
        else:
            temp = self.temperature
        return float(temp)

    def forward(self, feat_A: 'torch.Tensor', feat_B: 'torch.Tensor', index: 'torch.LongTensor'=None) ->torch.Tensor:
        """Computes loss

        Args:
            feat_A (torch.Tensor): Features from view A or modality A.
            feat_B (torch.Tensor): Features from view B or modality B.
            index (torch.LongTensor, optional): Labels for each sample. Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """
        assert feat_A.shape == feat_B.shape, (feat_A.shape, feat_B.shape)
        B = feat_A.shape[0]
        with torch.no_grad():
            if index is not None:
                assert index.shape[0] == feat_A.shape[0], (index.shape, feat_A.shape)
                index = index.unsqueeze(1)
                neg_mask = index != index.t()
            else:
                neg_mask = self.neg_eye_mat[:B, :B].clone()
            pos_mask = self.eye_mat[:B, :B]
            if not self.dcl:
                neg_mask[pos_mask] = True
            neg_mask_fl = neg_mask.type(feat_A.dtype)
        if self.temperature_trainable:
            temperature = torch.exp(self.temperature)
        else:
            temperature = self.temperature
        logits = feat_A @ feat_B.t() * temperature
        if self.margin > 0.0:
            logits[pos_mask] -= self.margin
        pos_logits = logits[pos_mask]
        exp_logits = logits.exp() * neg_mask_fl
        loss = 0
        if self.a2b:
            neg_A2B = torch.log(exp_logits.sum(1))
            loss_A2B = (-pos_logits + neg_A2B).mean()
            loss += loss_A2B
        if self.b2a:
            neg_B2A = torch.log(exp_logits.sum(0))
            loss_B2A = (-pos_logits + neg_B2A).mean()
            loss += loss_B2A
        if self.a2b and self.b2a:
            loss = loss / 2
        return loss


class MeanPoolingLayer(nn.Module):

    def __init__(self, in_dim: 'int'=0, out_dim: 'int'=0, bias: 'bool'=True, pre_proj: 'bool'=True, post_proj: 'bool'=True):
        """Mean pooling layer with linear layers.

        Args:
            in_dim (int, optional): Input dimension. Defaults to 0.
            out_dim (int, optional): Output dimension. Defaults to 0.
            bias (bool, optional): Linear layer bias. Defaults to True.
            pre_proj (bool, optional): Pre-projection layer. Defaults to True.
            post_proj (bool, optional): Post-projection layer. Defaults to True.
        """
        super().__init__()
        self.pre_proj = None
        self.post_proj = None
        if in_dim > 0 and out_dim > 0:
            if pre_proj:
                self.pre_proj = nn.Linear(in_dim, out_dim, bias=bias)
            if post_proj:
                self.post_proj = nn.Linear(in_dim if not pre_proj else out_dim, out_dim, bias=bias)

    def forward(self, x: 'torch.Tensor', x_len: 'torch.Tensor'=None) ->torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): Input features. (B, T, D)
            x_len (torch.Tensor): Feature lengths. (B, )

        Returns:
            torch.Tensor: Mean pooled features.
        """
        if self.pre_proj is not None:
            x = self.pre_proj(x)
        if x_len is not None:
            x = [x[b, :x_len[b]].mean(0) for b in range(len(x))]
            x = torch.stack(x, dim=0)
        else:
            x = x.mean(1)
        if self.post_proj is not None:
            x = self.post_proj(x)
        return x


class AttentivePoolingLayer(nn.Module):

    def __init__(self, dim_A: 'int', dim_B: 'int', degraded: 'bool'=False) ->None:
        """Attentative Pooling

        Args:
            dim_A (int): dimension for modality A
            dim_B (int): dimension for modality B
        """
        super().__init__()
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.degraded = degraded
        if not degraded:
            self.U = torch.nn.Parameter(torch.randn(self.dim_A, self.dim_B))
            self.U.requires_grad = True
        else:
            assert self.dim_A == self.dim_B
            self.U = torch.nn.Parameter(torch.eye(self.dim_A))
            self.U.requires_grad = False
        self.softmaxLayer = torch.nn.Softmax(dim=-1)

    def generate_input_msk(self, input_A_lens: 'torch.Tensor'=None, input_B_lens: 'torch.Tensor'=None, max_Alen: 'int'=1, max_Blen: 'int'=1) ->torch.Tensor:
        """Generate input mask for pooling

        Args:
            input_A_lens (torch.Tensor, optional): lengths for modality A, shape: (bsz,1). Defaults to None.
            input_B_lens (torch.Tensor, optional): lengths for modality B, shape: (bsz,1). Defaults to None.
            max_Alen (int): max input len for modality A
            max_Blen (int): max input len for modality B


        Returns:
            torch.Tensor: input mask, shape: ( bsz, max_Aseqlen , max_Bseqlen )
        """
        if input_A_lens is None and input_B_lens is None:
            raise ValueError('input_A_lens and input_B_lens cannot both be None')
        if input_A_lens is not None and input_B_lens is not None:
            assert input_A_lens.shape[0] == input_B_lens.shape[0], 'input_A_lens and input_B_lens must have same bsz, but got {} and {} instead'.format(input_A_lens.shape[0], input_B_lens.shape[0])
        if input_A_lens is not None:
            bsz = input_A_lens.shape[0]
            device = input_A_lens.device
        else:
            bsz = input_B_lens.shape[0]
            device = input_B_lens.device
        msk = torch.zeros((bsz, max_Alen, max_Blen), device=device, dtype=float)
        for _b in range(bsz):
            if input_A_lens is not None:
                assert not input_A_lens[_b] == 0, 'Modality A has 0 length on {}'.format(_b)
                msk[_b, input_A_lens[_b]:, :] = float('-inf')
            if input_B_lens is not None:
                assert not input_B_lens[_b] == 0, 'Modality B has 0 length on {}'.format(_b)
                msk[_b, :, input_B_lens[_b]:] = float('-inf')
        return msk

    def batch_forward(self, input_A: 'torch.Tensor', input_B: 'torch.Tensor', intput_msk: 'torch.Tensor'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """Forward Attentive poolilng with A and B (can be different in batch dimension)
        Assume modality B has fixed size

        Args:
            input_A (torch.Tensor): input features for modality A, shape: (bsz_A,dim,seq_len)
            input_B (torch.Tensor): input features for modality B, shape: (bsz_B,dim,seq_len)
            intput_msk (torch.Tensor,Optional): input features mask for modality A,B , shape: (bsz_A, seq_lenA, seq_lenB)

            mask: 0 for on and -inf for off
                if one of the dimension (seq_lenA or seq_lenB) has seq_len of 1, it will be auto broadcast to the input tensor shape

        Returns:
            Tuple[ torch.Tensor,torch.Tensor ]: (bsz_A,bsz_B,dimA), (bsz_A,bsz_B,dimB)
        """
        assert len(input_A.shape) == 3, 'input_A.shape must be (bsz_A,dim,seq_len)'
        assert len(input_B.shape) == 3, 'input_B.shape must be (bsz_B,dim,seq_len)'
        if intput_msk is not None:
            assert input_A.shape[0] == intput_msk.shape[0], 'input and intput_msk must have same bsz, but got {} and {} instead'.format(input_A.shape[0], intput_msk.shape[0])
            if intput_msk.shape[1] == 1:
                intput_msk = intput_msk.repeat(1, input_A.shape[1], 1)
            if intput_msk.shape[2] == 1:
                intput_msk = intput_msk.repeat(1, 1, input_B.shape[2])
        _align = torch.matmul(input_A.permute(0, 2, 1), self.U)
        _align = torch.einsum('acd,bdf->abcf', [_align, input_B])
        _align = torch.tanh(_align)
        if intput_msk is not None:
            intput_msk = intput_msk.unsqueeze(1).repeat(1, _align.shape[1], 1, 1)
            assert _align.shape == intput_msk.shape, '{},{}'.format(_align.shape, intput_msk.shape)
            intput_msk = intput_msk
            intput_msk = intput_msk.type_as(_align)
            _align = _align + intput_msk
        _align = _align.reshape(-1, input_A.shape[2], input_B.shape[2])
        _scoreA, _ = torch.max(_align, dim=2)
        _scoreB, _ = torch.max(_align, dim=1)
        del _align
        assert _scoreA.shape == (input_A.shape[0] * input_B.shape[0], input_A.shape[2])
        assert _scoreB.shape == (input_A.shape[0] * input_B.shape[0], input_B.shape[2])
        _scoreA = F.softmax(_scoreA, dim=-1)
        _scoreB = F.softmax(_scoreB, dim=-1)
        _scoreA = _scoreA.reshape(input_A.shape[0], input_B.shape[0], input_A.shape[2])
        _scoreB = _scoreB.reshape(input_A.shape[0], input_B.shape[0], input_B.shape[2])
        output_A = torch.matmul(input_A.unsqueeze(1).repeat(1, input_B.shape[0], 1, 1), _scoreA.unsqueeze(-1))
        output_B = torch.matmul(input_B.unsqueeze(0).repeat(input_A.shape[0], 1, 1, 1), _scoreB.unsqueeze(-1))
        del _scoreA, _scoreB
        output_A = output_A.reshape(input_A.shape[0], input_B.shape[0], input_A.shape[1])
        output_B = output_B.reshape(input_A.shape[0], input_B.shape[0], input_B.shape[1])
        return output_A, output_B

    def cal_batch_embedding(self, input_A: 'torch.Tensor', input_B: 'torch.Tensor', intput_msk: 'torch.Tensor'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """Calculate Embedding in Batch

        Assume that instance in modality B is represented by one vector for each

        Args:
            input_A (torch.Tensor): input features for modality A, shape: (bsz,dim,seq_len)
            input_B (torch.Tensor): input features for modality B, shape: (dim, total_data_pairs_count)
                                    len_B is the total number of data pairs in the dataset

            intput_msk (torch.Tensor, optional): input features mask for modality A,B , shape: (bsz, seq_lenA, 1). Defaults to None.

        Returns:
            torch.Tensor,torch.Tensor ]: modelity A's pooled representation (B is omitted, since it is the same after attentive pooling)
        """
        assert len(input_A.shape) == 3, 'input_A.shape must be (bsz,dim,seq_len)'
        assert len(input_B.shape) == 2, 'input_B.shape must be (dim,total_data_pairs_count)'
        if intput_msk is not None:
            assert input_A.shape[0] == intput_msk.shape[0], 'input and intput_msk must have same bsz, but got {} and {} instead'.format(input_A.shape[0], input_B.shape[0])
        _align = torch.matmul(self.U, input_B)
        _align = torch.matmul(input_A.permute(0, 2, 1), _align)
        _align = torch.tanh(_align)
        assert _align.shape == (input_A.shape[0], input_A.shape[2], input_B.shape[1]), '{} {}'.format(_align.shape, (input_A.shape[0], input_A.shape[2], input_B.shape[1]))
        if intput_msk is not None:
            assert _align.shape[:2] == intput_msk.shape[:2], '{},{}'.format(_align.shape, intput_msk.shape)
            assert intput_msk.shape[2] == 1
            intput_msk = intput_msk.repeat(1, 1, _align.shape[2])
            intput_msk = intput_msk
            intput_msk = intput_msk.type_as(_align)
            _align = _align + intput_msk
        _score = F.softmax(_align, dim=1)
        output_A = torch.matmul(input_A, _score)
        assert output_A.shape == (input_A.shape[0], input_A.shape[1], input_B.shape[1]), '{} {}'.format(output_A.shape, (input_A.shape[0], input_A.shape[1], input_B.shape[1]))
        return output_A

    def forward(self, input_A: 'torch.Tensor', input_B: 'torch.Tensor', intput_msk: 'torch.Tensor'=None) ->Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input_A (torch.Tensor): input features for modality A, shape: (bsz,dim,seq_len)
            input_B (torch.Tensor): input features for modality B, shape: (bsz,dim,seq_len)
            intput_msk (torch.Tensor,Optional): input features mask for modality A,B , shape: (bsz, seq_lenA, seq_lenB)

            mask: 0 for on and -inf for off
                if one of the dimension (seq_lenA or seq_lenB) has seq_len of 1, it will be auto broadcast to the input tensor shape

        Returns:
            Tuple[ torch.Tensor,torch.Tensor ]: (bsz,dimA), (bsz,dimB)
        """
        assert len(input_A.shape) == 3, 'input_A.shape must be (bsz,dim,seq_len)'
        assert len(input_B.shape) == 3, 'input_B.shape must be (bsz,dim,seq_len)'
        assert input_A.shape[0] == input_B.shape[0], 'input_A and input_B must have same bsz, but got {} and {} instead'.format(input_A.shape[0], input_B.shape[0])
        if intput_msk is not None:
            assert input_A.shape[0] == intput_msk.shape[0], 'input and intput_msk must have same bsz, but got {} and {} instead'.format(input_A.shape[0], input_B.shape[0])
            if intput_msk.shape[1] == 1:
                intput_msk = intput_msk.repeat(1, input_A.shape[1], 1)
            if intput_msk.shape[2] == 1:
                intput_msk = intput_msk.repeat(1, 1, input_B.shape[2])
        _align = torch.matmul(input_A.permute(0, 2, 1), self.U)
        _align = torch.matmul(_align, input_B)
        _align = torch.tanh(_align)
        if intput_msk is not None:
            assert _align.shape == intput_msk.shape, '{},{}'.format(_align.shape, intput_msk.shape)
            intput_msk = intput_msk
            intput_msk = intput_msk.type_as(_align)
            _align = _align + intput_msk
        _scoreA, _ = torch.max(_align, dim=2)
        _scoreB, _ = torch.max(_align, dim=1)
        assert _scoreA.shape == (input_A.shape[0], input_A.shape[2])
        assert _scoreB.shape == (input_B.shape[0], input_B.shape[2])
        _scoreA = F.softmax(_scoreA, dim=-1)
        _scoreB = F.softmax(_scoreB, dim=-1)
        _scoreA = _scoreA.unsqueeze(-1)
        _scoreB = _scoreB.unsqueeze(-1)
        output_A = torch.matmul(input_A, _scoreA).squeeze()
        output_B = torch.matmul(input_B, _scoreB).squeeze()
        return output_A, output_B


FEAT_SELECT_IDX_WEIGHTED_SUM_MODE = 'weighted_sum'


class WeightedSumLayer(nn.Module):

    def __init__(self, n_weights: 'int', normalize_features: 'bool'=False):
        """Weighted sum layer with learnable weights.

        Args:
            n_weights (int): Number of weights, i.e., number of hidden representations.
        """
        super().__init__()
        self.n_weights = n_weights
        self.weights = nn.Parameter(torch.zeros((n_weights,), dtype=torch.float))
        self.normalize_features = normalize_features
        if self.normalize_features:
            logger.info('Normalize feature before weighted sum')

    def forward(self, x: 'List[torch.Tensor]') ->torch.Tensor:
        """Weighted sum a list of tensors.

        Args:
            x (List[torch.Tensor]): Representations to be weighted summed.

        Returns:
            torch.Tensor: Weighted summed representations.
        """
        assert len(x) == self.n_weights, len(x)
        weights = torch.softmax(self.weights, dim=0)
        weights = weights.view(-1, *([1] * x[0].dim()))
        x = torch.stack(x, dim=0)
        if self.normalize_features:
            x = F.layer_norm(x, (x.shape[-1],))
        x = (weights * x).sum(0)
        return x


def freeze_model(m: 'nn.Module') ->None:
    for p in m.parameters():
        p.requires_grad = False


def init_weights(m: 'nn.Module') ->None:
    """Initialize module's weights

    Args:
        m (nn.Module): Module.
    """
    if isinstance(m, nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def random_crop_max_length(audio: 'torch.Tensor', max_len: 'int', orig_len: 'int'=1000000000) ->torch.Tensor:
    """Randomly crop an audio feature sequence into max_len.

    Args:
        audio (torch.Tensor): Audio features (T, *)
        max_len (int): Maximum length.
        orig_len (int, optional): Original length of audio. Defaults to 1000000000.

    Returns:
        torch.Tensor: Cropped audio features.
    """
    audio_len = min(len(audio), orig_len)
    if audio_len <= max_len or max_len < 0:
        return audio[:audio_len]
    offset = np.random.randint(audio_len - max_len)
    return audio[offset:offset + max_len]


class S3prlSpeechEncoderPlus(nn.Module):

    def __init__(self, name: 'str', pretrained: 'bool'=False, trainable: 'bool'=False, device: 'str'='cpu', feat_select_idx: 'Union[str, list]'='all', layer_drop: 'Union[str, float]'=0.0, max_audio_len: 'int'=-1, reinit_layers: 'List[int]'=[], unfreeze_layers: 'List[int]'=[], **kwargs):
        """Speech Encoder with S3PRL (v0.3.1)

        Args:
            name (str): Name of speech encoder.
            pretrained (bool, optional): Init with pretrained model. Defaults to False.
            trainable (bool, optional): Whether to update the model while training. Defaults to False.
            device (str, optional): Device. Defaults to "cpu".
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to "all".
            layerdrop (Union[str, float], optional): Layer drop rate. Defaults to 0.0.
        """
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.trainable = trainable
        self.device = device
        self.feat_select_idx = feat_select_idx
        self.max_audio_len = max_audio_len
        self.reinit_layers = reinit_layers
        self.unfreeze_layers = unfreeze_layers
        self.encoder = getattr(hub, name)()
        if hasattr(self.encoder, 'get_downsample_rates'):
            self.downsample_rate = self.encoder.get_downsample_rates('hidden_states')
        else:
            self.downsample_rate = 160
        if not pretrained:
            self.encoder.apply(init_weights)
        if not trainable:
            freeze_model(self.encoder)
        if self.name.startswith('hubert'):
            if isinstance(layer_drop, float) and layer_drop >= 0.0 and layer_drop <= 1.0:
                self.encoder.model.encoder.layerdrop = layer_drop
            elif layer_drop == 'original':
                pass
            else:
                raise ValueError(f'layer_drop = {layer_drop} is not supported.')
            assert not (len(reinit_layers) > 0 and len(unfreeze_layers) > 0)
            if len(reinit_layers) > 0:
                logger.warning(f'Reinitializing encoder layers {reinit_layers}')
                assert self.trainable
                for i, layer in enumerate(self.encoder.model.encoder.layers):
                    if i in reinit_layers:
                        layer.apply(init_weights)
                    else:
                        freeze_model(layer)
                freeze_model(self.encoder.model.encoder.pos_conv)
                freeze_model(self.encoder.model.layer_norm)
                freeze_model(self.encoder.model.feature_extractor)
                freeze_model(self.encoder.model.post_extract_proj)
                self.encoder.model.feature_grad_mult = 0
            if len(unfreeze_layers) > 0:
                logger.warning(f'Freezing encoder layers excluding {unfreeze_layers}')
                assert self.trainable
                for i, layer in enumerate(self.encoder.model.encoder.layers):
                    if i in unfreeze_layers:
                        pass
                    else:
                        freeze_model(layer)
                freeze_model(self.encoder.model.encoder.pos_conv)
                freeze_model(self.encoder.model.layer_norm)
                freeze_model(self.encoder.model.feature_extractor)
                freeze_model(self.encoder.model.post_extract_proj)
                self.encoder.model.feature_grad_mult = 0
        self.out_dim = 0
        self.upstream_model_hiddenstates_len = 0
        with torch.no_grad():
            wav = [torch.randn(16000, dtype=torch.float, device=device)]
            feat = self.encoder(wav)
            self.out_dim = feat['last_hidden_state'].shape[2]
            self.upstream_model_hiddenstates_len = len(feat['hidden_states'])
        logger.info(f'Loaded s3prl speech encoder ({name}): out_dim = {self.out_dim} layer_drop = {self.encoder.model.encoder.layerdrop}')
        if self.feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            logger.info(f'Using weighted sum for all hiddenstates({self.upstream_model_hiddenstates_len})')
            assert self.upstream_model_hiddenstates_len > 0
            self.weightedsum_layer = WeightedSumLayer(n_weights=self.upstream_model_hiddenstates_len)

    def trainable_params(self) ->list:
        if self.trainable and len(self.reinit_layers) == 0:
            return list(self.parameters())
        if self.trainable and len(self.reinit_layers) > 0:
            params = []
            for i in self.reinit_layers:
                params += list(self.encoder.model.encoder.layers[i].parameters())
            if not self.encoder.model.encoder.layer_norm_first:
                params += list(self.encoder.model.encoder.layer_norm.parameters())
            return params
        elif self.feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            logger.info('Adding weightedsum params')
            params = list(self.weightedsum_layer.parameters())
            return params
        else:
            return []

    def forward(self, wav: 'Union[torch.Tensor, list]', wav_len: 'Union[torch.Tensor, list]'=[], feat_select_idx: 'Union[str, list]'=None, return_hidden_states: 'bool'=False) ->Tuple[Union[torch.Tensor, list], torch.Tensor]:
        """Forward function for S3PRL speech encoder

        Args:
            wav (Union[torch.Tensor, list]): List of waveforms. (L, )
            wav_len (Union[torch.Tensor, list]): List of waveforms' lengths. Defaults to [].
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to None.

        Raises:
            KeyError: feat_select_idx is not "all", "hidden_states",
                      "last_hidden_state", or list.

        Returns:
            Tuple[Union[torch.Tensor, list], torch.Tensor]: Hidden features and their lengths.
        """
        if isinstance(wav, torch.Tensor):
            if wav.dim() == 2:
                if len(wav_len) > 0:
                    wav = [wav[b, :wav_len[b]] for b in range(len(wav))]
                else:
                    wav = [wav[b] for b in range(len(wav))]
            elif wav.dim() == 1:
                wav = [wav]
        if self.training:
            wav = [random_crop_max_length(wav[b], self.max_audio_len, len(wav[b])) for b in range(len(wav))]
        if self.trainable:
            feat = self.encoder(wav)
        else:
            with torch.no_grad():
                feat = self.encoder(wav)
        wav_len = [len(w) for w in wav]
        feat_len = torch.LongTensor([round(l / self.downsample_rate) for l in wav_len])
        feat_len = torch.clamp_max(feat_len, feat['hidden_states'][0].shape[1])
        if feat_select_idx is None:
            feat_select_idx = self.feat_select_idx
        return_list = []
        if feat_select_idx == 'all':
            return_list = [feat, feat_len]
        elif feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            return_list = [self.weightedsum_layer(feat['hidden_states']), feat_len]
        elif isinstance(feat_select_idx, list):
            feat = [feat['hidden_states'][i] for i in feat_select_idx]
            return_list = [feat, feat_len]
        elif feat_select_idx in feat:
            return_list = [feat[feat_select_idx], feat_len]
        else:
            raise KeyError(feat_select_idx)
        if return_hidden_states:
            return_list.append(feat['hidden_states'])
        return tuple(return_list)

    def to(self, *args, **kwargs):
        super()
        self.device = next(self.parameters()).device
        return self


def customFunc_hubert_forward(self, source: 'torch.Tensor', padding_mask: 'Optional[torch.Tensor]'=None, mask: 'bool'=True, output_layer: 'Optional[int]'=None) ->Dict[str, torch.Tensor]:
    """output layer is 1-based"""
    features = self.forward_features(source)
    features = features.transpose(1, 2)
    features = self.layer_norm(features)
    unmasked_features = features.clone()
    if padding_mask is not None:
        padding_mask = self.forward_padding_mask(features, padding_mask)
    if self.post_extract_proj is not None:
        features = self.post_extract_proj(features)
    features = self.dropout_input(features)
    unmasked_features = self.dropout_features(unmasked_features)
    if mask:
        x, mask_indices = self.apply_mask(features, padding_mask, None)
    else:
        x = features
        mask_indices = None
    x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=None if output_layer is None else output_layer - 1)
    return {'x': x, 'layer_results': layer_results}


def custom_FairseqTransformerEncoder_extract_features(self, x, padding_mask=None, tgt_layer=None):
    if padding_mask is not None:
        x = index_put(x, padding_mask, 0)
    x_conv = self.pos_conv(x.transpose(1, 2))
    x_conv = x_conv.transpose(1, 2)
    x = x + x_conv
    if not self.layer_norm_first:
        x = self.layer_norm(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = x.transpose(0, 1)
    layer_results = [x.transpose(0, 1)]
    r = None
    for i, layer in enumerate(self.layers):
        dropout_probability = np.random.random()
        if not self.training or dropout_probability > self.layerdrop:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
            layer_results.append(x.transpose(0, 1))
        if i == tgt_layer:
            r = x
            break
    if r is not None:
        x = r
    x = x.transpose(0, 1)
    return x, layer_results


class FairseqSpeechEncoder_Hubert(nn.Module):
    """FairseqSpeechEncoder_Hubert

    For Extracting HuBERT hiddenstates
    HuBERT load from fairseq

    """
    MODEL2URL = {'hubert': 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt', 'hubert_base': 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt', 'hubert_large_ll60k': 'https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt'}
    MODEL_DOWNSAMPLE_RATE = {'hubert': 320, 'hubert_base': 320, 'hubert_large_ll60k': 320}

    def __init__(self, name: 'str', pretrained: 'bool'=False, trainable: 'bool'=False, device: 'str'='cpu', feat_select_idx: 'Union[str, list]'='all', layer_drop: 'Union[str, float]'=0.0, max_audio_len: 'int'=-1, reinit_layers: 'List[int]'=[], unfreeze_layers: 'List[int]'=[], normalize_hiddenstates: 'bool'=False, normalize_type: 'str'='s3prl', **kwargs):
        """Speech Encoder with S3PRL (v0.3.1)
        Args:
            name (str): Name of speech encoder.
            pretrained (bool, optional): Init with pretrained model. Defaults to False.
            trainable (bool, optional): Whether to update the model while training. Defaults to False.
            device (str, optional): Device. Defaults to "cpu".
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to "all".
            layerdrop (Union[str, float], optional): Layer drop rate. Defaults to 0.0.
        """
        super().__init__()
        assert name in self.MODEL2URL, 'Model name({}) should be in {}'.format(name, self.MODEL2URL.keys())
        self.name = name
        self.pretrained = pretrained
        self.trainable = trainable
        self.feat_select_idx = feat_select_idx
        self.max_audio_len = max_audio_len
        self.reinit_layers = reinit_layers
        self.unfreeze_layers = unfreeze_layers
        self.normalize_hiddenstates = normalize_hiddenstates
        assert normalize_type in ['s3prl', 'method1', 'method2'], normalize_type
        if self.normalize_hiddenstates:
            logger.info('Normalize hidden states ({})'.format(normalize_type))
        self.normalize_type = normalize_type
        ckpt = _urls_to_filepaths(self.MODEL2URL[self.name], refresh=False)
        self.apply_customHubertForward()
        model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.encoder = model[0]
        self.encoder_task = task
        logger.info(f'Normalize waveform = ({self.encoder_task.cfg.normalize:})')
        if hasattr(self.encoder, 'get_downsample_rates'):
            self.downsample_rate = self.encoder.get_downsample_rates('hidden_states')
        else:
            self.downsample_rate = self.MODEL_DOWNSAMPLE_RATE[self.name]
        if not pretrained:
            self.encoder.apply(init_weights)
        if not trainable:
            freeze_model(self.encoder)
            self.encoder.eval()
        if self.name.startswith('hubert'):
            if isinstance(layer_drop, float) and layer_drop >= 0.0 and layer_drop <= 1.0:
                self.encoder.encoder.layerdrop = layer_drop
            elif layer_drop == 'original':
                pass
            else:
                raise ValueError(f'layer_drop = {layer_drop} is not supported.')
            assert not (len(reinit_layers) > 0 and len(unfreeze_layers) > 0)
            if len(reinit_layers) > 0:
                logger.warning(f'Reinitializing encoder layers {reinit_layers}')
                assert self.trainable
                for i, layer in enumerate(self.encoder.encoder.layers):
                    if i in reinit_layers:
                        layer.apply(init_weights)
                    else:
                        freeze_model(layer)
                freeze_model(self.encoder.encoder.pos_conv)
                freeze_model(self.encoder.layer_norm)
                freeze_model(self.encoder.feature_extractor)
                freeze_model(self.encoder.post_extract_proj)
                self.encoder.feature_grad_mult = 0
            if len(unfreeze_layers) > 0:
                logger.warning(f'Freezing encoder layers excluding {unfreeze_layers}')
                assert self.trainable
                for i, layer in enumerate(self.encoder.encoder.layers):
                    if i in unfreeze_layers:
                        pass
                    else:
                        freeze_model(layer)
                freeze_model(self.encoder.encoder.pos_conv)
                freeze_model(self.encoder.layer_norm)
                freeze_model(self.encoder.feature_extractor)
                freeze_model(self.encoder.post_extract_proj)
                self.encoder.feature_grad_mult = 0
        self.out_dim = 0
        with torch.no_grad():
            wav = [torch.randn(16000, dtype=torch.float, device='cpu')]
            padded_wav, wav_padding_mask = self.preprocess_input(wavs=wav)
            output = self.encoder.customHubertForward(source=padded_wav, padding_mask=wav_padding_mask, mask=None)
            self.upstream_model_hiddenstates_len = len(output['layer_results'])
            self.out_dim = output['x'].shape[2]
        logger.info(f'Loaded s3prl speech encoder ({name}): out_dim = {self.out_dim} layer_drop = {self.encoder.encoder.layerdrop}')
        if self.feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            logger.info(f'Using weighted sum for all hiddenstates({self.upstream_model_hiddenstates_len})')
            assert self.upstream_model_hiddenstates_len > 0
            self.weightedsum_layer = WeightedSumLayer(n_weights=self.upstream_model_hiddenstates_len, normalize_features=self.normalize_hiddenstates and self.normalize_type == 's3prl')

    def trainable_params(self) ->list:
        if self.trainable and len(self.reinit_layers) == 0:
            return list(self.parameters())
        if self.trainable and len(self.reinit_layers) > 0:
            params = []
            for i in self.reinit_layers:
                params += list(self.encoder.encoder.layers[i].parameters())
            if not self.encoder.encoder.layer_norm_first:
                params += list(self.encoder.encoder.layer_norm.parameters())
            return params
        elif self.feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            logger.info('Adding weightedsum params')
            params = list(self.weightedsum_layer.parameters())
            return params
        else:
            return []

    def apply_customHubertForward(self):
        TransformerEncoder.extract_features = copy_func(custom_FairseqTransformerEncoder_extract_features)
        HubertModel.customHubertForward = copy_func(customFunc_hubert_forward)

    def preprocess_input(self, wavs):
        if self.encoder_task.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs])
        wav_padding_mask = ~torch.lt(torch.arange(max(wav_lengths)).unsqueeze(0), wav_lengths.unsqueeze(1))
        padded_wav = pad_sequence(wavs, batch_first=True)
        return padded_wav, wav_padding_mask

    def forward(self, wav: 'Union[torch.Tensor, list]', wav_len: 'Union[torch.Tensor, list]'=[], feat_select_idx: 'Union[str, list]'=None, return_hidden_states: 'bool'=False) ->Tuple[Union[torch.Tensor, list], torch.Tensor]:
        """Forward function for S3PRL speech encoder
        Args:
            wav (Union[torch.Tensor, list]): List of waveforms. (L, )
            wav_len (Union[torch.Tensor, list]): List of waveforms' lengths. Defaults to [].
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to None.
        Raises:
            KeyError: feat_select_idx is not "all", "hidden_states",
                      "last_hidden_state", or list.
        Returns:
            Tuple[Union[torch.Tensor, list], torch.Tensor]: Hidden features and their lengths.
        """
        if isinstance(wav, torch.Tensor):
            if wav.dim() == 2:
                if len(wav_len) > 0:
                    wav = [wav[b, :wav_len[b]] for b in range(len(wav))]
                else:
                    wav = [wav[b] for b in range(len(wav))]
            elif wav.dim() == 1:
                wav = [wav]
        if self.training:
            wav = [random_crop_max_length(wav[b], self.max_audio_len, len(wav[b])) for b in range(len(wav))]
        padded_wav, wav_padding_mask = self.preprocess_input(wavs=wav)
        if self.trainable:
            features = self.encoder.customHubertForward(padded_wav, padding_mask=wav_padding_mask, mask=None)
        else:
            with torch.no_grad():
                features = self.encoder.customHubertForward(padded_wav, padding_mask=wav_padding_mask, mask=None)
        if self.normalize_hiddenstates:
            if self.normalize_type.startswith('method'):
                for i in range(len(features['layer_results'])):
                    if self.normalize_type == 'method1':
                        features['layer_results'][i] = features['layer_results'][i] / (torch.norm(features['layer_results'][i], dim=-1, keepdim=True) + 1e-08)
                    elif self.normalize_type == 'method2':
                        features['layer_results'][i] = features['layer_results'][i] / torch.mean(torch.norm(features['layer_results'][i], dim=-1), dim=-1).view(-1, 1, 1)
        feat = {'last_hidden_state': features['layer_results'][-1], 'hidden_states': tuple(features['layer_results'])}
        wav_len = [len(w) for w in wav]
        feat_len = torch.LongTensor([round(l / self.downsample_rate) for l in wav_len]).type_as(feat['last_hidden_state']).long()
        feat_len = torch.clamp_max(feat_len, feat['last_hidden_state'].shape[1])
        if feat_select_idx is None:
            feat_select_idx = self.feat_select_idx
        return_list = []
        if feat_select_idx == 'all':
            return_list.extend([feat, feat_len])
        elif feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            return_list.extend([self.weightedsum_layer(feat['hidden_states']), feat_len])
        elif isinstance(feat_select_idx, list):
            feat = [feat['hidden_states'][i] for i in feat_select_idx]
            return_list.extend([feat, feat_len])
        elif feat_select_idx in feat:
            return_list.extend([feat[feat_select_idx], feat_len])
        else:
            raise KeyError(feat_select_idx)
        if return_hidden_states:
            return_list.append(feat['hidden_states'])
        return tuple(return_list)


class SimpleVectorQuantizer(nn.Module):
    """SimpleVectorQuantizer"""

    def __init__(self, temp, groundTruthPerplexity=None, time_first=True, use_gumbel=False, hard=True):
        super().__init__()
        self.time_first = time_first
        self.use_gumbel = use_gumbel
        self.hard = hard
        if isinstance(temp, str):
            if temp.startswith('learnable='):
                self.temp_type = 'learnable'
                temp = temp.replace('learnable=', '')
                temp = ast.literal_eval(temp)
                self.curr_temp = nn.parameter.Parameter(torch.FloatTensor([temp]))
                logger.info('Setting vq temp learnable (init={})'.format(temp))
            elif temp.startswith('fixed='):
                self.temp_type = 'fixed'
                temp = temp.replace('fixed=', '')
                temp = ast.literal_eval(temp)
                self.register_buffer('curr_temp', torch.FloatTensor([temp]))
                logger.info('Setting vq temp fixed={}'.format(temp))
            else:
                self.temp_type = 'scheduled'
                temp = ast.literal_eval(temp)
                assert len(temp) == 3, f'{temp}, {len(temp)}'
                self.max_temp, self.min_temp, self.temp_decay = temp
                logger.info('Setting vq temp scheduled = ({},{},{})'.format(*temp))
                self.curr_temp = self.max_temp
        self.codebook_indices = None
        self.groundTruthPerplexity = groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            self.perplexity_criteria = nn.MSELoss()

    def set_num_updates(self, num_updates):
        if self.temp_type == 'scheduled':
            self.curr_temp = max(self.max_temp * self.temp_decay ** num_updates, self.min_temp)

    def forward(self, x, prob_msk=[0, 2, 3], produce_targets=True):
        if not self.time_first:
            x = x.transpose(1, 2)
        result = {'num_vars': x.shape[-1]}
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = x.view(bsz * tsz * 1, -1)
        for i in prob_msk:
            x[:, i] += float('-inf')
        _, k = x.max(-1)
        hard_x = x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, 1, -1)
        hard_x = hard_x.squeeze()
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result['code_perplexity'] = torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-07), dim=-1)).sum()
        avg_probs = torch.softmax(x.view(bsz * tsz, 1, -1).float(), dim=-1).mean(dim=0)
        probs_per_t = torch.softmax(x.view(bsz, tsz, -1), dim=-1).contiguous().permute(1, 0, 2)
        assert probs_per_t.shape[0] == tsz
        assert probs_per_t.shape[1] == bsz
        ent_per_t = -torch.sum(probs_per_t * torch.log(probs_per_t + 1e-09), dim=-1)
        ent_per_t = ent_per_t.mean(dim=-1)
        del probs_per_t
        result['ent_per_t'] = ent_per_t
        result['prob_perplexity'] = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-07), dim=-1)).sum()
        result['temp'] = self.curr_temp.item()
        if self.training:
            if self.use_gumbel:
                x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=self.hard).type_as(x)
            else:
                x = x / self.curr_temp
                x = F.softmax(x, dim=-1).type_as(x)
                if self.hard:
                    x = hard_x + x - x.detach()
        else:
            x = hard_x
        x = x.view(bsz * tsz, -1)
        result['subword_prob'] = x.view(bsz, tsz, -1)
        if self.groundTruthPerplexity is not None:
            result['diversity_loss'] = self.perplexity_criteria(result['prob_perplexity'], torch.tensor(self.groundTruthPerplexity).type_as(x)) / (result['num_vars'] - self.groundTruthPerplexity) ** 2
        else:
            result['diversity_loss'] = (result['num_vars'] - result['prob_perplexity']) / result['num_vars']
        if produce_targets:
            result['targets'] = x.view(bsz * tsz * 1, -1).argmax(dim=-1).view(bsz, tsz, 1).detach()
        return result


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (AttentivePoolingLayer,
     lambda: ([], {'dim_A': 4, 'dim_B': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MaskedContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
    (MeanPoolingLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SupConLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
]

