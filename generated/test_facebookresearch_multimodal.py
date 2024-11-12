
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


from typing import List


from typing import Optional


from typing import Tuple


import torch


from torch import Tensor


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import DistributedSampler


from typing import Callable


from typing import Union


import re


from torchvision import transforms


import random


import time


import torch.backends.cudnn as cudnn


import torch.distributed as dist


from torch.optim import AdamW


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


import copy


from typing import Any


from typing import Dict


import torch.nn.functional as F


from torch import nn


import warnings


from functools import partial


import logging


from torchvision.datasets import CocoCaptions


import torchvision


from torch.utils.data.distributed import DistributedSampler


import numpy as np


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl


from torch.distributed.elastic.multiprocessing.errors import record


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


from torch.distributed.fsdp import MixedPrecision


from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from torch.nn.parallel import DistributedDataParallel as DDP


from torch.utils.tensorboard import SummaryWriter


from torch import distributed as dist


from torchvision.datasets import CocoDetection


from typing import Sequence


from torchvision.ops.boxes import box_iou


from torchvision.ops.boxes import box_convert


from typing import Iterable


import torchvision.transforms.functional as F


from torchvision import transforms as T


from collections import OrderedDict


from scipy.optimize import linear_sum_assignment


from torchvision.ops.boxes import generalized_box_iou


from copy import deepcopy


import functools


from collections import defaultdict


from collections import deque


import math


import torch.utils.data as data


from torchvision.models.video import S3D


import torchvision.datasets.samplers as video_samplers


from torch.utils.data.dataloader import default_collate


from torchvision.transforms.functional import InterpolationMode


import scipy.io


import torchvision.transforms as T


from torchvision.datasets.vision import VisionDataset


from torchvision.transforms import autoaugment


from torchvision.transforms import functional as F


from torchvision.transforms import InterpolationMode


import torch.utils.data


import torch.nn as nn


import torch.distributions as tdist


from torch.nn import CrossEntropyLoss


from torchvision.models.vision_transformer import VisionTransformer


from math import inf


from torch.nn import functional as F


from itertools import repeat


from torchvision.models.video.swin_transformer import PatchEmbed3d


import torch.optim as optim


from torch import tensor


from itertools import product


from itertools import chain


from torch import multiprocessing as mp


from torch import optim


import torch.multiprocessing as mp


from typing import NamedTuple


from torchvision.transforms import ToPILImage


from torch.utils.checkpoint import checkpoint


from enum import Enum


import torchvision.transforms as tv


from torch import nn as nn


from torch.distributions import Distribution


from torch.distributions import Normal


from abc import abstractmethod


from typing import Protocol


from typing import runtime_checkable


from typing import Generator


from abc import abstractproperty


from torchvision.models.feature_extraction import create_feature_extractor


from collections import namedtuple


from torchvision.models.resnet import Bottleneck


from torchvision.models.resnet import ResNet


from torch.nn import TransformerEncoder


from torch.nn import TransformerEncoderLayer


from torchvision.ops import StochasticDepth


from torch.optim.lr_scheduler import LambdaLR


from torch.optim.lr_scheduler import LRScheduler


from torch.optim.lr_scheduler import SequentialLR


from torch.optim.optimizer import Optimizer


from torchvision.models._api import Weights


from torchvision.models._utils import IntermediateLayerGetter


from torchvision.models.resnet import resnet101


from torchvision.models.resnet import ResNet101_Weights


from typing import cast


from torch import Size


from torchvision.models.video.swin_transformer import PatchMerging


from torchvision.models.video.swin_transformer import SwinTransformer3d as TVSwinTransformer3d


from typing import Mapping


import itertools


from torchvision.ops.stochastic_depth import StochasticDepth


from typing import OrderedDict


from functools import lru_cache


from typing import Set


from torchvision import transforms as image_transforms


from torchvision.transforms.functional import normalize


from torchvision.transforms.functional import resize


from torch.distributed import all_gather as all_gather_no_backprop


from torch.distributed.nn.functional import all_gather as all_gather_with_backprop


class PredictionHead(nn.Module):
    """
    Predict the following token autoregressively.

    Args:
        vocab_size (int): The number of different tokens the prediction_head can predict.
        hidden_size (int): The hidden size of the prediction_head.
        layer_norm_eps (float): The epsilon used by the prediction_head normalization layer.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function in the prediction_head.

    Inputs:
        hidden_states (Tensor): The hidden states of preceding tokens.

    Returns:
        Tensor: Prediction scores for the following token.
    """

    def __init__(self, vocab_size: 'int'=30522, hidden_size: 'int'=768, layer_norm_eps: 'float'=1e-12, transform_act_fn: 'Callable[[Tensor], Tensor]'=nn.functional.gelu) ->None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: 'Tensor') ->Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


def get_causal_attention_mask(tgt_seq_len: 'int', src_seq_len: 'Optional[int]'=None) ->Tensor:
    """
    Generates causal attention masks of dimensions (target_sequence_length, source_sequence_length).
    """
    if src_seq_len is None:
        src_seq_len = tgt_seq_len
    return torch.tril(torch.ones(tgt_seq_len, src_seq_len))


class ALBEFDecoder(nn.Module):
    """
    Generate the prediction scores for answers from image and question hidden states.

    Args:
        text_embeddings (ALBEFTextEmbeddings): Instantiated ALBEFTextEmbeddings.
        multimodal_encoder (ALBEFMultimodalEncoder): Instantiated ALBEFMultimodalEncoder.
        prediction_head (PredictionHead): Instantiated PredictionHead.

    Inputs:
        input_ids (Tensor of shape (batch_size, seq_len)):
            Input ids for input text tokens.
        attention_mask (Tensor of shape (batch_size, seq_len)):
            Input attention mask to avoid performing attention on padding token indices.
        encoder_hidden_states (Tensor of shape (batch_size, encoder_seq_len, hidden_size)):
            The encoder hidden states.
        encoder_attention_mask (Tensor of shape (batch_size, encoder_seq_len)):
            The attention mask for encoder hidden states.

    Returns:
        Tensor: Prediction scores for answers.
    """

    def __init__(self, text_embeddings: 'BERTTextEmbeddings', multimodal_encoder: 'ALBEFMultimodalEncoder', prediction_head: 'PredictionHead') ->None:
        super().__init__()
        self.text_embeddings = text_embeddings
        self.multimodal_encoder = multimodal_encoder
        self.prediction_head = prediction_head

    def get_extended_attention_mask_for_decoder(self, attention_mask: 'Tensor') ->Tensor:
        """
        Apply a causal mask in addition to the padding mask and make the mask broadcastable,
        such that future and masked tokens are ignored.

        Args:
            attention_mask (Tensor):
                Padding mask with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            extended_attention_mask (Tensor):
                The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
        """
        device = attention_mask.device
        batch_size, seq_length = attention_mask.shape
        causal_mask = get_causal_attention_mask(seq_length)
        causal_mask = causal_mask.repeat(batch_size, 1).view(batch_size, seq_length, seq_length)
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask
        return extended_attention_mask

    def forward(self, input_ids: 'Tensor', attention_mask: 'Tensor', encoder_hidden_states: 'Tensor', encoder_attention_mask: 'Tensor') ->Tensor:
        hidden_states = self.text_embeddings(input_ids)
        attention_mask = self.get_extended_attention_mask_for_decoder(attention_mask)
        decoder_output = self.multimodal_encoder(hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        prediction_scores = self.prediction_head(decoder_output)
        return prediction_scores


@torch.no_grad()
def momentum_update(model: 'nn.Module', model_m: 'nn.Module', momentum: 'float') ->None:
    for param, param_m in zip(model.parameters(), model_m.parameters()):
        param_m.data = param_m.data * momentum + param.data * (1 - momentum)


@torch.no_grad()
def remove_grad(model: 'nn.Module') ->None:
    for param in model.parameters():
        param.requires_grad = False


class ALBEFModelForVQA(nn.Module):
    """
    ALBEF Model for VQA finetuning and inference.

    Args:
        model (ALBEFModel): Instantiated ALBEFModel.
        answer_decoder (ALBEFDecoder): Instantiated ALBEFDecoder.
        loss (CausalLanguageModelingLoss): Instantiated CausalLanguageModelingLoss.

    Inputs:
        image (Tensor of shape (B, C, H, W)): Image features.
        question (Tensor of shape (B, L)): Question text features.
        question_atts (Tensor of shape (B, L)): Question attention mask.
        answers (Tensor of shape (N, M)): Answer text features.
        answers_atts (Tensor of shape (N, M)): Answer attention mask.
        ans_weights (Optional[Tensor] of shape (N)): Weights for each answer.
            Required if is_train is True.
        ans_lengths (Optional[List[int]] of length B): Number of answers for each question.
            ans_lengths should sum to N.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between clm_loss and loss_distill.
            Required if is_train is True.
        k (Optional[int]): The number of answers to return for inference.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.

    Returns:
        is_train is True:
            Tensor: The masked language modeling loss for input.
        is_train is False:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
    """

    def __init__(self, model: 'ALBEFModel', answer_decoder: 'ALBEFDecoder', loss: 'CausalLanguageModelingLoss') ->None:
        super().__init__()
        self.model = model
        self.answer_decoder = answer_decoder
        self.loss = loss
        self.answer_decoder_m = copy.deepcopy(self.answer_decoder)
        remove_grad(self.answer_decoder_m)

    def _train_forward(self, image: 'Tensor', question: 'Tensor', question_atts: 'Tensor', answers: 'Tensor', answers_atts: 'Tensor', ans_weights: 'Tensor', ans_lengths: 'List[int]', alpha: 'float') ->Tensor:
        """
        Forward step for training. Encode the inputs with the ALBEFModel.
        Generate pseudo-targets using answer_decoder_m (momentum decoder model).
        Generate answer predictions using answer_decoder.
        Compute masked language modeling loss of the predictions using answers as labels,
            pseudo-targets as soft-labels, and alpha as their interpolation value.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answers_atts (Tensor of shape (N, M)): Answer attention mask.
            ans_weights (Tensor of shape (N)): Weights for each answer.
            ans_lengths (List[int] of length B): Number of answers for each question.
                ans_lengths should sum to N.
            alpha (float): The interpolation value between clm_loss and loss_distill.

        Returns:
            Tensor: The masked language modeling loss for input.
        """
        encoder_outputs = self.model(image, question, question_atts)
        encoder_hidden_states, encoder_hidden_states_m, encoder_attention_mask = self._encoder_hidden_states(encoder_outputs.multimodal_embeddings, encoder_outputs.multimodal_embeddings_m, question_atts, ans_lengths)
        with torch.no_grad():
            momentum_update(self.answer_decoder, self.answer_decoder_m, self.model.momentum)
            prediction_scores_m = self.answer_decoder_m(input_ids=answers, attention_mask=answers_atts, encoder_hidden_states=encoder_hidden_states_m, encoder_attention_mask=encoder_attention_mask)
        prediction_scores = self.answer_decoder(input_ids=answers, attention_mask=answers_atts, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        labels = answers.masked_fill(answers == 0, self.loss.mask_token_id)
        loss = self.loss(labels, prediction_scores, prediction_scores_m, alpha)
        loss = ans_weights * loss
        loss = loss.sum() / image.size(0)
        return loss

    def _eval_forward(self, image: 'Tensor', question: 'Tensor', question_atts: 'Tensor', answers: 'Tensor', answer_atts: 'Tensor', k: 'int'=128) ->Tuple[Tensor, Tensor]:
        """
        Forward step for evaluation. Encode the inputs with the ALBEFModel.
        Generate answer autoregressively using the decoder, starting with the [CLS] token.
        Compute the answer ids and their perspective probabilities of the top k predictions.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answer_atts (Tensor of shape (N, M)): Answer attention mask.
            k (int): The number of answers to return for inference.

        Returns:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
        """
        encoder_outputs = self.model(image, question, question_atts)
        num_ques = question.size(0)
        start_ids = answers[0, 0].repeat(num_ques, 1)
        atts = torch.ones(start_ids.shape)
        prediction_scores = self.answer_decoder(input_ids=start_ids, attention_mask=atts, encoder_hidden_states=encoder_outputs.multimodal_embeddings, encoder_attention_mask=question_atts)
        logits = prediction_scores[:, 0, :]
        answer_first_token = answers[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)
        input_ids = []
        input_atts = []
        for topk_id in topk_ids:
            input_ids.append(answers.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids)
        input_atts = torch.cat(input_atts)
        targets_ids = input_ids.masked_fill(input_ids == 0, self.loss.mask_token_id)
        question_states = encoder_outputs.multimodal_embeddings.repeat_interleave(k, dim=0)
        question_atts = question_atts.repeat_interleave(k, dim=0)
        prediction_scores = self.answer_decoder(input_ids=input_ids, attention_mask=input_atts, encoder_hidden_states=question_states, encoder_attention_mask=question_atts)
        answer_loss = self.loss(targets_ids, prediction_scores)
        answer_loss = answer_loss.view(input_ids.size(0), -1)
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)
        topk_probs = F.softmax(log_probs_sum, dim=-1)
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)
        return topk_ids, topk_probs

    def _encoder_hidden_states(self, multimodal_embeds: 'Tensor', multimodal_embeds_m: 'Tensor', question_atts: 'Tensor', ans_lengths: 'List[int]') ->Tuple[Tensor, Tensor, Tensor]:
        """
        Repeat each image-question input, repeat its embedding and mask to match the number of answers it has.

        Args:
            multimodal_embeds (Tensor): Image-question embeddings.
            multimodal_embeds_m (Tensor): Image-question embeddings from the momentum model.
            question_atts (Tensor): Question attention mask.
            ans_lengths (List[int]): The number of answers each image-question input has.

        Returns:
            encoder_hidden_states (Tensor): Image-question embeddings after the repetition.
            encoder_hidden_states_m (Tensor): Image-question embeddings from the momentum model after the repetition.
            encoder_attention_mask (Tensor): Question attention mask after the repetition.
        """
        encoder_hidden_states = []
        encoder_attention_mask = []
        for b, n in enumerate(ans_lengths):
            encoder_hidden_states += [multimodal_embeds[b]] * n
            encoder_attention_mask += [question_atts[b]] * n
        encoder_hidden_states = torch.stack(encoder_hidden_states)
        encoder_attention_mask = torch.stack(encoder_attention_mask)
        with torch.no_grad():
            encoder_hidden_states_m = []
            for b, n in enumerate(ans_lengths):
                encoder_hidden_states_m += [multimodal_embeds_m[b]] * n
            encoder_hidden_states_m = torch.stack(encoder_hidden_states_m)
        return encoder_hidden_states, encoder_hidden_states_m, encoder_attention_mask

    def forward(self, image: 'Tensor', question: 'Tensor', question_atts: 'Tensor', answers: 'Tensor', answers_atts: 'Tensor', ans_weights: 'Optional[Tensor]'=None, ans_lengths: 'Optional[List[int]]'=None, alpha: 'Optional[float]'=0.0, k: 'Optional[int]'=128, is_train: 'Optional[bool]'=True) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(image, question, question_atts, answers, answers_atts, ans_weights, ans_lengths, alpha)
        else:
            return self._eval_forward(image, question, question_atts, answers, answers_atts, k)


class ALBEFModelForRetrieval(nn.Module):
    """
    ALBEF Model for Retrieval finetuning and inference.
    In training mode, the forward step computes image-text contrastive loss and
    image-text matching loss.
    In evaluation mode, the forward step takes 3 types of input:
        image: encode image input, project and normalize the embeddings.
        text: encode text input, project and normalize the embeddings.
        multimodal: create multimodal embeddings from image and text
            embeddings, and compute image-text matching scores.

    Args:
        model_with_similarity (ALBEFModelWithSimilarity): Instantiated ALBEFModelWithSimilarity.
        itc_loss (ImageTextContrastiveLoss): Instantiated ImageTextContrastiveLoss.
        hidden_size (int): Dimensionality of encoder outputs.

    Inputs:
        image (Optional[Tensor] of shape (B, C, H, W)): Image features.
            Required if is_train is True.
            Required if input_type is "image" or "multimodal".
        text (Optional[Tensor] of shape (B, L)): Text features.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        text_atts (Tensor of shape (B, L)): Text attention mask.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        idx (Tensor of shape (B)): Identifier for each image sample.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between clm_loss and loss_distill.
            Default is 0.
        input_type (Optional[str]): "image", "text", or "multimodal" indicating the encoding type.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.
            Default is True.

    Returns:
        is_train is True:
            Tensor: The sum of itc loss and itm loss.
        is_train is False:
            input_type is "image":
                Tuple[Tensor, Tensor]: Image embeddings and projected image features.
            input_type is "text":
                Tuple[Tensor, Tensor]: Text embeddings and projected text features.
            input_type is "multimodal"
                Tensor: Scores for the retrieval task.
    """

    def __init__(self, model_with_similarity: 'ALBEFModelWithSimilarity', itc_loss: 'ImageTextContrastiveLoss', hidden_size: 'int') ->None:
        super().__init__()
        self.model_with_similarity = model_with_similarity
        self.itc_loss = itc_loss
        self.itm_head = nn.Linear(hidden_size, 2)

    def _train_forward(self, image: 'Tensor', text: 'Tensor', text_atts: 'Tensor', idx: 'Tensor', alpha: 'float') ->Tensor:
        encoder_output = self.model_with_similarity(image, text, text_atts, idx)
        similarity_outputs = encoder_output.similarity
        similarity_targets = encoder_output.sim_targets
        itc_loss = self.itc_loss(similarity_outputs.sim_i2t, similarity_outputs.sim_t2i, similarity_outputs.sim_i2t_m, similarity_outputs.sim_t2i_m, similarity_targets, alpha)
        pos_embeddings = encoder_output.multimodal_embeddings[:, 0, :]
        neg_embeddings = encoder_output.multimodal_embeddings_neg[:, 0, :]
        vl_embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(pos_embeddings.size(0), dtype=torch.long), torch.zeros(neg_embeddings.size(0), dtype=torch.long)], dim=0)
        itm_loss = F.cross_entropy(vl_output, itm_labels)
        loss = itc_loss + itm_loss
        return loss

    def _encode_image(self, image: 'Tensor') ->Tuple[Tensor, Tensor]:
        image_embed = self.model_with_similarity.albef_model.vision_encoder(image)
        image_feat = F.normalize(self.model_with_similarity.vision_proj(image_embed[:, 0, :]), dim=-1)
        return image_embed, image_feat

    def _encode_text(self, text: 'Tensor', text_atts: 'Tensor') ->Tuple[Tensor, Tensor]:
        text_embed = self.model_with_similarity.albef_model.text_encoder(text, text_atts).last_hidden_state
        text_feat = F.normalize(self.model_with_similarity.text_proj(text_embed[:, 0, :]), dim=-1)
        return text_embed, text_feat

    def _image_text_matching_score(self, image: 'Tensor', text: 'Tensor', text_atts: 'Tensor') ->Tensor:
        multimodal_embeds = self.model_with_similarity.albef_model.multimodal_encoder(text, text_atts, image)
        score = self.itm_head(multimodal_embeds[:, 0, :])[:, 1]
        return score

    def _eval_forward(self, input_type: 'str', image: 'Optional[Tensor]', text: 'Optional[Tensor]', text_atts: 'Optional[Tensor]') ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if input_type == 'image':
            assert image is not None, 'image input tensor cannot be None'
            return self._encode_image(image)
        elif input_type == 'text':
            assert text is not None and text_atts is not None, 'text and text attention mask cannot be None'
            return self._encode_text(text, text_atts)
        elif input_type == 'multimodal':
            assert image is not None and text is not None and text_atts is not None, 'image embeddings, text embeddings, and text attention mask cannot be None'
            return self._image_text_matching_score(image, text, text_atts)
        else:
            raise ValueError('input_type must be image, text, or multimodal')

    def forward(self, image: 'Optional[Tensor]'=None, text: 'Optional[Tensor]'=None, text_atts: 'Optional[Tensor]'=None, idx: 'Optional[Tensor]'=None, alpha: 'Optional[Tensor]'=0.0, input_type: 'Optional[str]'=None, is_train: 'Optional[bool]'=True) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(image, text, text_atts, idx, alpha)
        else:
            return self._eval_forward(input_type, image, text, text_atts)


class CNNEncoder(nn.Module):
    """A CNN encoder.

    Stacks n layers of (Conv2d, MaxPool2d, BatchNorm2d), where n is determined
    by the length of the input args.

    Args:
        input_dims (List[int]): List of input dimensions.
        output_dims (List[int]): List of output dimensions. Should match
            input_dims offset by one.
        kernel_sizes (List[int]): Kernel sizes for convolutions. Should match
            the sizes of cnn_input_dims and cnn_output_dims.

    Inputs:
        x (Tensor): Tensor containing a batch of images.
    ​
    """

    def __init__(self, input_dims: 'List[int]', output_dims: 'List[int]', kernel_sizes: 'List[int]'):
        super().__init__()
        conv_layers: 'List[nn.Module]' = []
        assert len(input_dims) == len(output_dims) and len(output_dims) == len(kernel_sizes), 'input_dims, output_dims, and kernel_sizes should all have the same length'
        assert input_dims[1:] == output_dims[:-1], 'output_dims should match input_dims offset by one'
        for in_channels, out_channels, kernel_size in zip(input_dims, output_dims, kernel_sizes):
            padding_size = kernel_size // 2
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding_size)
            max_pool2d = nn.MaxPool2d(2, stride=2)
            batch_norm_2d = nn.BatchNorm2d(out_channels)
            conv_layers.append(nn.Sequential(conv, nn.LeakyReLU(), max_pool2d, batch_norm_2d))
        conv_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*conv_layers)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.cnn(x)


class LSTMEncoder(nn.Module):
    """An LSTM encoder. Stacks an LSTM on an embedding layer.

    Args:
        vocab_size (int): The size of the vocab for embeddings.
        embedding_dim (int): The size of each embedding vector.
        input_size (int): The number of features in the LSTM input.
        hidden_size (int): The number of features in the hidden state.
        bidirectional (bool): Whether to use bidirectional LSTM.
        batch_first (bool): Whether to provide batches as (batch, seq, feature)
            or (seq, batch, feature).

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(self, vocab_size: 'int', embedding_dim: 'int', input_size: 'int', hidden_size: 'int', bidirectional: 'bool', batch_first: 'bool'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, batch_first=batch_first)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        _, x = self.lstm(self.embedding(x))
        x = x[0].transpose(0, 1)
        assert x.size(1) == 2, 'hidden state (final) should have 1st dim as 2'
        x = torch.cat([x[:, 0, :], x[:, 1, :]], dim=-1)
        return x


CKPT_KEY = 'flava_full'


class DalleConv2d(nn.Module):

    def __init__(self, n_in: 'int', n_out: 'int', kw: 'int') ->None:
        super().__init__()
        w = torch.empty((n_out, n_in, kw, kw), dtype=torch.float32)
        w.normal_(std=1 / math.sqrt(n_in * kw ** 2))
        b = torch.zeros((n_out,), dtype=torch.float32)
        self.w, self.b = nn.Parameter(w), nn.Parameter(b)
        self.kw = kw

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return nn.functional.conv2d(x, self.w, self.b, padding=(self.kw - 1) // 2)


class DalleEncoderBlock(nn.Module):

    def __init__(self, n_in: 'int', n_out: 'int', n_layers: 'int') ->None:
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1 / n_layers ** 2
        self.id_path = DalleConv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([('relu_1', nn.ReLU()), ('conv_1', DalleConv2d(n_in, n_hid, 3)), ('relu_2', nn.ReLU()), ('conv_2', DalleConv2d(n_hid, n_hid, 3)), ('relu_3', nn.ReLU()), ('conv_3', DalleConv2d(n_hid, n_hid, 3)), ('relu_4', nn.ReLU()), ('conv_4', DalleConv2d(n_hid, n_out, 1))]))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


class DalleEncoder(nn.Module):

    def __init__(self, group_count: 'int'=4, n_hid: 'int'=256, n_blk_per_group: 'int'=2, input_channels: 'int'=3, vocab_size: 'int'=8192, **kwargs: Any) ->None:
        super().__init__()
        self.input_channels = input_channels
        n_layers = group_count * n_blk_per_group
        output_conv = DalleConv2d(8 * n_hid, vocab_size, 1)
        self.blocks = nn.Sequential(OrderedDict([('input', DalleConv2d(input_channels, 1 * n_hid, 7)), ('group_1', self._create_group(n_layers, n_blk_per_group, 1 * n_hid, 1 * n_hid)), ('group_2', self._create_group(n_layers, n_blk_per_group, 1 * n_hid, 2 * n_hid)), ('group_3', self._create_group(n_layers, n_blk_per_group, 2 * n_hid, 4 * n_hid)), ('group_4', self._create_group(n_layers, n_blk_per_group, 4 * n_hid, 8 * n_hid, use_pool=False)), ('output', nn.Sequential(OrderedDict([('relu', nn.ReLU()), ('conv', output_conv)])))]))

    def _create_group(self, n_layers: 'int', n_blk_per_group: 'int', n_in: 'int', n_hid: 'int', use_pool: 'bool'=True) ->nn.Module:
        make_blk = partial(DalleEncoderBlock, n_layers=n_layers)
        blk_range = range(n_blk_per_group)
        blocks: 'OrderedDict[str, nn.Module]' = OrderedDict()
        for i in blk_range:
            if i == 0:
                blocks[f'block_{i + 1}'] = make_blk(n_in, n_hid)
            else:
                blocks[f'block_{i + 1}'] = make_blk(n_hid, n_hid)
        if use_pool:
            blocks['pool'] = nn.MaxPool2d(kernel_size=2)
        return nn.Sequential(blocks)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        return self.blocks(x)


class DalleVAEEncoder(nn.Module):

    def __init__(self, image_size: 'Union[int, Tuple[int, int]]'=112, pretrained: 'bool'=True):
        super().__init__()
        self.image_size = image_size
        self.encoder = DalleEncoder()
        if pretrained:
            self.load_model()

    def load_model(self) ->Any:
        encoder_state_dict = torch.hub.load_state_dict_from_url('https://cdn.openai.com/dall-e/encoder.pkl')
        self.encoder.load_state_dict(encoder_state_dict.state_dict())
        return self.state_dict()

    def get_codebook_indices(self, images: 'Tensor') ->Tensor:
        z_logits = self.encoder(images)
        return torch.argmax(z_logits, axis=1)

    def get_codebook_probs(self, images: 'Tensor') ->Tensor:
        z_logits = self.encoder(images)
        return nn.Softmax(dim=1)(z_logits)

    def forward(self, img_seq_prob: 'Tensor') ->Tensor:
        return self.get_codebook_indices(img_seq_prob)


class ModelOutput(OrderedDict):

    def keys(self) ->Any:
        for field in fields(self):
            yield field.name

    def __getitem__(self, key: 'Any') ->Any:
        return getattr(self, key)

    def __iter__(self) ->Any:
        yield from self.keys()

    def values(self) ->Any:
        for field in fields(self):
            yield getattr(self, field.name)

    def items(self) ->Any:
        for field in fields(self):
            yield field.name, getattr(self, field.name)


class BackpropType(Enum):
    """
    How to backpropagate gradients during all-gather op. GLOBAL will backpropagate
    to all workers, LOCAL to only the current worker, and NONE will not backpropagate
    at all.
    """
    GLOBAL = 0
    LOCAL = 1
    NONE = 2


def get_rank() ->int:
    """get rank util for distributed training"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def gather_tensor(tensor: 'Tensor', backprop_type: 'BackpropType'=BackpropType.GLOBAL) ->List[Tensor]:
    """Gathers a tensor across all GPUs.

    Args:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_type (BackpropType): whether to backpropagate gradients to all
            workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
            Default: BackpropType.GLOBAL

    Returns:
        List[Tensor]: List of gathered tensors across all GPUs.
    """
    world_size = torch.distributed.get_world_size()
    if backprop_type == BackpropType.GLOBAL:
        return all_gather_with_backprop(tensor)
    else:
        tensor_all_gpus = [torch.zeros_like(tensor) for _ in range(world_size)]
        all_gather_no_backprop(tensor_all_gpus, tensor)
        if backprop_type == BackpropType.LOCAL:
            tensor_all_gpus[get_rank()] = tensor
        return tensor_all_gpus


def _gather_embeddings_and_labels(embeddings_a: 'Tensor', embeddings_b: 'Tensor', backprop_type: 'BackpropType'=BackpropType.GLOBAL) ->Tuple[Tensor, Tensor, Tensor]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        labels = torch.arange(embeddings_a.size(0), device=embeddings_a.device)
        return embeddings_a, embeddings_b, labels
    embeddings_a_all_gpus = gather_tensor(embeddings_a, backprop_type)
    embeddings_b_all_gpus = gather_tensor(embeddings_b, backprop_type)
    local_batch_size = embeddings_a.size(0)
    labels = local_batch_size * torch.distributed.get_rank() + torch.arange(local_batch_size, device=embeddings_a.device)
    return torch.cat(embeddings_a_all_gpus), torch.cat(embeddings_b_all_gpus), labels


class Pooler(nn.Module):

    def __init__(self, hidden_size: 'int'=768, **kwargs: Any):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: 'Tensor') ->Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TwoWayHead(nn.Module):

    def __init__(self, hidden_size: 'int'=768, **kwargs: Any):
        super().__init__()
        self.seq_relationship = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output: 'Tensor') ->Tensor:
        return self.seq_relationship(pooled_output)


def assert_labels_are_present(labels: 'Optional[Tensor]', category: 'str'='labels') ->None:
    assert labels is not None, f'Model is in training model but {category} are not passed'


class Fp32LayerNorm(nn.LayerNorm):

    def __init__(self, *args: Any, **kwargs: Any) ->None:
        super().__init__(*args, **kwargs)

    def forward(self, x: 'Tensor') ->Tensor:
        output = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(x)


class MaskedPredictionHead(nn.Module):

    def __init__(self, hidden_size: 'int'=768, vocab_size: 'int'=30522, transform_act_fn: 'Callable[[Tensor], Tensor]'=nn.functional.gelu, layer_norm_eps: 'float'=1e-05, use_fp32_layer_norm: 'bool'=True, **kwargs: Any):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm: 'nn.LayerNorm'
        if use_fp32_layer_norm:
            self.layer_norm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states: 'Tensor') ->Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


FLAVA_FOR_PRETRAINED_MAPPING = {CKPT_KEY: 'https://download.pytorch.org/models/multimodal/flava/flava_for_pretraining_unified_text_encoder.pt'}


FLAVAOutput = namedtuple('FLAVAOutput', ['image', 'image_masked', 'text', 'text_masked', 'multimodal', 'multimodal_masked', 'projected_image_embeddings', 'projected_text_embeddings'], defaults=(None, None, None, None, None, None, None, None))


class TransformerOutput(NamedTuple):
    last_hidden_state: 'Optional[Tensor]' = None
    pooler_output: 'Optional[Tensor]' = None
    hidden_states: 'Optional[List[Tensor]]' = None
    attentions: 'Optional[List[Tensor]]' = None
    image_labels: 'Optional[Tensor]' = None
    current_key_values: 'Optional[List[Tuple[Tensor, Tensor]]]' = None


class FLAVAModel(nn.Module):

    def __init__(self, image_encoder: 'nn.Module', text_encoder: 'nn.Module', mm_encoder: 'nn.Module', image_to_mm_projection: 'nn.Module', text_to_mm_projection: 'nn.Module', text_projection: 'nn.Module', image_projection: 'nn.Module', **kwargs: Any) ->None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.mm_encoder = mm_encoder
        self.image_to_mm_projection = image_to_mm_projection
        self.text_to_mm_projection = text_to_mm_projection
        self.text_projection = text_projection
        self.image_projection = image_projection

    def forward(self, image: 'Optional[Tensor]'=None, text: 'Optional[Tensor]'=None, image_patches_mask: 'Optional[Tensor]'=None, text_masked: 'Optional[Tensor]'=None, required_embedding: 'Optional[EMBEDDING_OPTIONS]'=None, skip_unmasked_mm_encoder: 'bool'=True) ->FLAVAOutput:
        if required_embedding is None:
            if image is not None and text is not None:
                required_embedding = 'mm'
            elif image is not None:
                required_embedding = 'image'
            else:
                required_embedding = 'text'
        image_encoding_out = self._encode_data_to_embeddings(image, required_embedding, ['image', 'mm'], partial(self.encode_image, projection=True))
        if len(image_encoding_out) == 2:
            image_outputs, projected_image_embeddings = image_encoding_out[0], image_encoding_out[1]
        else:
            image_outputs = image_encoding_out
            projected_image_embeddings = None
        text_encoding_out = self._encode_data_to_embeddings(text, required_embedding, ['text', 'mm'], partial(self.encode_text, projection=True))
        if len(text_encoding_out) == 2:
            text_outputs, projected_text_embeddings = text_encoding_out[0], text_encoding_out[1]
        else:
            text_outputs = text_encoding_out
            projected_text_embeddings = None
        image_masked_outputs = self._encode_data_to_embeddings(image, required_embedding, ['image', 'mm'], partial(self.encode_image, image_patches_mask=image_patches_mask))
        assert type(image_masked_outputs) == TransformerOutput
        text_masked_outputs = self._encode_data_to_embeddings(text_masked, required_embedding, ['text', 'mm'], self.encode_text)
        assert type(text_masked_outputs) == TransformerOutput
        multimodal_outputs = TransformerOutput()
        multimodal_masked_outputs = TransformerOutput()
        if required_embedding == 'mm':
            if not skip_unmasked_mm_encoder:
                multimodal_outputs = self.encode_mm(image_outputs.hidden_states[-1] if image_outputs.hidden_states else None, text_outputs.hidden_states[-1] if text_outputs.hidden_states else None)
            multimodal_masked_outputs = self.encode_mm(image_masked_outputs.hidden_states[-1] if image_masked_outputs.hidden_states else None, text_masked_outputs.hidden_states[-1] if text_masked_outputs.hidden_states else None)
        return FLAVAOutput(image=image_outputs, image_masked=image_masked_outputs, text=text_outputs, text_masked=text_masked_outputs, multimodal=multimodal_outputs, multimodal_masked=multimodal_masked_outputs, projected_image_embeddings=projected_image_embeddings, projected_text_embeddings=projected_text_embeddings)

    def encode_image(self, image: 'Tensor', image_patches_mask: 'Optional[Tensor]'=None, projection: 'bool'=False) ->Union[Tuple[TransformerOutput, Tensor], Optional[TransformerOutput]]:
        if image_patches_mask is not None:
            encoded_image = self.image_encoder(image, image_patches_mask)
        else:
            encoded_image = self.image_encoder(image)
        if projection:
            projected_embeddings = self.image_projection(encoded_image.last_hidden_state[:, 0, :])
            return encoded_image, projected_embeddings
        return encoded_image

    def encode_text(self, text: 'Tensor', text_mask: 'Optional[Tensor]'=None, projection: 'bool'=False) ->Union[Tuple[TransformerOutput, Tensor], Optional[TransformerOutput]]:
        encoded_text = self.text_encoder(input_ids=text, attention_mask=text_mask, return_attn_weights=True, return_hidden_states=True)
        if projection:
            projected_embeddings = self.text_projection(encoded_text.last_hidden_state[:, 0, :])
            return encoded_text, projected_embeddings
        return encoded_text

    def _encode_data_to_embeddings(self, data: 'Optional[Tensor]', selected_head_encoder: 'EMBEDDING_OPTIONS', encoder_options: 'List[EMBEDDING_OPTIONS]', encode_callable: 'Callable[..., Union[Tuple[TransformerOutput, Tensor], Optional[TransformerOutput]]]') ->Union[Tuple[TransformerOutput, Tensor], Optional[TransformerOutput]]:
        output: 'Union[Tuple[TransformerOutput, Tensor], TransformerOutput]' = TransformerOutput()
        if data is not None and selected_head_encoder in encoder_options:
            output = encode_callable(data)
        return output

    def encode_mm(self, image_embedding: 'Tensor', text_embedding: 'Tensor') ->TransformerOutput:
        if image_embedding is None or text_embedding is None:
            return TransformerOutput()
        image_embedding = self.image_to_mm_projection(image_embedding)
        text_embedding = self.text_to_mm_projection(text_embedding)
        fused_state = torch.cat([image_embedding, text_embedding], dim=1)
        return self.mm_encoder(fused_state)


FLAVA_MODEL_MAPPING = {CKPT_KEY: 'https://download.pytorch.org/models/multimodal/flava/flava_model_unified_text_encoder.pt'}


class PatchEmbeddingsOutput(NamedTuple):
    embeddings: 'Tensor'
    random_mask: 'Optional[Tensor]' = None
    ids_restore: 'Optional[Tensor]' = None


class RandomMaskingOutput(NamedTuple):
    x_masked: 'Tensor'
    mask: 'Tensor'
    ids_restore: 'Tensor'
    ids_keep: 'Tensor'


def random_masking(x: 'torch.Tensor', mask_ratio: 'float') ->RandomMaskingOutput:
    """
    Original paper: https://arxiv.org/pdf/2111.06377.pdf
    OSS implementation: https://github.com/facebookresearch/mae/blob/main/models_mae.py#L123
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    n, l, d = x.shape
    len_keep = int(l * (1 - mask_ratio))
    noise = torch.rand(n, l, device=x.device)
    assert len_keep >= 1, 'must keep at least 1 patch'
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
    mask = torch.ones([n, l], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return RandomMaskingOutput(x_masked=x_masked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep)


def _random_masking_1d(x: 'Tensor', mask_ratio: 'float', num_patches_h: 'int', num_patches_w: 'int') ->Tuple[Tensor, int]:
    n, _, _, d = x.shape
    len_keep = int(num_patches_h * (1 - mask_ratio))
    noise = torch.rand(n, num_patches_h, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num_patches_w, d)
    x_masked = torch.gather(x, dim=1, index=index)
    return x_masked, len_keep


def random_masking_2d(x: 'torch.Tensor', mask_ratio_h: 'float', mask_ratio_w: 'float', num_patches_h: 'int', num_patches_w: 'int') ->Tensor:
    """
    Perform 2d masking as described in audio mae paper https://arxiv.org/pdf/2207.06405.pdf
    Code adapted from https://github.com/facebookresearch/AudioMAE/blob/main/models_vit.py#L88
    Args:
        x: Input tensor containing patches of shape bsz x seq_len x embed_dim
        mask_ratio_h: masking ratio for height dimension
        mask_ratio_w: masking ratio for width dimension
        num_patches_h: number of patches in height dimension
        num_patches_w: number of patches in width dimension
    """
    n, _, d = x.shape
    x = x.reshape(n, num_patches_h, num_patches_w, d)
    x_masked, len_keep_h = _random_masking_1d(x, mask_ratio_h, num_patches_h, num_patches_w)
    x_masked = x_masked.transpose(1, 2)
    x_masked, len_keep_w = _random_masking_1d(x_masked, mask_ratio_w, num_patches_w, len_keep_h)
    x_masked = x_masked.transpose(1, 2)
    x_masked = x_masked.reshape(n, len_keep_h * len_keep_w, d)
    return x_masked


class PatchEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings for vision transformer
    Args:
        image_size (Union[int, Tuple[int, int]]): Size of the input. If set to an int, we assume a square input. Defaults to 224.
        patch_size (int): Size of the patch. Defaults to 16.
        num_channels (int): Number of channels in the input. Defaults to 3.
        hidden_size (int): Embedding dimension of the output. Defaults to 768.
        hidden_dropout_prob (float): Dropout probability applied after adding position embeddings. Defaults to 0.0.
        use_image_masking (bool): Whether to use image masking or not. Defaults to False.
        patch_drop_rate (Optional[Union[float, Tuple[float, float]]]): ratio of patches to be masked out
        or dropped if single float. Set to tuple if dimension wise masking is needed i.e. 2d masking
        after adding position embeddings as described in https://arxiv.org/pdf/2212.00794.pdf. Defaults to None.
        include_cls_embed (bool): Whether to include the [CLS] token embedding. Defaults to True.
    """

    def __init__(self, image_size: 'Union[int, Tuple[int, int]]'=224, patch_size: 'int'=16, num_channels: 'int'=3, hidden_size: 'int'=768, hidden_dropout_prob: 'float'=0.0, use_image_masking: 'bool'=False, patch_drop_rate: 'Optional[Union[float, Tuple[float, float]]]'=None, include_cls_embed: 'bool'=True) ->None:
        super().__init__()
        if isinstance(image_size, int):
            image_size = image_size, image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError('Image size needs to be divisible by patch size')
        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        num_patches = self.num_patches_h * self.num_patches_w
        self.include_cls_embed = include_cls_embed
        if self.include_cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            num_patches = num_patches + 1
        self.conv_projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self._init_conv_weights()
        self.image_size: 'Tuple[int, int]' = image_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)
        if use_image_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.mask_token = None
        self.patch_drop_rate = patch_drop_rate

    def _init_conv_weights(self) ->None:
        fan_in = self.conv_projection.in_channels * self.conv_projection.kernel_size[0] * self.conv_projection.kernel_size[1]
        nn.init.trunc_normal_(self.conv_projection.weight, std=math.sqrt(1 / fan_in))
        assert self.conv_projection.bias is not None
        nn.init.zeros_(self.conv_projection.bias)

    def forward(self, pixel_values: 'Tensor', image_patches_mask: 'Optional[Tensor]'=None) ->PatchEmbeddingsOutput:
        batch_size, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(f"Input image size ({height}*{width}) doesn't match image size                 {self.image_size[0]}*{self.image_size[1]} expected by model")
        embeddings = self.conv_projection(pixel_values).flatten(2).transpose(1, 2)
        _, seq_len, _ = embeddings.size()
        if image_patches_mask is not None:
            if self.mask_token is not None:
                mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
                w = image_patches_mask.unsqueeze(-1).type_as(mask_tokens)
                embeddings = embeddings * (1 - w) + mask_tokens * w
            else:
                warnings.warn('image_patches_mask passed but use_image_masking in init was false. Ignoring.')
        if self.include_cls_embed:
            embeddings = embeddings + self.position_embeddings[:, 1:, :]
        else:
            embeddings = embeddings + self.position_embeddings
        random_mask = None
        ids_restore = None
        if self.training and self.patch_drop_rate is not None:
            if isinstance(self.patch_drop_rate, Iterable):
                embeddings = random_masking_2d(embeddings, mask_ratio_h=self.patch_drop_rate[0], mask_ratio_w=self.patch_drop_rate[1], num_patches_h=self.num_patches_h, num_patches_w=self.num_patches_w)
            else:
                embeddings, random_mask, ids_restore, _ = random_masking(embeddings, mask_ratio=self.patch_drop_rate)
        if self.include_cls_embed:
            assert hasattr(self, 'cls_token'), 'CLS token must be defined to include CLS embedding'
            cls_token = self.cls_token + self.position_embeddings[:, :1, :]
            cls_tokens = cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = self.dropout(embeddings)
        return PatchEmbeddingsOutput(embeddings=embeddings, random_mask=random_mask, ids_restore=ids_restore)


class ImageEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self, image_size: 'int'=224, patch_size: 'int'=16, num_channels: 'int'=3, hidden_size: 'int'=768, hidden_dropout_prob: 'float'=0.0, use_image_masking: 'bool'=True) ->None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.patch_embeddings = PatchEmbeddings(image_size=image_size, patch_size=patch_size, num_channels=num_channels, embed_dim=hidden_size)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)
        if use_image_masking:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.mask_token = None

    def interpolate_pos_encoding(self, embeddings: 'Tensor', height: 'int', width: 'int') ->Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        npatch = embeddings.shape[1] - 1
        n = self.position_embeddings.shape[1] - 1
        if npatch == n and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.patch_embeddings.patch_size[0]
        w0 = width // self.patch_embeddings.patch_size[1]
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(n)), int(math.sqrt(n)), dim).permute(0, 3, 1, 2), scale_factor=(h0 / math.sqrt(n), w0 / math.sqrt(n)), mode='bicubic', align_corners=False)
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: 'Tensor', image_patches_mask: 'Optional[Tensor]'=None, interpolate_pos_encoding: 'bool'=False) ->Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        _, seq_len, _ = embeddings.size()
        if image_patches_mask is not None:
            if self.mask_token is not None:
                mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
                w = image_patches_mask.unsqueeze(-1).type_as(mask_tokens)
                embeddings = embeddings * (1 - w) + mask_tokens * w
            else:
                warnings.warn('image_patches_mask passed but use_image_masking in init was false. Ignoring.')
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


def init_transformer_weights(module: 'nn.Module', initializer_range: 'float') ->None:
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class ImageTransformer(nn.Module):

    def __init__(self, embeddings: 'nn.Module', encoder: 'nn.Module', layernorm: 'nn.Module', pooler: 'nn.Module', weight_init_fn: 'Optional[Callable]'=None, initializer_range: 'float'=0.02, **kwargs: Any) ->None:
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        if weight_init_fn is None:
            weight_init_fn = partial(init_transformer_weights, initializer_range=initializer_range)
        self.apply(weight_init_fn)

    def forward(self, pixel_values: 'Optional[Tensor]'=None, image_patches_mask: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None) ->TransformerOutput:
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        embedding_output = self.embeddings(pixel_values, image_patches_mask=image_patches_mask)
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask, return_attn_weights=True, return_hidden_states=True)
        sequence_output = encoder_output.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return TransformerOutput(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_output.hidden_states, attentions=encoder_output.attentions)


def flava_image_encoder(hidden_size: 'int'=768, num_attention_heads: 'int'=12, num_hidden_layers: 'int'=12, use_image_masking: 'bool'=False, dropout: 'float'=0.0, intermediate_size: 'int'=3072, intermediate_activation: 'Callable[..., nn.Module]'=nn.GELU, layer_norm_eps: 'float'=1e-12, image_size: 'int'=224, patch_size: 'int'=16, num_channels: 'int'=3) ->ImageTransformer:
    embeddings = ImageEmbeddings(image_size=image_size, patch_size=patch_size, num_channels=num_channels, hidden_size=hidden_size, hidden_dropout_prob=dropout, use_image_masking=use_image_masking)
    encoder = TransformerEncoder(n_layer=num_hidden_layers, d_model=hidden_size, n_head=num_attention_heads, dim_feedforward=intermediate_size, activation=intermediate_activation, layer_norm_eps=layer_norm_eps, dropout=dropout, norm_first=True)
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)
    return ImageTransformer(embeddings=embeddings, encoder=encoder, layernorm=layernorm, pooler=pooler)


class FLAVATransformerWithoutEmbeddings(nn.Module):

    def __init__(self, encoder: 'nn.Module', layernorm: 'nn.Module', pooler: 'nn.Module', hidden_size: 'int'=768, weight_init_fn: 'Optional[Callable]'=None, initializer_range: 'float'=0.02, use_cls_token: 'bool'=True, **kwargs: Any):
        super().__init__()
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.cls_token = None
        if weight_init_fn is None:
            weight_init_fn = partial(init_transformer_weights, initializer_range=initializer_range)
        self.apply(weight_init_fn)

    def forward(self, hidden_states: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None) ->TransformerOutput:
        if hidden_states is None:
            raise ValueError('You have to specify hidden_states')
        if self.cls_token is not None:
            batch_size = hidden_states.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
        encoder_output = self.encoder(hidden_states, attention_mask=attention_mask, return_hidden_states=True, return_attn_weights=True)
        sequence_output = encoder_output.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return TransformerOutput(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_output.hidden_states, attentions=encoder_output.attentions)


def flava_multimodal_encoder(hidden_size: 'int'=768, num_attention_heads: 'int'=12, num_hidden_layers: 'int'=12, dropout: 'float'=0.0, intermediate_size: 'int'=3072, intermediate_activation: 'Callable[..., nn.Module]'=nn.GELU, layer_norm_eps: 'float'=1e-12) ->FLAVATransformerWithoutEmbeddings:
    encoder = TransformerEncoder(n_layer=num_hidden_layers, d_model=hidden_size, n_head=num_attention_heads, dim_feedforward=intermediate_size, activation=intermediate_activation, layer_norm_eps=layer_norm_eps, dropout=dropout, norm_first=True)
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)
    return FLAVATransformerWithoutEmbeddings(encoder=encoder, layernorm=layernorm, pooler=pooler, hidden_size=hidden_size)


class BERTTextEmbeddings(nn.Module):
    """Construct word, position, and token type embeddings following BERT, similar to HuggingFace BertEmbeddings

    Attributes:
        hidden_size (int): size of embedding space. Default is 768.
        vocab_size (int): size of vocabulary. Default is 30522.
        pad_token_id (int): id used for padding token. Default is 0.
        max_position_embeddings (int): the highest position id number, or max sequence length. Default is 512.
        type_vocab_size (int): the highest token type id number. Default is 2.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        dropout (float): dropout probability after all embeddings and layernorm
        offset_pos_ids (bool): if True, shift position ids by one for the padding token. Used in RoBERTa.
            Default is False.

    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere
    """

    def __init__(self, hidden_size: 'int'=768, vocab_size: 'int'=30522, pad_token_id: 'int'=0, max_position_embeddings: 'int'=512, type_vocab_size: 'int'=2, layer_norm_eps: 'float'=1e-12, dropout: 'float'=0.0, offset_pos_ids: 'bool'=False) ->None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id
        self.offset_pos_ids = offset_pos_ids

    def create_position_ids_from_input_ids(self, input_ids: 'Tensor') ->Tensor:
        """
        Replace non-padding symbols with their position numbers.
        Position numbers begin at pad_token_id+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Inputs: input_ids (Tensor): Tensor from which to create position IDs.
                pad_token_id (int): Padding index
                    (determines starting point of position IDs).
        """
        mask = input_ids.ne(self.pad_token_id).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + self.pad_token_id

    def forward(self, input_ids: 'Optional[Tensor]'=None, token_type_ids: 'Optional[Tensor]'=None, position_ids: 'Optional[Tensor]'=None, inputs_embeds: 'Optional[Tensor]'=None) ->Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError('input_ids or inputs_embeds must not be None')
        seq_length = input_shape[1]
        if position_ids is None:
            if self.offset_pos_ids:
                position_ids = self.create_position_ids_from_input_ids(input_ids)
            else:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def get_extended_attention_mask(attention_mask: 'Tensor') ->Tensor:
    """Makes attention masks broadcastable along head and sequence dimensions.

    Accepting two types of attention masks:
        - Causal: masks that prevent attending to future positions of dimensions
            ``(batch_size, query_seq_len, key_seq_len)``
        - Padding: masks that prevent attending to token paddings of dimensions
            ``(batch_size, seq_len)``

    Args:
        attention_mask (Tensor):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.

    Returns:
        extended_attention_mask (Tensor):
            The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
    """
    if attention_mask.dim() == 4:
        extended_attention_mask = attention_mask
    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError('Wrong shape for attention_mask (shape {})'.format(attention_mask.shape))
    extended_attention_mask = extended_attention_mask
    return extended_attention_mask


class BERTTextEncoder(nn.Module):
    """
    General text transformer encoder with embeddings, following BERT.
    Can be constructed with any user-provided embeddings and encoder.

    Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L870

    Attributes:
        embeddings (nn.Module): Module that projects text token ids into embeddings.
            See :py:class: `torchmultimodal.modules.layers.text_embedding.BERTTextEmbeddings` for interface.
        encoder (nn.Module): Module for transformer encoder. See :py:class:
            `torchmultimodal.modules.layers.transformer.TransformerEncoder` for interface.
        layernorm (nn.Module, optional): Module for layernorm to be applied after encoder. Defaults to ``None``.
        pooler (nn.Module, optional): Module for pooler to be applied after layernorm. Defaults to ``None``.
        weight_init_fn (Callable, optional): function for custom weight initialization of both the transformer
            encoder and embeddings. See :py:func: `torchmultimodal.models.flava.transformer.init_transformer_weights`
            as an example. Defaults to ``None``.

    Args:
        input_ids (Tensor, optional): Tensor of input vocab token ids of shape [batch, seq_len].
        attention_mask (Tensor, optional): Tensor indicating which tokens to attend to, shape [batch, seq_len]
        token_type_ids (Tensor, optional): Tensor of input token type ids of shape [batch, seq_len]. In BERT,
            used to indicate whether a word is in sentence A or B for next sentence prediction
        position_ids (Tensor, optional): Tensor of input position ids of shape [batch, seq_len]
        inputs_embeds (Tensor, optional): Tensor of input embeddings of shape [batch, hidden_size],
            if embeddings are calculated elsewhere

    Raises:
        ValueError: if input_ids and inputs_embeds are both ``None``.
    """

    def __init__(self, embeddings: 'nn.Module', encoder: 'nn.Module', layernorm: 'Optional[nn.Module]'=None, pooler: 'Optional[nn.Module]'=None, weight_init_fn: 'Optional[Callable]'=None) ->None:
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.layernorm = layernorm
        self.pooler = pooler
        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(self, input_ids: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, token_type_ids: 'Optional[Tensor]'=None, position_ids: 'Optional[Tensor]'=None, inputs_embeds: 'Optional[Tensor]'=None, return_attn_weights: 'bool'=False, return_hidden_states: 'bool'=False) ->TransformerOutput:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError('input_ids or inputs_embeds must not be None')
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            if hasattr(self.embeddings, 'pad_token_id'):
                attention_mask[input_ids == self.embeddings.pad_token_id] = 0
        attention_mask = get_extended_attention_mask(attention_mask)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask, return_attn_weights=return_attn_weights, return_hidden_states=return_hidden_states)
        last_hidden_state = encoder_output.last_hidden_state
        pooled_output = encoder_output.pooler_output
        if self.layernorm:
            last_hidden_state = self.layernorm(last_hidden_state)
        if self.pooler:
            pooled_output = self.pooler(last_hidden_state)
        return TransformerOutput(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_output.hidden_states, attentions=encoder_output.attentions)


def flava_text_encoder(num_hidden_layers: 'int'=12, hidden_size: 'int'=768, num_attention_heads: 'int'=12, intermediate_size: 'int'=3072, intermediate_activation: 'Callable[..., nn.Module]'=nn.GELU, layer_norm_eps: 'float'=1e-12, dropout: 'float'=0.0, vocab_size: 'int'=30522, pad_token_id: 'int'=0, type_vocab_size: 'int'=2, max_position_embeddings: 'int'=512, initializer_range: 'float'=0.02) ->BERTTextEncoder:
    embeddings = BERTTextEmbeddings(hidden_size=hidden_size, vocab_size=vocab_size, pad_token_id=pad_token_id, type_vocab_size=type_vocab_size, max_position_embeddings=max_position_embeddings, layer_norm_eps=layer_norm_eps, dropout=dropout)
    encoder = TransformerEncoder(n_layer=num_hidden_layers, d_model=hidden_size, n_head=num_attention_heads, dim_feedforward=intermediate_size, activation=intermediate_activation, layer_norm_eps=layer_norm_eps, dropout=dropout, norm_first=True)
    layernorm = Fp32LayerNorm(hidden_size, eps=layer_norm_eps)
    pooler = Pooler(hidden_size=hidden_size)
    weight_init_fn = partial(init_transformer_weights, initializer_range=initializer_range)
    return BERTTextEncoder(embeddings=embeddings, encoder=encoder, layernorm=layernorm, pooler=pooler, weight_init_fn=weight_init_fn)


def load_module_from_url(model: 'nn.Module', url: 'str', strict: 'bool'=True, progress: 'bool'=True) ->None:
    local_path = _PATH_MANAGER.get_local_path(url)
    if not torch.cuda.is_available():
        state_dict = torch.load(local_path, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(local_path)
    model.load_state_dict(state_dict, strict=strict)


def flava_model(image_hidden_size: 'int'=768, image_num_attention_heads: 'int'=12, image_num_hidden_layers: 'int'=12, image_dropout: 'float'=0.0, image_intermediate_size: 'int'=3072, image_intermediate_activation: 'Callable[..., nn.Module]'=nn.GELU, image_layer_norm_eps: 'float'=1e-12, use_image_masking: 'bool'=True, image_size: 'int'=224, patch_size: 'int'=16, num_channels: 'int'=3, text_hidden_size: 'int'=768, text_num_attention_heads: 'int'=12, text_num_hidden_layers: 'int'=12, text_dropout: 'float'=0.0, text_intermediate_size: 'int'=3072, text_intermediate_activation: 'Callable[..., nn.Module]'=nn.GELU, text_layer_norm_eps: 'float'=1e-12, vocab_size: 'int'=30522, pad_token_id: 'int'=0, type_vocab_size: 'int'=2, max_position_embeddings: 'int'=512, multimodal_hidden_size: 'int'=768, multimodal_num_attention_heads: 'int'=12, multimodal_num_hidden_layers: 'int'=6, multimodal_dropout: 'float'=0.0, multimodal_intermediate_size: 'int'=3072, multimodal_intermediate_activation: 'Callable[..., nn.Module]'=nn.GELU, multimodal_layer_norm_eps: 'float'=1e-12, text_and_image_proj_size: 'int'=768, pretrained: 'bool'=False, **kwargs: Any) ->FLAVAModel:
    image_encoder = flava_image_encoder(hidden_size=image_hidden_size, num_attention_heads=image_num_attention_heads, num_hidden_layers=image_num_hidden_layers, use_image_masking=use_image_masking, dropout=image_dropout, intermediate_size=image_intermediate_size, intermediate_activation=image_intermediate_activation, layer_norm_eps=image_layer_norm_eps, image_size=image_size, patch_size=patch_size, num_channels=num_channels)
    text_encoder = flava_text_encoder(hidden_size=text_hidden_size, num_attention_heads=text_num_attention_heads, num_hidden_layers=text_num_hidden_layers, dropout=text_dropout, intermediate_size=text_intermediate_size, intermediate_activation=text_intermediate_activation, layer_norm_eps=text_layer_norm_eps, vocab_size=vocab_size, pad_token_id=pad_token_id, type_vocab_size=type_vocab_size, max_position_embeddings=max_position_embeddings)
    mm_encoder = flava_multimodal_encoder(hidden_size=multimodal_hidden_size, num_attention_heads=multimodal_num_attention_heads, num_hidden_layers=multimodal_num_hidden_layers, dropout=multimodal_dropout, intermediate_size=multimodal_intermediate_size, intermediate_activation=multimodal_intermediate_activation, layer_norm_eps=multimodal_layer_norm_eps)
    image_to_mm_projection = nn.Linear(image_hidden_size, multimodal_hidden_size)
    text_to_mm_projection = nn.Linear(text_hidden_size, multimodal_hidden_size)
    image_projection = nn.Linear(image_hidden_size, text_and_image_proj_size)
    text_projection = nn.Linear(text_hidden_size, text_and_image_proj_size)
    flava = FLAVAModel(image_encoder=image_encoder, text_encoder=text_encoder, mm_encoder=mm_encoder, image_to_mm_projection=image_to_mm_projection, text_to_mm_projection=text_to_mm_projection, text_projection=text_projection, image_projection=image_projection)
    if pretrained:
        load_module_from_url(flava, FLAVA_MODEL_MAPPING[CKPT_KEY])
    return flava


class FLAVAPreTrainModule(nn.Module):

    def __init__(self, use_bf16: 'bool'=True, **flava_pretraining_kwargs: Any):
        super().__init__()
        self.model = flava_model_for_pretraining(**flava_pretraining_kwargs)
        self.use_bf16 = use_bf16

    def forward(self, batch, action=None):
        if action == 'encode_text':
            return self.model.encode_text(batch)
        elif action == 'encode_image':
            return self.model.encode_image(batch)
        if 'image' in batch and ('text' in batch or 'text_masked' in batch):
            required_embedding = 'mm'
        elif 'image' in batch:
            required_embedding = 'image'
        elif 'text' in batch or 'text_masked' in batch:
            required_embedding = 'text'
        else:
            raise RuntimeError("Batch needs to have either or both 'image' and 'text'.")
        output = self.model(image=batch.get('image'), image_for_codebook=batch.get('image_for_codebook'), image_patches_mask=batch.get('image_patches_mask'), text=batch.get('text'), text_masked=batch.get('text_masked'), mlm_labels=batch.get('mlm_labels'), itm_labels=batch.get('itm_labels'), required_embedding=required_embedding)
        return output

    def encode_text(self, *args, **kwargs):
        return self.model.encode_text(*args, **kwargs)


class MDETRLoss(nn.Module):

    def __init__(self, soft_token_loss: 'Callable[..., Tensor]', box_losses: 'Callable[..., BoxLosses]', contrastive_alignment_loss: 'Optional[nn.Module]'=None, vqa_losses: 'Optional[Iterable[Callable[..., Dict[str, Tensor]]]]'=None):
        super().__init__()
        self.soft_token_loss = soft_token_loss
        self.box_losses = box_losses
        self.contrastive_alignment_loss = contrastive_alignment_loss
        self.vqa_losses = vqa_losses

    def get_average_num_boxes_across_workers(self, num_boxes: 'Tensor'):
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return torch.clamp(num_boxes, min=1).item()
        torch.distributed.all_reduce(num_boxes)
        num_boxes_all_workers = torch.clamp(num_boxes / torch.distributed.get_world_size(), min=1).item()
        return num_boxes_all_workers

    def total_losses_with_weights(self, loss_dict: 'Dict[str, Tensor]', weight_dict: 'Optional[Dict[str, float]]'=None) ->torch.Tensor:
        for k in weight_dict.keys():
            if k not in loss_dict.keys():
                raise ValueError(f'Weight dict contains invalid key {k}')
        return sum([(weight_dict[k] * loss_dict[k]) for k in weight_dict.keys()])

    def forward(self, pred_logits: 'Tensor', pred_boxes: 'Tensor', targets: 'List[Dict[str, Any]]', positive_map, indices: 'List[Tuple[Tensor, Tensor]]', contrastive_query_embeddings: 'Optional[Tensor]'=None, contrastive_token_embeddings: 'Optional[Tensor]'=None, tokenized: 'Optional[Any]'=None, vqa_preds: 'Optional[Dict[str, Tensor]]'=None, vqa_labels: 'Optional[Dict[str, Tensor]]'=None, vqa_masks: 'Optional[Dict[str, Tensor]]'=None, weight_dict: 'Optional[Dict[str, float]]'=None) ->Dict[str, Tensor]:
        target_boxes = [t['boxes'] for t in targets]
        target_tokens = [t['tokens_positive'] for t in targets]
        n_target_boxes = [len(t) for t in target_boxes]
        num_boxes = sum(n_target_boxes)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)
        num_boxes_all_workers = self.get_average_num_boxes_across_workers(num_boxes)
        self.pred_logits = pred_logits
        self.n_target_boxes = n_target_boxes
        self.positive_map = positive_map
        self.indices = indices
        self.num_boxes_all_workers = num_boxes_all_workers
        soft_token_loss = self.soft_token_loss(pred_logits, n_target_boxes, positive_map, indices, num_boxes_all_workers)
        box_losses = self.box_losses(pred_boxes, target_boxes, indices, num_boxes_all_workers)
        loss_dict = {'soft_token_loss': soft_token_loss, 'l1_loss': box_losses.l1_loss, 'giou_loss': box_losses.giou_loss}
        if self.contrastive_alignment_loss is not None:
            if contrastive_query_embeddings is None or contrastive_token_embeddings is None or tokenized is None:
                raise ValueError('For contrastive alignment loss must pass contrastive query/token embeddings and tokenized text')
            contrastive_alignment_loss = self.contrastive_alignment_loss(contrastive_query_embeddings, contrastive_token_embeddings, target_tokens, indices, num_boxes_all_workers, tokenized)
            loss_dict.update(contrastive_alignment_loss=contrastive_alignment_loss)
        if self.vqa_losses is not None:
            if vqa_preds is None or vqa_labels is None:
                raise ValueError('For QA loss qa_preds and qa_labels must not be None')
            for vqa_loss in self.vqa_losses:
                loss_dict.update(vqa_loss(vqa_preds, vqa_labels, vqa_masks))
        if weight_dict is not None:
            total_loss = self.total_losses_with_weights(loss_dict, weight_dict)
            loss_dict.update(total_loss=total_loss)
        return loss_dict


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this,
    in general, there are more predictions than targets. In this case, we do a 1-to-1
    matching of the best predictions, while the others are un-matched (and thus treated
    as non-objects). This implementation is based on the MDETR repo:
    https://github.com/ashkamath/mdetr/blob/main/models/matcher.py#L13

    Attributes:
        cost_class (float): Relative weight of the classification error in the
            matching cost. Default: ``1``
        cost_bbox (float): Relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: ``1``
        cost_giou (float): Relative weight of the giou loss of the bounding box in
            the matching cost. Default: ``1``


    Args:
        pred_logits (Tensor): Classification logits.
            Size: (batch_size, num_queries, num_classes)
        pred_boxes (Tensor): Predicted box coordinates.
            Size: (batch_size, num_queries, 4)
        target_boxes_per_sample (List[Tensor]): A list of target bounding boxes.
            Length = batch_size.
            Each element is a tensor of size (n_boxes_for_sample, 4).
        positive_map (Tensor): :math:`	ext{positive_map}[i,j] = 1` when box i maps to class j.
            Size: (total_boxes, num_classes) where total_boxes is the sum of
            n_boxes_for_sample over every sample in the batch.

    Returns:
        A list of size batch_size, containing tuples of ``(index_i, index_j)`` where:
            - ``index_i`` is the indices of the selected predictions (in order)
            - ``index_j`` is the indices of the corresponding selected targets
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

    Raises:
        ValueError: If all costs are zero or first dim of target boxes and positive map
            don't match or classification cost and bbox cost shapes don't match.
    """

    def __init__(self, cost_class: 'float'=1, cost_bbox: 'float'=5, cost_giou: 'float'=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError('At least one cost must be nonzero')

    @torch.no_grad()
    def forward(self, pred_logits: 'Tensor', pred_boxes: 'Tensor', target_boxes_per_sample: 'List[Tensor]', positive_map: 'Tensor') ->List[Tuple[Tensor, Tensor]]:
        bs, num_queries = pred_logits.shape[:2]
        target_boxes = torch.cat(target_boxes_per_sample)
        out_prob = F.softmax(pred_logits.flatten(0, 1), dim=-1)
        out_bbox = pred_boxes.flatten(0, 1)
        if target_boxes.size(0) != positive_map.size(0):
            raise ValueError('Total of target boxes should match first dim of positive map')
        cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)
        cost_bbox = torch.cdist(out_bbox, target_boxes, p=1)
        if cost_class.shape != cost_bbox.shape:
            raise ValueError(f"""
            Classification and bounding box cost shapes do not match.
            Classification cost shape: {cost_class.shape},
            Bounding box cost shape: {cost_bbox.shape}
            """)
        cost_giou = -generalized_box_iou(box_convert(out_bbox, in_fmt='cxcywh', out_fmt='xyxy'), box_convert(target_boxes, in_fmt='cxcywh', out_fmt='xyxy'))
        cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()
        sizes = [x.size(0) for x in target_boxes_per_sample]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class TextTokenizer(nn.Module):
    """Converts between text and tokens / embedings

    Wrapper around the tokenizer to be consistent with the API required by
    :py:class:`torchmultimodal.models.video_gpt.gpt.MultimodalGPT`. It also contains the
    embedding layer to enable lookup by token ids.
    """

    def __init__(self, context_len: 'int', d_model: 'int', tokenizer: 'nn.Module') ->None:
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.encode('[PAD]')[0]
        self.vocab_size = self.tokenizer.vocab_size
        self.context_len = context_len
        self.num_text_tokens = self.vocab_size + context_len
        self.embedding = nn.Embedding(self.num_text_tokens, d_model)

    def text_to_tokens(self, sentences: 'List[str]') ->Tensor:
        """Pads the sentences to be of equal lengths"""
        tokens = [self.tokenizer.encode(sentence.strip().lower() + ' [SEP]') for sentence in sentences]
        token_ids = [t[:self.context_len] for t in tokens]
        for i, t in enumerate(token_ids):
            t += [self.pad_id] * (self.context_len - len(t))
            token_ids[i] = t
        return torch.Tensor(token_ids).type(torch.int64)

    def encode(self, sentences: 'List[str]', device: 'str') ->Tensor:
        """Encodes sentences to token ids"""
        token_ids = self.text_to_tokens(sentences)
        unique_pad_ids = torch.arange(self.context_len, device=device) + self.vocab_size
        token_ids = torch.where(token_ids == self.pad_id, unique_pad_ids, token_ids)
        return token_ids

    def _filter_token_ids(self, token_ids: 'List[int]') ->List[Optional[int]]:
        """Filters out token ids out side of vocab"""
        return [token_id for token_id in token_ids if token_id > 0 and token_id <= self.vocab_size]

    def decode(self, token_ids: 'Tensor') ->List[str]:
        """Decodes token ids back to sentences"""
        sentences = []
        for _token_ids in token_ids:
            _token_ids = self._filter_token_ids(_token_ids.tolist())
            sentence = self.tokenizer.decode(_token_ids)
            sentences.append(sentence)
        return sentences

    def lookup(self, token_ids: 'Tensor') ->Tensor:
        return self.embedding(token_ids)


class TextEncoder(nn.Module):
    """Encode tokenized text to the last hidden state representation of the CLS token using
        DistilBERT. DistilBERT prepends a CLS (classification) token to every text so the
        token's hidden state represents the entire text.

    Adapted from MUGEN's text encoder
        (https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/videoclip/modules.py)

    Args:
        model_config (Optional[Dict[str, Any]]): model config for DistilBERT.
            Defaults to ``None``, indicating the default DistilBERT config.
        padding_value (int): value that was used to pad the input text.
            Defaults to ``0``, Hugging Face's BERT pad token.

    Inputs:
        input_ids (Tensor): tensor of (batch, text_length) tokenized text

    Returns:
        Tensor: encoded text with dimensions (batch, ``model_config.dim``).
            Default ``model_config.dim`` is ``768``.
    """

    def __init__(self, model_config: 'Optional[Dict[str, Any]]'=None, padding_value: 'int'=0):
        super().__init__()
        self.padding_value = padding_value
        self.target_token_idx = 0
        distilbert_config = DistilBertConfig(**model_config) if model_config else DistilBertConfig()
        self.model = DistilBertModel(config=distilbert_config)
        self.out_dim = self.model.config.dim

    def build_attention_mask(self, input_ids: 'torch.Tensor') ->torch.Tensor:
        return input_ids != self.padding_value

    def forward(self, input_ids: 'torch.Tensor') ->torch.Tensor:
        attention_mask = self.build_attention_mask(input_ids)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


def scaled_dot_product_attention(q: 'Tensor', k: 'Tensor', v: 'Tensor', attention_mask: 'Optional[Tensor]'=None, head_mask: 'Optional[Tensor]'=None, attn_dropout: 'float'=0.0) ->Tuple[Tensor, Tensor]:
    """Similar to PyTorch Core's _scaled_dot_product_attention but generalized
    to handle n-dimensional input tokens (images, video) and support multihead.
    Computes attention as described in Attention Is All You Need (Vaswani et al. 2017)

    Args:
        q (Tensor): Query of shape ``(b, h, d1, ..., dn, dim_qk)`` or ``(b, h, seq_len, dim_qk)`` where
            ``h`` is number of attention heads, ``d1, ..., dn`` are latent dimensions and ``dim_qk` is
            the embedding dim of the query tensor.
        k (Tensor): Key of shape ``(b, h, d1', ...., dn', dim_qk)`` or ``(b, h, seq_len', dim_qk)`` where
            ``h`` is the number of attention heads, ``d1', ..., dn'` are latent dimensions and ``dim_qk``
            is the key embedding dim aligned with query embedding dim,
            see :class:`~torchmultimodal.modules.layers.attention.MultiHeadAttention`.
        v (Tensor): Value of shape ``(b, h, d1', ..., dn', dim_v)`` or ``(b, h, seq_len', dim_v)`` where
            ``h`` is the number of attention heads, ``d1', ..., dn'`` are latent dimensions and ``dim_v``
            is the embedding dim of the value tensor.
        attention_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
            Contains 1s for positions to attend to and 0s for masked positions. Applied before softmax.
        head_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
            Contains 1s for positions to attend to and 0s for masked positions.
            Applied after dropout, before matrix multiplication with values.
        attn_dropout (float): Probability of dropout after softmax. Default is ``0.0``.

    Returns:
        A tuple of output tensor and attention probabilities.
    """
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / torch.sqrt(torch.tensor(q.shape[-1]))
    if attention_mask is not None:
        attn = attn.masked_fill(attention_mask == 0, float('-inf'))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn)
    attn = F.dropout(attn, p=attn_dropout)
    if head_mask is not None:
        attn = attn * head_mask
    a = torch.matmul(attn, v)
    return a, attn


def shift_dim(x: 'Tensor', src_dim: 'int'=-1, dest_dim: 'int'=-1, make_contiguous: 'bool'=True) ->Tensor:
    """Permutes tensor x by moving src_dim to dest_dim.
    i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)

    Code taken from VideoGPT
    https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/utils.py

    Args:
        x (Tensor): input Tensor you want to permute
        src_dim (int, optional): the axis you want to move. Negative indexing supported. Defaults to -1.
        dest_dim (int, optional): the axis you want to move to. Negative indexing supported. Defaults to -1.
        make_contiguous (bool, optional): if you want the output tensor to be contiguous in memory. Defaults to True.

    Returns:
        Tensor: permuted Tensor
    """
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim
    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims
    dims = [i for i in range(n_dims) if i != src_dim]
    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


class AxialAttention(nn.Module):
    """Computes attention over a single axis of the input. Other dims are flattened into the batch dimension.

    Args:
        axial_dim (int): Dimension to compute attention on, indexed by input dimensions
            (i.e., ``0`` for first input dimension, ``1`` for second).
        attn_dropout (float): Probability of dropout after softmax. Default is ``0.0``.
    """

    def __init__(self, axial_dim: 'int', attn_dropout: 'float'=0.0) ->None:
        super().__init__()
        self.axial_dim = axial_dim + 2
        self.attn_dropout = attn_dropout

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', attention_mask: 'Optional[Tensor]'=None, head_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
        """
        Args:
            q (Tensor): Query input of shape ``(b, h, d1, ..., dn, dim_q)`` where ``h`` is the number of
                attention heads, ``(d1, ..., dn)`` are the query latent dimensions and ``dim_q`` is the dimension
                of the query embeddings.
            k, v (Tensor): Key/value input of shape ``(b, h, d1', ..., dn', dim_kv)`` where ``h`` is the number
                of attention heads, ``(d1', ..., dn')`` are the key/value latent dimensions and ``dim_kv`` is
                the dimension of the key/value embeddings.
            attention_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)`` where ``q_dn`` is
                the dimension of the axis to compute attention on of the query and ``k_dn`` that of the key.
                Contains 1s for positions to attend to and 0s for masked positions.
            head_mask (Tensor, optional): Tensor of shape ``(b, h, d1, ..., q_dn, k_dn)``.
                Contains 1s for positions to attend to and 0s for masked positions.

        Returns:
            A tuple of output tensor and attention probabilities.
        """
        if self.axial_dim >= len(q.shape) - 1:
            raise ValueError('axial dim does not match input shape')
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)
        out, attn_probs = scaled_dot_product_attention(q, k, v, attention_mask=attention_mask, head_mask=head_mask, attn_dropout=self.attn_dropout if self.training else 0.0)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out, attn_probs


class SelfAttention(nn.Module):
    """Computes attention over the entire n-dimensional input.

    Args:
        attn_dropout (float, optional): Probability of dropout after softmax. Default is ``0.0``.
    """

    def __init__(self, attn_dropout: 'float'=0.0) ->None:
        super().__init__()
        self.attn_dropout = attn_dropout

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', attention_mask: 'Optional[Tensor]'=None, head_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
        """
        Args:
            q (Tensor): Query input of shape ``(b, h, d1, ..., dn, dim_q)`` where ``h`` is the number of
                attention heads, ``(d1, ..., dn)`` are the query latent dimensions and ``dim_q`` is the dimension
                of the query embeddings.
            k, v (Tensor): Key/value input of shape ``(b, h, d1', ..., dn', dim_kv)`` where ``h`` is the number
                of attention heads, ``(d1', ..., dn')`` are the key/value latent dimensions and ``dim_kv`` is
                the dimension of the key/value embeddings.
            attention_mask (Tensor, optional): Tensor of shape ``(b, h, q_dn, k_dn)`` where ``q_dn`` is the
                dimension of the flattened query input along its latent dimensions and ``k_dn`` that of the
                flattened key input. Contains 1s for positions to attend to and 0s for masked positions.
            head_mask (Tensor, optional): Tensor of shape ``(b, h, q_dn, k_dn)``.
                Contains 1s for positions to attend to and 0s for masked positions.

        Returns:
            A tuple of output tensor and attention probabilities.
        """
        _, _, *shape, _ = q.shape
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)
        out, attn_probs = scaled_dot_product_attention(q, k, v, attention_mask=attention_mask, head_mask=head_mask, attn_dropout=self.attn_dropout if self.training else 0.0)
        return out.unflatten(2, shape), attn_probs


def merge_multihead(x: 'Tensor') ->Tensor:
    """Moves head dim back to original location and concatenates heads
    (b, n_head, d1, ..., dn, c // n_head) -> (b, d1, ..., dn, c)"""
    return shift_dim(x, 1, -2).flatten(start_dim=-2)


def split_multihead(x: 'Tensor', n_head: 'int') ->Tensor:
    """Splits channel dimension of input tensor of size (b, d1, ..., dn, c)
    into multiple heads, (b, n_head, d1, ..., dn, c // n_head)"""
    x = x.unflatten(-1, (n_head, -1))
    x = shift_dim(x, -2, 1)
    return x


class MultiHeadAttention(nn.Module):
    """Computes multihead attention with flexible attention mechanism and caching for fast decoding.

    Multihead attention linearly projects and divides queries, keys, and values into
    multiple 'heads'. This enables the computation of attention multiple times in
    parallel, creating more varied representations and allows the model to jointly
    attend to information from different representation subspaces at different positions,
    as described in `"Attention Is All You Need (Vaswani et al. 2017)"<https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:
        dim_q (int): Dimensionality of query input. Also the embedding dimension of the model.
        dim_kv (int): Dimensionality of key/value input. Projects to the embedding dimension of the model, ``dim_q``.
        n_head (int): Number of attention heads.
        attn_module (nn.Module): Module of attention mechanism to use. Default is ``SelfAttention``.
            See :class:`~torchmultimodal.modules.layers.attention.SelfAttention` for API details.
        add_bias (bool): Whether to add bias to the q, k, v, linear layers or not. Default is ``True``.

    Attributes:
        cache (Dict[str, Tensor]): Dictionary that stores past key/value vectors.

    Raises:
        ValueError: When ``dim_q`` or ``dim_kv`` is not divisible by ``n_head``.
    """

    def __init__(self, dim_q: 'int', dim_kv: 'int', n_head: 'int', attn_module: 'nn.Module'=SelfAttention(), add_bias: 'bool'=True) ->None:
        super().__init__()
        if dim_q % n_head != 0 or dim_kv % n_head != 0:
            raise ValueError('The hidden size of q, k, v must be a multiple of the number of attention heads.')
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.n_head = n_head
        self.query = nn.Linear(dim_q, dim_q, bias=add_bias)
        self.key = nn.Linear(dim_kv, dim_q, bias=add_bias)
        self.value = nn.Linear(dim_kv, dim_q, bias=add_bias)
        self.output = nn.Linear(dim_q, dim_q, bias=True)
        self.attn = attn_module
        self.cache: 'Optional[Dict[str, Tensor]]' = None

    def forward(self, q: 'Tensor', kv: 'Optional[Tensor]'=None, return_attn_weights: 'bool'=False, use_cache: 'bool'=False, causal: 'bool'=False, **attn_kwargs: Any) ->Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            q (Tensor): Query of shape ``(b, d1, ..., dn, dim_q)`` or ``(b, seq_len, dim_q)``
                (for autoregressive decoding it's typical to pass in flattened tensors).
            kv (Tensor, optional): Key (and value) of shape ``(b, d1', ..., dn', dim_kv)`` or
                ``(b, seq_len', dim_kv)``. If this argument is specified, cross-attention will be applied.
                Default is ``None``.
            use_cache (bool): If ``True``, caches past ``k`` and ``v`` tensors for faster decoding.
                If ``False``, recomputes ``k`` and ``v`` for each decoding step. Default is ``False``.
            causal (bool): Whether to use causal attention or not. Default is ``False``.

        Returns:
            * If ``return_attn_weights`` is ``True``: A tuple of output tensor and attention probabilities.
            * If ``return_attn_weights`` is ``False``: A single output tensor.
        """
        k = v = q if kv is None else kv
        q = split_multihead(self.query(q), self.n_head)
        if causal or not self.cache:
            k = split_multihead(self.key(k), self.n_head)
            v = split_multihead(self.value(v), self.n_head)
        if use_cache:
            if not self.cache:
                self.cache = dict(k=k.clone(), v=v.clone())
            else:
                if causal:
                    k_, v_ = self.cache['k'], self.cache['v']
                    self.cache['k'] = torch.cat([k_, k], dim=2)
                    self.cache['v'] = torch.cat([v_, v], dim=2)
                k, v = self.cache['k'], self.cache['v']
        attn_out = self.attn(q, k, v, **attn_kwargs)
        attn_probs = None
        if isinstance(attn_out, tuple):
            attn_out, attn_probs = attn_out
        a = merge_multihead(attn_out)
        a = self.output(a)
        if return_attn_weights:
            return a, attn_probs
        else:
            return a


class AxialAttentionBlock(nn.Module):
    """Computes multihead axial attention across all dims of the input.

    Axial attention is an alternative to standard full attention, where instead
    of computing attention across the entire flattened input, you compute it for
    each dimension. To capture the global context that full attention does, stacking
    multiple axial attention layers will allow information to propagate among the
    multiple dimensions of the input. This enables attention calculations on high
    dimensional inputs (images, videos) where full attention would be computationally
    expensive and unfeasible. For more details, see `"Axial Attention in
    Multidimensional Transformers (Ho et al. 2019)"<https://arxiv.org/pdf/1912.12180.pdf>`_
    and `"CCNet: Criss-Cross Attention for Semantic Segmentation (Huang et al. 2019)
    "<https://arxiv.org/pdf/1811.11721.pdf>`_.

    Follows implementation by VideoGPT:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        n_dims (int): Dimensionality of input data, not including batch or embedding dims.
        qkv_dim (int): Dimensionality of query/key/value embedding vectors.
        n_head (int): Number of heads in multihead attention. Must divide into ``qkv_dim``
            evenly.
    """

    def __init__(self, n_dims: 'int', qkv_dim: 'int', n_head: 'int') ->None:
        super().__init__()
        self.qkv_dim = qkv_dim
        self.mha_attns = nn.ModuleList([MultiHeadAttention(dim_q=qkv_dim, dim_kv=qkv_dim, n_head=n_head, attn_module=AxialAttention(d), add_bias=False) for d in range(n_dims)])

    def forward(self, x: 'Tensor') ->Tensor:
        n_channel = x.shape[1]
        if n_channel != self.qkv_dim:
            raise ValueError(f'Input channel dimension is {n_channel}, expected {self.qkv_dim}')
        h = shift_dim(x, 1, -1)
        attn_out = torch.zeros_like(h)
        for mha_attn in self.mha_attns:
            attn_out += mha_attn(h, causal=False)
        h = attn_out
        h = shift_dim(h, -1, 1)
        return h


def calculate_same_padding(kernel_size: 'Union[int, Tuple[int, ...]]', stride: 'Union[int, Tuple[int, ...]]', input_shape: 'Union[Size, Tuple[int, ...]]') ->Tuple:
    """Calculates padding amount on each dimension based on given kernel size and stride.

    Pads to match the 'SAME' padding in Keras, i.e., with a stride of 1 output is guaranteed
    to have the same shape as input, with stride 2 the dimensions of output are halved. If
    stride does not divide into input evenly, then output = ceil(input / stride), following
    the TensorFlow implementation explained here:
    https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        kernel_size (int or Tuple[int, ...]): Size of convolutional kernel.
        stride (int or Tuple[int, ...]): Stride amount of kernel.
        input_shape (Size or Tuple[int, ...]): Shape of input, without batch or channel dimension.

    Returns:
        A tuple of the padding amount in a tuple of tuples for each dimension.
    """
    n_dims = len(input_shape)
    if isinstance(kernel_size, int):
        kernel_size = tuple(repeat(kernel_size, n_dims))
    if isinstance(stride, int):
        stride = tuple(repeat(stride, n_dims))
    if not len(kernel_size) == len(stride) == len(input_shape):
        raise ValueError('dims for kernel, stride, and input must match')
    total_pad = []
    for k, s, d in zip(kernel_size, stride, input_shape):
        if d % s == 0:
            pad = max(k - s, 0)
        else:
            pad = max(k - d % s, 0)
        total_pad.append(pad)
    pad_input = []
    for p in total_pad[::-1]:
        pad_input.append(p // 2 + p % 2)
        pad_input.append(p // 2)
    pad_input = tuple(pad_input)
    return pad_input


class SamePadConv3d(nn.Module):
    """Performs a same padded convolution on a 3D input.

    This maintains input shape with unit stride, and divides input dims by non-unit stride.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (int): Number of channels in input, same as ``nn.Conv3d``.
        out_channels (int): Number of channels for output, same as ``nn.Conv3d``.
        kernel_size (int or Tuple[int, int, int]): Size of convolutional filter, same as ``nn.Conv3d``.
        stride (int or Tuple[int, int, int], optional): Stride for convolution, same as ``nn.Conv3d``.
        bias (bool, optional): If ``True`` use a bias for convolutional layer or not,
            same as ``nn.Conv3d``. Defaults to ``True``.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int, int, int]]', stride: 'Union[int, Tuple[int, int, int]]'=1, bias: 'bool'=True, **kwargs: Any) ->None:
        super().__init__()
        self.pad_input: 'Tuple' = None
        self.kernel_size = kernel_size
        self.stride = stride
        if 'padding' in kwargs:
            warnings.warn('Padding was specified but will not be used in favor of same padding,                 use Conv3d directly for custom padding')
        self.conv = nn.Conv3d(in_channels, out_channels, self.kernel_size, stride=self.stride, bias=bias, **kwargs)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        if self.pad_input is None:
            self.pad_input = calculate_same_padding(self.kernel_size, self.stride, x.shape[2:])
        return self.conv(F.pad(x, self.pad_input))


class AttentionResidualBlock(nn.Module):
    """Residual block with axial attention.

    Implements the component as proposed in `"VideoGPT: Video Generation using VQ-VAE and
    Transformers (Yan et al. 2022)"<https://arxiv.org/pdf/2104.10157.pdf>`_.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        hidden_dim (int, optional): Size of channel dim of input. Default is ``240``.
        n_head (int, optional): Number of heads in multihead attention. Must divide into hidden_dim evenly.
            Default is ``2``.

    Raises:
        ValueError: If ``hidden_dim`` is less than ``2``.
    """

    def __init__(self, hidden_dim: 'int'=240, n_head: 'int'=2) ->None:
        super().__init__()
        if hidden_dim < 2:
            raise ValueError('hidden dim must be at least 2')
        self.block = nn.Sequential(nn.BatchNorm3d(hidden_dim), nn.ReLU(), SamePadConv3d(hidden_dim, hidden_dim // 2, 3, bias=False), nn.BatchNorm3d(hidden_dim // 2), nn.ReLU(), SamePadConv3d(hidden_dim // 2, hidden_dim, 1, bias=False), nn.BatchNorm3d(hidden_dim), nn.ReLU(), AxialAttentionBlock(3, hidden_dim, n_head))

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        return x + self.block(x)


class VideoEncoder(nn.Module):
    """Encoder for Video VQVAE.

    Stacks specified number of ``SamePadConv3d`` layers
    followed by a stack of ``AttentionResidualBlocks`` and a final ``SamePadConv3d``
    layer before the codebook. The residual blocks use Axial Attention to enhance
    representations of video data without significantly increasing computational
    cost.

    Follows VideoGPT's implementation:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channel_dims (Tuple[int, ...]): Input channel dimension for each layer in conv stack.
        kernel_sizes (Tuple[Tuple[int, int, int], ...]): Kernel sizes for each layer in conv stack.
        strides (Tuple[Tuple[int, int, int], ...]): Strides for each layer in conv stack.
        output_dim (int): Size of hidden dimension of final output.
        n_res_layers (int, optional): Number of ``AttentionResidualBlocks`` to include. Default is ``4``.
        attn_hidden_dim (int, optional): Size of hidden dimension in attention block. Default is ``240``.
        kwargs (Any): Keyword arguments to be passed into ``SamePadConv3d`` and used by ``nn.Conv3d``.

    Raises:
        ValueError: If the lengths of ``in_channel_dims``, ``kernel_sizes``, and ``strides`` are not
            all equivalent.
    """

    def __init__(self, in_channel_dims: 'Tuple[int, ...]', kernel_sizes: 'Tuple[Tuple[int, int, int], ...]', strides: 'Tuple[Tuple[int, int, int], ...]', output_dim: 'int', n_res_layers: 'int'=4, attn_hidden_dim: 'int'=240, **kwargs: Any):
        super().__init__()
        assert_equal_lengths(in_channel_dims, kernel_sizes, strides, msg='in_channel_dims, kernel_sizes, and strides must be same length.')
        convolutions: 'List[nn.Module]' = []
        n_conv_layers = len(in_channel_dims)
        for i in range(n_conv_layers):
            in_channel = in_channel_dims[i]
            out_channel = in_channel_dims[i + 1] if i < n_conv_layers - 1 else attn_hidden_dim
            kernel = kernel_sizes[i]
            stride = strides[i]
            convolutions.append(SamePadConv3d(in_channel, out_channel, kernel, stride, bias=True, **kwargs))
            if i < n_conv_layers - 1:
                convolutions.append(nn.ReLU())
        self.convs = nn.Sequential(*convolutions)
        self.res_stack = nn.Sequential(*[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)], nn.BatchNorm3d(attn_hidden_dim), nn.ReLU())
        self.conv_out = SamePadConv3d(attn_hidden_dim, output_dim, kernel_size=1, stride=1)

    def get_latent_shape(self, input_shape: 'Union[Tuple, Size]') ->Tuple:
        """Return shape of encoder output based on number of downsampling conv layers"""
        latent_shape = list(input_shape)
        for layer in self.convs:
            if isinstance(layer, SamePadConv3d):
                latent_shape = [(latent_shape[dim] // layer.conv.stride[dim]) for dim in range(len(input_shape))]
        return tuple(latent_shape)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): Input video data with shape ``(b, c, d1, d2, d3)``.
        """
        in_channel = x.shape[1]
        if in_channel != self.convs[0].conv.in_channels:
            raise ValueError(f'expected input channel dim to be {self.convs[0].conv.in_channels}, but got {in_channel}')
        h = self.convs(x)
        h = self.res_stack(h)
        h = self.conv_out(h)
        return h


class Projection(nn.Module):
    """Project embeddings to a fixed dimension by adding the hidden-layer output and final output of a MLP.

    Args:
        in_dim (int): dimension of input.
        out_dim (int): dimension of output.
            Defaults to ``256``, the value used by MUGEN.
        dropout_prob (float): dropout probability.
            Defaults to ``0.1``, the value used by MUGEN.

    Inputs:
        x (Tensor): embeddings (batch, dim_in)

    Returns:
        Tensor: projected embeddings (batch, dim_out)

    """

    def __init__(self, in_dim, out_dim=256, dropout_prob=0.1) ->None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.drop = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.gelu(embed1)
        embed2 = self.linear2(embed2)
        embed2 = self.drop(embed2)
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class RandomMixup(torch.torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: 'int', p: 'float'=0.5, alpha: 'float'=1.0, inplace: 'bool'=False) ->None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f'Please provide a valid positive value for the num_classes. Got num_classes={num_classes}')
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: 'Tensor', target: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (..., H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim < 4:
            raise ValueError(f'Batch ndim should be 4. Got {batch.ndim}')
        if target.ndim != 1:
            raise ValueError(f'Target ndim should be 1. Got {target.ndim}')
        if not batch.is_floating_point():
            raise TypeError(f'Batch dtype should be a float tensor. Got {batch.dtype}.')
        if target.dtype != torch.int64:
            raise TypeError(f'Target dtype should be torch.int64. Got {target.dtype}')
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        if torch.rand(1).item() >= self.p:
            return batch, target
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        return batch, target

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}(num_classes={self.num_classes}, p={self.p}, alpha={self.alpha}, inplace={self.inplace})'
        return s


class RandomCutmix(torch.torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: 'int', p: 'float'=0.5, alpha: 'float'=1.0, inplace: 'bool'=False) ->None:
        super().__init__()
        if num_classes < 1:
            raise ValueError('Please provide a valid positive value for the num_classes.')
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: 'Tensor', target: 'Tensor') ->Tuple[Tensor, Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (..., H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim < 4:
            raise ValueError(f'Batch ndim should be 4. Got {batch.ndim}')
        if target.ndim != 1:
            raise ValueError(f'Target ndim should be 1. Got {target.ndim}')
        if not batch.is_floating_point():
            raise TypeError(f'Batch dtype should be a float tensor. Got {batch.dtype}.')
        if target.dtype != torch.int64:
            raise TypeError(f'Target dtype should be torch.int64. Got {target.dtype}')
        if not self.inplace:
            batch = batch.clone()
            target = target.clone()
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
        if torch.rand(1).item() >= self.p:
            return batch, target
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        h, w = batch.shape[-2:]
        r_x = torch.randint(w, (1,))
        r_y = torch.randint(h, (1,))
        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * w)
        r_h_half = int(r * h)
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=w))
        y2 = int(torch.clamp(r_y + r_h_half, max=h))
        batch[..., y1:y2, x1:x2] = batch_rolled[..., y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (w * h))
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        return batch, target

    def __repr__(self) ->str:
        s = f'{self.__class__.__name__}(num_classes={self.num_classes}, p={self.p}, alpha={self.alpha}, inplace={self.inplace})'
        return s


class Unsqueeze(torch.torch.nn.Module):

    def __init__(self, pos=0):
        super().__init__()
        self.pos = pos

    def forward(self, x):
        return x.unsqueeze(self.pos)


class ConvertTCHWtoCTHW(torch.nn.Module):
    """Convert tensor from (T, C, H, W) to (C, T, H, W)"""

    def forward(self, vid: 'torch.Tensor') ->torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class DropChannels(torch.nn.Module):
    """
    Drops Channels with predefined probability values.
    Pads the dropped channels with `pad_value`.
    Channels can be tied using `tie_channels`
    For example, for RGBD input, RGB can be tied by using `tie_channels=[0,1,2]`.
    In this case, channels [0,1,2] will be dropped all at once or not at all.
    Assumes input is of the form CxHxW or TxCxHxW
    """

    def __init__(self, channel_probs, fill_values, tie_channels=None, all_channel_drop=False):
        """
        channel_probs: List of probabilities
        fill_values: List of values to fill the dropped channels with
        tie_channels: List of indices. Tie dropping of certain channels.
        all_channel_drop: Bool variable to prevent cases where all channels are dropped.
        """
        super().__init__()
        channel_probs = np.array(channel_probs, dtype=np.float32)
        self.channel_probs = channel_probs
        self.fill_values = fill_values
        self.tie_channels = tie_channels
        self.all_channel_drop = all_channel_drop
        if tie_channels is not None:
            tie_probs = [channel_probs[x] for x in tie_channels]
            assert len(set(tie_probs)) == 1, 'All tie_channel probs must be equal'

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        if x.ndim == 3:
            num_channels = x.shape[0]
            channel_index = 0
        elif x.ndim == 4:
            num_channels = x.shape[1]
            channel_index = 1
        else:
            raise ValueError(f'Unexpected number of dims {x.ndim}. Expected 3 or 4.')
        assert num_channels == len(self.channel_probs), f'channel_probs is {len(self.channel_probs)} but got {num_channels} channels'
        to_drop = [(np.random.random() < self.channel_probs[c]) for c in range(num_channels)]
        if self.tie_channels is not None:
            first_drop = to_drop[self.tie_channels[0]]
            for idx in self.tie_channels[1:]:
                to_drop[idx] = first_drop
        if all(to_drop) and self.all_channel_drop is False:
            to_drop = [(False) for _ in range(num_channels)]
        for c in range(num_channels):
            if not to_drop[c]:
                continue
            if channel_index == 0:
                x[c, ...] = self.fill_values[c]
            elif channel_index == 1:
                x[:, c, ...] = self.fill_values[c]
            else:
                raise NotImplementedError()
        return x


class DepthNorm(torch.nn.Module):
    """
    Normalize the depth channel: in an RGBD input of shape (4, H, W),
    only the last channel is modified.
    The depth channel is also clamped at 0.0. The Midas depth prediction
    model outputs inverse depth maps - negative values correspond
    to distances far away so can be clamped at 0.0
    """

    def __init__(self, max_depth: 'float', clamp_max_before_scale: 'bool'=False, min_depth: 'float'=0.01):
        """
        Args:
            max_depth (float): The max value of depth for the dataset
            clamp_max (bool): Whether to clamp to max_depth or to divide by max_depth
        """
        super().__init__()
        if max_depth < 0.0:
            raise ValueError('max_depth must be > 0; got %.2f' % max_depth)
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale
        self.min_depth = min_depth

    def forward(self, image: 'torch.Tensor'):
        c, h, w = image.shape
        if c != 4:
            err_msg = f'This transform is for 4 channel RGBD input only; got {image.shape}'
            raise ValueError(err_msg)
        color_img = image[:3, ...]
        depth_img = image[3:4, ...]
        depth_img = depth_img.clamp(min=self.min_depth)
        if self.clamp_max_before_scale:
            depth_img = depth_img.clamp(max=self.max_depth)
        depth_img /= self.max_depth
        img = torch.cat([color_img, depth_img], dim=0)
        return img


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device='cpu'):

        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        super().__init__(model, device, ema_avg, use_buffers=True)


class DummyADM(nn.Module):

    def __init__(self, variance_flag=False):
        super().__init__()
        self.variance_flag = variance_flag

    def forward(self, x, t, c):
        c_sum = 0
        count = 0
        for i, (k, v) in enumerate(c.items()):
            c_sum += v
            count += 1
        prediction = x + t[..., None, None] + c_sum[..., None, None] + count
        if self.variance_flag:
            variance_value = x
        else:
            variance_value = None
        return DiffusionOutput(prediction=prediction, variance_value=variance_value)


class DummyUNet(nn.Module):

    def __init__(self, learn_variance=True):
        super().__init__()
        self.learn_variance = learn_variance
        channels = 6 if learn_variance else 3
        self.net = nn.Conv2d(3, channels, 3, 1, padding=1)

    def forward(self, x, t, c):
        x = F.tanh(self.net(x))
        var_value = None
        if self.learn_variance:
            x, var_value = x.chunk(2, dim=1)
            var_value = (var_value + 1) / 2
        return DiffusionOutput(prediction=x, variance_value=var_value)


class DummyEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(8, 2, batch_first=True), num_layers=1, norm=nn.LayerNorm(8))

    def forward(self, x):
        return self.transformer(x)[:, 0, :]


class ADMStackModuleType(Enum):
    ResidualBlock = 0
    AttentionBlock = 1
    SimpleBlock = 2


class ADMStack(nn.Module):
    """A container that acts like a ModuleList of ADM blocks and handles passing timestep and
    context embeddings correctly to its children. Usually, blocks such as residual blocks consume
    timestep embeddings, while attention blocks consume optional contextual embeddings in addition
    to the input x. This container allows us to wrap the modules so that they can be stacked in a
    `nn.Sequential`, in order to simplify the code for the `forward` method.

    We have to implement the stack in this way rather than inherting from `nn.ModuleList` to
    avoid FSDP/Activation Checkpointing/PT2 incompatibility issues.

    Code ref: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py#L35
    """

    def __init__(self) ->None:
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self._module_list = nn.ModuleList()
        self._module_types: 'List[ADMStackModuleType]' = []

    def append_attention_block(self, module: 'nn.Module') ->None:
        self._module_list.append(module)
        self._module_types.append(ADMStackModuleType.AttentionBlock)

    def append_residual_block(self, module: 'nn.Module') ->None:
        self._module_list.append(module)
        self._module_types.append(ADMStackModuleType.ResidualBlock)

    def append_simple_block(self, module: 'nn.Module') ->None:
        self._module_list.append(module)
        self._module_types.append(ADMStackModuleType.SimpleBlock)

    def forward(self, x: 'Tensor', residual_conditional_embedding: 'Tensor', attention_conditional_embedding: 'Optional[Union[Tensor, Sequence[Tensor]]]') ->Tensor:
        h = x
        for name, block in zip(self._module_types, self._module_list):
            if name == ADMStackModuleType.ResidualBlock:
                h = block(h, residual_conditional_embedding)
            elif name == ADMStackModuleType.AttentionBlock:
                h = block(h, attention_conditional_embedding)
            else:
                h = block(h)
        return h


class Fp32GroupNorm(nn.GroupNorm):
    """
    GroupNorm that supports mixed-precision / fp16 training by performing normalization
    in fp32 and converting back.

    Code ref:
    https://github.com/facebookresearch/fairseq/blob/0338cdc3094ca7d29ff4d36d64791f7b4e4b5e6e/fairseq/modules/fp32_group_norm.py#L13
    """

    def __init__(self, *args: Any, **kwargs: Any) ->None:
        super().__init__(*args, **kwargs)

    def forward(self, x: 'Tensor') ->Tensor:
        output = nn.functional.group_norm(x.float(), self.num_groups, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(x)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for use of encoding timestep
    in diffusion models. Timestep is mapped onto positions on different
    frequencies of sinusoidal waveforms. Slightly different than the original
    Transformer implementation in paper "Attention Is All You Need".
    Taken from code of original author of DDPM paper, Ho et al. 2020.

    Code ref: https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/nn.py#L90

    Attributes:
        embed_dim (int): dimensionality of position embeddings. Default is 128, from original DDPM.

    Args:
        t (Tensor): Tensor of input timesteps of shape (batch, ).
    """

    def __init__(self, embed_dim: 'int'=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: 'Tensor') ->Tensor:
        half_dim = self.embed_dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.embed_dim % 2 == 1:
            embeddings = nn.functional.pad(embeddings, (0, 1))
        return embeddings


class ADMCrossAttention(nn.Module):
    """Similar to standard cross-attention, except conditioning inputs are passed through a separate projection
    and then concatenated with the key and value vectors before scaled dot product attention.

    Code ref: https://fburl.com/code/rxl1md57

    Attributes:
        dim_qkv (int): embedding dimension of query, key, and value vectors. conditional_embedding is projected into this
            dimension * 2, to account for k and v.
        dim_cond (int, optional): embedding dimension of conditional input. If unspecified, this class becomes standard
            self attention.

    Args:
        q, k, v (Tensor): Query/key/value of shape [b, h, d1, ..., dn, dim_qkv // h] or [b, h, seq_len, dim_qkv //h] where
            h is number of attention heads, d1, ..., dn are spatial dimensions and dim_qkv is
            the embedding dim.
        conditional_embedding (Tensor, Optional): tensor of shape [b, d1, ..., dn, dim_cond] to condition k and v on

    Returns:
        A tensor of shape [b, h, d1, ..., dn, dim_qkv // h] with the output of the attention calculation.
    """

    def __init__(self, dim_qkv: 'int', dim_cond: 'Optional[int]'=None) ->None:
        super().__init__()
        self.dim_qkv = dim_qkv
        self.cond_proj: 'Optional[nn.Module]' = None
        if dim_cond is not None:
            self.cond_proj = nn.Linear(dim_cond, dim_qkv * 2)

    def forward(self, q: 'Tensor', k: 'Tensor', v: 'Tensor', conditional_embedding: 'Optional[Tensor]'=None) ->Tensor:
        _, n_head, *spatial_dims, dim_q = q.shape
        dim_q = dim_q * n_head
        dim_k = k.shape[-1] * n_head
        dim_v = v.shape[-1] * n_head
        if self.dim_qkv != dim_q or self.dim_qkv != dim_k or self.dim_qkv != dim_v:
            raise ValueError(f'The embedding dim of q, k, v does not match expected embedding dim of {self.dim_qkv}.')
        if self.dim_qkv % n_head != 0:
            raise ValueError('The embedding dim of q, k, v must be a multiple of the number of attention heads.')
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)
        if conditional_embedding is not None and self.cond_proj is not None:
            cond = split_multihead(self.cond_proj(conditional_embedding), n_head)
            cond = cond.flatten(start_dim=2, end_dim=-2)
            cond_k, cond_v = cond.split(self.dim_qkv // n_head, dim=-1)
            k = torch.cat([cond_k, k], dim=2)
            v = torch.cat([cond_v, v], dim=2)
        attn = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False)
        attn = attn.unflatten(2, spatial_dims)
        return attn


def adm_attention(num_channels: 'int', dim_cond: 'Optional[int]'=None, num_heads: 'int'=1) ->nn.Module:
    attn = ADMCrossAttention(dim_qkv=num_channels, dim_cond=dim_cond)
    return MultiHeadAttention(dim_q=num_channels, dim_kv=num_channels, n_head=num_heads, attn_module=attn)


class ADMAttentionBlock(nn.Module):
    """Attention block in the ADM net that consists of group norm, multihead attention, and a residual connection.

    Follows the architecture described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233)

    Code ref:
    https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py#L259

    Attributes:
        num_channels (int): channel dim expected in input, determines embedding dim of q, k, v in attention module.
            Needs to be divisible by norm_groups.
        dim_cond (Optional[int]): dimensionality of conditional input for cross attention. If not specified,
            do not use conditional input.
        rescale_skip_connection (bool): whether to rescale skip connection by 1/sqrt(2), as described in "Diffusion
            Models Beat GANs on Image Synthesis" (https://arxiv.org/abs/2105.05233). Defaults to False.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]
        conditional_embedding (Optional[Tensor]): tokens of shape [b, n, dim_cond] where n is the number of tokens.
            If provided, will be passed as cross-attention input to the MultiHeadAttention block. Defaults to None.
    """

    def __init__(self, num_channels: 'int', dim_cond: 'Optional[int]'=None, num_heads: 'int'=1, rescale_skip_connection: 'bool'=False, norm_groups: 'int'=32):
        super().__init__()
        if num_channels % norm_groups != 0:
            raise ValueError('Channel dims need to be divisible by norm_groups')
        self.norm = Fp32GroupNorm(norm_groups, num_channels)
        self.attn = adm_attention(num_channels, dim_cond, num_heads)
        self.rescale_skip_connection = rescale_skip_connection

    def forward(self, x: 'Tensor', conditional_embedding: 'Optional[Tensor]'=None) ->Tensor:
        norm_out = self.norm(x)
        norm_out = shift_dim(norm_out, 1, -1)
        attn_out = self.attn(norm_out, conditional_embedding=conditional_embedding)
        attn_out = shift_dim(attn_out, -1, 1)
        if self.rescale_skip_connection:
            return (x + attn_out) / 1.414
        else:
            return x + attn_out


def adm_attn_block(num_channels: 'int', dim_cond: 'Optional[int]'=None) ->ADMAttentionBlock:
    return ADMAttentionBlock(num_channels=num_channels, dim_cond=dim_cond)


class ResBlock(nn.Module):
    """Residual block in the ADM net. Supports projecting a conditional embedding to add to the hidden state.
    This typically contains the timestep embedding, but can also contain class embedding for classifier free guidance,
    CLIP image embedding and text encoder output for text-to-image generation as in DALL-E 2, or anything you want to
    condition the diffusion model on. If conditional embedding is not passed, the hidden state is simply passed through.

    Follows the architecture described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233) and BigGAN residual blocks (https://arxiv.org/abs/1809.11096).

    Code ref:
    https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py#L143


    Attributes:
        in_channels (int): num channels expected in input. Needs to be divisible by norm_groups.
        out_channels (int): num channels desired in output. Needs to be divisible by norm_groups.
        use_upsample (bool): include nn.Upsample layer before first conv on hidden state and on skip connection.
            Defaults to False. Cannot be True if use_downsample is True.
        use_downsample (bool): include nn.AvgPool2d layer before first conv on hidden state and on skip connection.
            Defaults to False. Cannot be True if use_upsample is True.
        activation (nn.Module): activation used before convs. Defaults to nn.SiLU().
        skip_conv (nn.Module): module used for additional convolution on skip connection. Defaults to nn.Identity().
        cond_proj (Optional[nn.Module]): module used for conditional embedding projection. Defaults to None.
        rescale_skip_connection (bool): whether to rescale skip connection by 1/sqrt(2), as described in "Diffusion
            Models Beat GANs on Image Synthesis" (https://arxiv.org/abs/2105.05233). Defaults to False.
        scale_shift_conditional (bool): if True, splits conditional embedding into two separate projections,
            and adds to hidden state as Norm(h)(w + 1) + b, as described in Appendix A in
            "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672).
            Defaults to True.
        pre_outconv_dropout (float): dropout probability before the second conv. Defaults to 0.1.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): Epsilon used in the GroupNorm layer. Defaults to 1e-5.


    Args:
        x (Tensor): input Tensor of shape [B x C x H x W]
        conditional_embedding (Tensor, optional): conditioning embedding vector of shape [B x C].
            If None, hidden state is passed through.

    Raises:
        TypeError: When skip_conv is not defined and in_channels != out_channels.
        TypeError: When use_upsample and use_downsample are both True.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', use_upsample: 'bool'=False, use_downsample: 'bool'=False, activation: 'nn.Module'=nn.SiLU(), skip_conv: 'nn.Module'=nn.Identity(), cond_proj: 'Optional[nn.Module]'=None, rescale_skip_connection: 'bool'=False, scale_shift_conditional: 'bool'=True, pre_outconv_dropout: 'float'=0.1, norm_groups: 'int'=32, norm_eps: 'float'=1e-05):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        if isinstance(skip_conv, nn.Identity) and in_channels != out_channels:
            raise ValueError('You must specify a skip connection conv if out_channels != in_channels')
        if in_channels % norm_groups != 0 or out_channels % norm_groups != 0:
            raise ValueError('Channel dims need to be divisible by norm_groups')
        if use_downsample and use_upsample:
            raise ValueError('Cannot use both upsample and downsample in res block')
        else:
            hidden_updownsample_layer: 'Union[nn.AvgPool2d, nn.Upsample, nn.Identity]'
            skip_updownsample_layer: 'Union[nn.AvgPool2d, nn.Upsample, nn.Identity]'
            if use_downsample:
                hidden_updownsample_layer = nn.AvgPool2d(kernel_size=2, stride=2)
                skip_updownsample_layer = nn.AvgPool2d(kernel_size=2, stride=2)
            if use_upsample:
                hidden_updownsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
                skip_updownsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
            else:
                hidden_updownsample_layer = nn.Identity()
                skip_updownsample_layer = nn.Identity()
        self.cond_proj = cond_proj
        self.in_block = nn.Sequential(Fp32GroupNorm(norm_groups, in_channels, eps=norm_eps), activation, hidden_updownsample_layer, nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.out_group_norm = Fp32GroupNorm(norm_groups, out_channels, eps=norm_eps)
        self.out_block = nn.Sequential(activation, nn.Dropout(pre_outconv_dropout), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.skip_block = nn.Sequential(skip_updownsample_layer, skip_conv)
        self.scale_shift_conditional = scale_shift_conditional
        self.rescale_skip_connection = rescale_skip_connection

    def forward(self, x: 'Tensor', conditional_embedding: 'Optional[Tensor]'=None) ->Tensor:
        skip = self.skip_block(x)
        h = self.in_block(x)
        if conditional_embedding is not None and self.cond_proj is not None:
            t = self.cond_proj(conditional_embedding)
            t = t.unsqueeze(-1).unsqueeze(-1)
            if self.scale_shift_conditional:
                h = self.out_group_norm(h)
                scale, shift = torch.chunk(t, 2, dim=1)
                h = h * (1 + scale) + shift
                h = self.out_block(h)
            else:
                h = self.out_block(self.out_group_norm(h + t))
        else:
            h = self.out_block(self.out_group_norm(h))
        if self.rescale_skip_connection:
            h = (skip + h) / 1.414
        else:
            h = skip + h
        return h


def adm_cond_proj(dim_cond: 'int', cond_channels: 'int', scale_shift_conditional: 'bool'=True) ->nn.Module:
    if scale_shift_conditional:
        cond_channels *= 2
    return nn.Sequential(nn.SiLU(), nn.Linear(dim_cond, cond_channels))


def adm_res_skipconv_block(in_channels: 'int', out_channels: 'int', dim_cond: 'int', rescale_skip_connection: 'bool'=False) ->ResBlock:
    skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    return ResBlock(in_channels=in_channels, out_channels=out_channels, skip_conv=skip_conv, rescale_skip_connection=rescale_skip_connection, cond_proj=adm_cond_proj(dim_cond, out_channels))


def adm_res_block(in_channels: 'int', out_channels: 'int', dim_cond: 'int', rescale_skip_connection: 'bool'=False) ->ResBlock:
    if in_channels != out_channels:
        return adm_res_skipconv_block(in_channels, out_channels, dim_cond)
    return ResBlock(in_channels=in_channels, out_channels=out_channels, rescale_skip_connection=rescale_skip_connection, cond_proj=adm_cond_proj(dim_cond, out_channels))


def adm_res_upsample_block(num_channels: 'int', dim_cond: 'int', rescale_skip_connection: 'bool'=False) ->ResBlock:
    return ResBlock(in_channels=num_channels, out_channels=num_channels, use_upsample=True, rescale_skip_connection=rescale_skip_connection, cond_proj=adm_cond_proj(dim_cond, num_channels))


def adm_stack_res(in_channels: 'int', out_channels: 'int', dim_cond: 'int') ->nn.Module:
    adm_stack = ADMStack()
    adm_stack.append_residual_block(adm_res_block(in_channels=in_channels, out_channels=out_channels, dim_cond=dim_cond))
    return adm_stack


def adm_stack_res_attn(in_channels: 'int', out_channels: 'int', dim_res_cond: 'int', dim_attn_cond: 'Optional[int]'=None) ->nn.Module:
    adm_stack = ADMStack()
    adm_stack.append_residual_block(adm_res_block(in_channels=in_channels, out_channels=out_channels, dim_cond=dim_res_cond))
    adm_stack.append_attention_block(adm_attn_block(num_channels=out_channels, dim_cond=dim_attn_cond))
    return adm_stack


def adm_res_downsample_block(num_channels: 'int', dim_cond: 'int', rescale_skip_connection: 'bool'=False) ->ResBlock:
    return ResBlock(in_channels=num_channels, out_channels=num_channels, use_downsample=True, rescale_skip_connection=rescale_skip_connection, cond_proj=adm_cond_proj(dim_cond, num_channels))


def adm_stack_res_down(num_channels: 'int', dim_cond: 'int') ->nn.Module:
    adm_stack = ADMStack()
    adm_stack.append_residual_block(adm_res_downsample_block(num_channels=num_channels, dim_cond=dim_cond))
    return adm_stack


class ADMResBlock(nn.Module):
    """Residual block in the ADM net. Supports projecting a conditional embedding to add to the hidden state.
    This typically contains the timestep embedding, but can also contain class embedding for classifier free guidance,
    CLIP image embedding and text encoder output for text-to-image generation as in DALL-E 2, or anything you want to
    condition the diffusion model on.

    Follows the architecture described in "Diffusion Models Beat GANs on Image Synthesis"
    (https://arxiv.org/abs/2105.05233) and BigGAN residual blocks (https://arxiv.org/abs/1809.11096).

    Code ref:
    https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py#L143

    Attributes:
        in_channels (int): num channels expected in input. Needs to be divisible by norm_groups.
        out_channels (int): num channels desired in output. Needs to be divisible by norm_groups.
        dim_cond (int): dimensionality of conditional projection layer
        use_upsample (bool): include nn.Upsample layer before first conv on hidden state and on skip connection.
            Defaults to False. Cannot be True if use_downsample is True.
        use_downsample (bool): include nn.AvgPool2d layer before first conv on hidden state and on skip connection.
            Defaults to False. Cannot be True if use_upsample is True.
        activation (nn.Module): activation used before convs. Defaults to nn.SiLU().
        skip_conv (nn.Module): module used for additional convolution on skip connection. Defaults to nn.Identity().
        rescale_skip_connection (bool): whether to rescale skip connection by 1/sqrt(2), as described in "Diffusion
            Models Beat GANs on Image Synthesis" (https://arxiv.org/abs/2105.05233). Defaults to False.
        scale_shift_conditional (bool): if True, splits conditional embedding into two separate projections,
            and adds to hidden state as Norm(h)(w + 1) + b, as described in Appendix A in
            "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672).
            Defaults to True.
        pre_outconv_dropout (float): dropout probability before the second conv. Defaults to 0.1.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): Epsilon used in the GroupNorm layer. Defaults to 1e-5.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]
        conditional_embedding (Tensor): conditioning embedding vector of shape [b, c].

    Raises:
        TypeError: When skip_conv is not defined and in_channels != out_channels.
        TypeError: When use_upsample and use_downsample are both True.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', dim_cond: 'int', use_upsample: 'bool'=False, use_downsample: 'bool'=False, activation: 'nn.Module'=nn.SiLU(), skip_conv: 'nn.Module'=nn.Identity(), rescale_skip_connection: 'bool'=False, scale_shift_conditional: 'bool'=True, pre_outconv_dropout: 'float'=0.1, norm_groups: 'int'=32, norm_eps: 'float'=1e-05):
        super().__init__()
        if isinstance(skip_conv, nn.Identity) and in_channels != out_channels:
            raise ValueError('You must specify a skip connection conv if out_channels != in_channels')
        if in_channels % norm_groups != 0 or out_channels % norm_groups != 0:
            raise ValueError('Channel dims need to be divisible by norm_groups')
        hidden_updownsample_layer: 'nn.Module'
        skip_updownsample_layer: 'nn.Module'
        if use_downsample and use_upsample:
            raise ValueError('Cannot use both upsample and downsample in res block')
        elif use_downsample:
            hidden_updownsample_layer = nn.AvgPool2d(kernel_size=2, stride=2)
            skip_updownsample_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        elif use_upsample:
            hidden_updownsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
            skip_updownsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            hidden_updownsample_layer = nn.Identity()
            skip_updownsample_layer = nn.Identity()
        cond_channels = 2 * out_channels if scale_shift_conditional else out_channels
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(dim_cond, cond_channels))
        self.in_block = nn.Sequential(Fp32GroupNorm(norm_groups, in_channels, eps=norm_eps), activation, hidden_updownsample_layer, nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.out_group_norm = Fp32GroupNorm(norm_groups, out_channels, eps=norm_eps)
        self.out_block = nn.Sequential(activation, nn.Dropout(pre_outconv_dropout), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.skip_block = nn.Sequential(skip_updownsample_layer, skip_conv)
        self.scale_shift_conditional = scale_shift_conditional
        self.rescale_skip_connection = rescale_skip_connection

    def forward(self, x: 'Tensor', conditional_embedding: 'Tensor') ->Tensor:
        t = self.cond_proj(conditional_embedding)
        t = t.unsqueeze(-1).unsqueeze(-1)
        skip = self.skip_block(x)
        h = self.in_block(x)
        if self.scale_shift_conditional:
            h = self.out_group_norm(h)
            scale, shift = torch.chunk(t, 2, dim=1)
            h = h * (1 + scale) + shift
            h = self.out_block(h)
        else:
            h = self.out_block(self.out_group_norm(h + t))
        if self.rescale_skip_connection:
            h = (skip + h) / 1.414
        else:
            h = skip + h
        return h


class Dalle2ImageTransform(nn.Module):
    """Dalle image transform normalizes the data between a min and max value. Defaults
    are -1 and 1 like the Dalle2 Paper.

    Args:
        image_size (int): desired output image size.
        image_min (float): min of images, used for normalization.
        image_max (float): max of images, used for normalization.
        image_field (str): key name for the image

    Inputs:
        image (Union[List[Image], Image]): Image or batch of images upon which
            to apply the transform.
    """

    def __init__(self, image_size: 'int'=64, image_min: 'float'=-1.0, image_max: 'float'=1.0, image_field: 'str'='x') ->None:
        super().__init__()
        self.image = image_field
        self.image_transform = tv.Compose([partial(cascaded_resize, resolution=image_size), tv.CenterCrop(image_size), tv.ToTensor(), partial(normalize, image_min=image_min, image_max=image_max)])

    def forward(self, x: 'Dict[str, Any]') ->Dict[str, Any]:
        assert self.image in x, f'{type(self).__name__} expects key {self.image}'
        image = x[self.image]
        if isinstance(image, Image):
            im = self.image_transform(image)
        else:
            im = torch.stack([self.image_transform(x) for x in image])
        x[self.image] = im
        return x


class AttentionResBlock(nn.Module):
    """Attention block in the LDM Autoencoder that consists of group norm, attention,
    conv projection and a residual connection.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/model.py#LL150C1-L150C6

    Attributes:
        num_channels (int): channel dim expected in input, determines embedding dim of
            q, k, v in attention module. Needs to be divisible by norm_groups.
        attn_module (nn.Module): Module of attention mechanism to use.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): epsilon used in the GroupNorm layer. Defaults to 1e-6.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]

    Raises:
        ValueError: If `num_channels` is not divisible by `norm_groups`.
    """

    def __init__(self, num_channels: 'int', attn_module: 'nn.Module', norm_groups: 'int'=32, norm_eps: 'float'=1e-06):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        if num_channels % norm_groups != 0:
            raise ValueError('Channel dims need to be divisible by norm_groups')
        self.net = nn.Sequential(OrderedDict([('norm', Fp32GroupNorm(norm_groups, num_channels, norm_eps)), ('attn', attn_module), ('out', nn.Conv2d(num_channels, num_channels, kernel_size=1))]))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.net(x) + x


class VanillaAttention(nn.Module):
    """Attention module used in the LDM Autoencoder. Similar to standard Q, k V attention,
    but using 2d convolutions instead of linear projections for obtaining q, k, v tensors.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/model.py#LL150C1-L150C6

    Attributes:
        num_channels (int): channel dim expected in input, determines embedding dim of q, k, v.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]
    """

    def __init__(self, num_channels: 'int'):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.query = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.key = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.value = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, x: 'Tensor') ->Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        B, C, H, W = q.shape
        q, k, v = (t.reshape(B, C, H * W).permute(0, 2, 1) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return out.permute(0, 2, 1).reshape(B, C, H, W)


def attention_res_block(channels: 'int', norm_groups: 'int'=32, norm_eps: 'float'=1e-06) ->AttentionResBlock:
    return AttentionResBlock(channels, VanillaAttention(channels), norm_groups, norm_eps)


def res_block(in_channels: 'int', out_channels: 'int', dropout: 'float'=0.0, norm_groups: 'int'=32, norm_eps: 'float'=1e-06) ->ResBlock:
    res_block_partial = partial(ResBlock, in_channels=in_channels, out_channels=out_channels, pre_outconv_dropout=dropout, scale_shift_conditional=False, norm_groups=norm_groups, norm_eps=norm_eps)
    if in_channels != out_channels:
        return res_block_partial(skip_conv=nn.Conv2d(in_channels, out_channels, kernel_size=1))
    else:
        return res_block_partial()


class Downsample2D(nn.Module):
    """2-Dimensional downsampling layer with zero padding and 2D convolution,
    used for image encoders.

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L134

    Attributes:
        channels (int): Number of channels in the input.
        asymmetric_padding (bool): Whether to use asymmetric padding.
            Defaults to True.

    Args:
        x (Tensor): 2-D image input tensor with shape (n, c, h, w).
    """

    def __init__(self, channels: 'int', asymmetric_padding: 'bool'=True):
        super().__init__()
        padding: 'Union[int, Tuple[int, int, int, int]]'
        if asymmetric_padding:
            padding = 0, 1, 0, 1
        else:
            padding = 1
        self.op = nn.Sequential(nn.ZeroPad2d(padding), nn.Conv2d(channels, channels, kernel_size=3, stride=2))

    def forward(self, x: 'Tensor') ->Tensor:
        return self.op(x)


class Upsample2D(nn.Module):
    """2-Dimensional upsampling layer with nearest neighbor interpolation and
    2D convolution, used for image decoders.

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L91

    Attributes:
        channels (int): Number of channels in the input.

    Args:
        x (Tensor): 2-D image input tensor with shape (n, c, h, w).
    """

    def __init__(self, channels: 'int'):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: 'Tensor') ->Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


def res_block_stack(in_channels: 'int', out_channels: 'int', num_blocks: 'int', dropout: 'float'=0.0, needs_upsample: 'bool'=False, needs_downsample: 'bool'=False, norm_groups: 'int'=32, norm_eps: 'float'=1e-06) ->nn.Module:
    if needs_upsample and needs_downsample:
        raise ValueError('Cannot use both upsample and downsample in res block')
    block_in, block_out = in_channels, out_channels
    block_stack = nn.Sequential()
    for _ in range(num_blocks):
        block_stack.append(res_block(block_in, block_out, dropout, norm_groups, norm_eps))
        block_in = block_out
    if needs_downsample:
        block_stack.append(Downsample2D(out_channels))
    if needs_upsample:
        block_stack.append(Upsample2D(out_channels))
    return block_stack


class ResNetEncoder(nn.Module):
    """Resnet encoder used in the LDM Autoencoder that consists of a init convolution,
    downsampling resnet blocks, middle resnet blocks with attention and output convolution block
    with group normalization and nonlinearity.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/model.py#L368

    Attributes:
        in_channels (int): number of input channels.
        z_channels (int): number of latent channels.
        channels (int): number of channels in the initial convolution layer.
        num_res_block (int): number of residual blocks at each resolution.
        channel_multipliers (Sequence[int]): list of channel multipliers. Defaults to [1, 2, 4, 8].
        dropout (float): dropout probability. Defaults to 0.0.
        double_z (bool): whether to use double z_channels for images or not. Defaults to True.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): epsilon used in the GroupNorm layer. Defaults to 1e-6.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]

    Raises:
        ValueError: If `channels` * `channel_multipliers[-1]` is not divisible by `norm_groups`.
    """

    def __init__(self, in_channels: 'int', z_channels: 'int', channels: 'int', num_res_blocks: 'int', channel_multipliers: 'Sequence[int]'=(1, 2, 4, 8), dropout: 'float'=0.0, double_z: 'bool'=True, norm_groups: 'int'=32, norm_eps: 'float'=1e-06):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.init_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.down_block = nn.Sequential()
        channels_list = tuple([(channels * multiplier) for multiplier in [1] + list(channel_multipliers)])
        num_resolutions = len(channel_multipliers)
        for level_idx in range(num_resolutions):
            block_in = channels_list[level_idx]
            block_out = channels_list[level_idx + 1]
            self.down_block.append(res_block_stack(block_in, block_out, num_res_blocks, dropout, needs_downsample=True if level_idx != num_resolutions - 1 else False, norm_groups=norm_groups, norm_eps=norm_eps))
        mid_channels = channels_list[-1]
        self.mid_block = nn.Sequential(res_block(mid_channels, mid_channels, dropout, norm_groups, norm_eps), attention_res_block(mid_channels, norm_groups, norm_eps), res_block(mid_channels, mid_channels, dropout, norm_groups, norm_eps))
        if mid_channels % norm_groups != 0:
            raise ValueError('Channel dims obtained by multiplying channels with last item in channel_multipliers needs to be divisible by norm_groups')
        self.out_block = nn.Sequential(Fp32GroupNorm(num_groups=norm_groups, num_channels=mid_channels, eps=norm_eps), nn.SiLU(), nn.Conv2d(mid_channels, out_channels=2 * z_channels if double_z else z_channels, kernel_size=3, padding=1))

    def forward(self, x: 'Tensor') ->Tensor:
        h = self.init_conv(x)
        h = self.down_block(h)
        h = self.mid_block(h)
        h = self.out_block(h)
        return h


class ResNetDecoder(nn.Module):
    """Resnet decoder used in the LDM Autoencoder that consists of a init convolution,
    middle resnet blocks with attention, upsamling resnet blocks and output convolution
    block with group normalization and nonlinearity. Optionally, also supports alpha
    channel in output.

    Follows the architecture described in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/diffusionmodules/model.py#L462

    Attributes:
        out_channels (int): number of channels in output image.
        z_channels (int): number of latent channels.
        channels (int): number of channels to be used with channel multipliers.
        num_res_block (int): number of residual blocks at each resolution.
        channel_multipliers (Sequence[int]): list of channel multipliers used by the encoder.
            Decoder uses them in reverse order. Defaults to [1, 2, 4, 8].
        dropout (float): dropout probability. Defaults to 0.0.
        norm_groups (int): number of groups used in GroupNorm layer. Defaults to 32.
        norm_eps (float): epsilon used in the GroupNorm layer. Defaults to 1e-6.
        output_alpha_channel (bool): whether to include an alpha channel in the output.
            Defaults to False.

    Args:
        z (Tensor): input Tensor of shape [b, c, h, w]

    Raises:
        ValueError: If `channels` * `channel_multipliers[-1]` is not divisible by `norm_groups`.
    """

    def __init__(self, out_channels: 'int', z_channels: 'int', channels: 'int', num_res_blocks: 'int', channel_multipliers: 'Sequence[int]'=(1, 2, 4, 8), dropout: 'float'=0.0, norm_groups: 'int'=32, norm_eps: 'float'=1e-06, output_alpha_channel: 'bool'=False):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.output_alpha_channel = output_alpha_channel
        channels_list = tuple(reversed([(channels * multiplier) for multiplier in list(channel_multipliers) + [channel_multipliers[-1]]]))
        mid_channels = channels_list[0]
        self.init_conv = nn.Conv2d(z_channels, mid_channels, kernel_size=3, padding=1)
        self.mid_block = nn.Sequential(res_block(mid_channels, mid_channels, dropout, norm_groups, norm_eps), attention_res_block(mid_channels, norm_groups, norm_eps), res_block(mid_channels, mid_channels, dropout, norm_groups, norm_eps))
        self.up_block = nn.Sequential()
        num_resolutions = len(channel_multipliers)
        for level_idx in range(num_resolutions):
            block_in = channels_list[level_idx]
            block_out = channels_list[level_idx + 1]
            self.up_block.append(res_block_stack(block_in, block_out, num_res_blocks + 1, dropout, needs_upsample=True if level_idx != num_resolutions - 1 else False, norm_groups=norm_groups, norm_eps=norm_eps))
        post_upsample_channels = channels_list[-1]
        if post_upsample_channels % norm_groups != 0:
            raise ValueError('Channel dims obtained by multiplying channels with first item in channel_multipliers needs to be divisible by norm_groups')
        self.out_nonlinearity_block = nn.Sequential(Fp32GroupNorm(num_groups=norm_groups, num_channels=post_upsample_channels, eps=norm_eps), nn.SiLU())
        self.conv_out = nn.Conv2d(post_upsample_channels, out_channels, kernel_size=3, padding=1)
        if self.output_alpha_channel:
            self.alpha_conv_out = nn.Conv2d(post_upsample_channels, 1, kernel_size=3, padding=1)

    def forward(self, z: 'Tensor') ->Tensor:
        h = self.init_conv(z)
        h = self.mid_block(h)
        h = self.up_block(h)
        h = self.out_nonlinearity_block(h)
        if self.output_alpha_channel:
            h = torch.cat((self.conv_out(h), self.alpha_conv_out(h)), dim=1)
        else:
            h = self.conv_out(h)
        return h


class VAEOutput(NamedTuple):
    posterior: 'Distribution'
    decoder_output: 'Tensor'


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (https://arxiv.org/abs/1906.02691) is a special type of autoencoder
    where the encoder outputs the the parameters of the posterior latent distribution instead of
    outputting fixed vectors in the latent space. The decoder consumes a sample from the latent
    distribution to reconstruct the inputs.

    Follows the architecture used in "High-Resolution Image Synthesis with Latent
    Diffusion Models" (https://arxiv.org/abs/2112.10752)

    Code ref:
    https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py#L285

    Attributes:
        encoder (nn.Module): instance of encoder module.
        decoder (nn.Module): instance of decoder module.

    Args:
        x (Tensor): input Tensor of shape [b, c, h, w]
        sample_posterior (bool): if True, sample from posterior instead of distribution mpde.
            Defaults to True.
    """

    def __init__(self, encoder: 'nn.Module', decoder: 'nn.Module'):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: 'Tensor') ->Distribution:
        h = self.encoder(x)
        mean, log_variance = torch.chunk(h, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30.0, 20.0)
        stddev = torch.exp(log_variance / 2.0)
        posterior = Normal(mean, stddev)
        return posterior

    def decode(self, z: 'Tensor') ->Tensor:
        return self.decoder(z)

    def forward(self, x: 'Tensor', sample_posterior: 'bool'=True) ->VAEOutput:
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.rsample()
        else:
            z = posterior.mode
        decoder_out = self.decode(z)
        return VAEOutput(posterior=posterior, decoder_output=decoder_out)


class VLBLoss(nn.Module):
    """VLBLoss minimizes the KL divergence between the distribution of the forward diffusion process (noising) and the
    learned reverse process (denoising). Its name is derived from the Variational Lower Bound which is being optimized.
    This loss function can be used on it's own or used in conjunction with a simpler loss method as proposed by
    Nicol & Dhariwal 2021.

    The details of the loss function are described in "Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2006.11239) and "Improved Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2102.09672)

    Code ref:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/losses.py

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time

    Args:
        pred_mean (Tensor): predicted mean for time t
        pred_log_variance (Tensor): predicted log variance for time t
        x0 (Tensor): target data point
        xt (Tensor): corrupted datapoint
        t (Tensor): diffusion step

    """

    def __init__(self, schedule: 'DiscreteGaussianSchedule'):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.schedule = schedule

    def approx_standard_normal_cdf(self, x: 'Tensor') ->Tensor:
        return 0.5 * (1.0 + torch.tanh((2.0 / math.pi) ** 0.5 * (x + 0.044715 * x.pow(3))))

    def discretized_gaussian_log_likelihood(self, x: 'Tensor', mean: 'Tensor', log_scale: 'Tensor', thres: 'float'=0.999, eps: 'float'=1e-12) ->Tensor:
        if not x.shape == mean.shape == log_scale.shape:
            ValueError('x, mean, and log_scale must all be the same shape')
        centered_x = x - mean
        inv_stdv = torch.exp(-log_scale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = cdf_plus.clamp(min=eps).log()
        log_one_minus_cdf_min = (1.0 - cdf_min).clamp(min=eps).log()
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(x < -thres, log_cdf_plus, torch.where(x > thres, log_one_minus_cdf_min, cdf_delta.clamp(min=eps).log()))
        return log_probs

    def meanflat(self, x: 'Tensor') ->Tensor:
        return x.mean(dim=tuple(range(1, len(x.shape))))

    def normal_kl(self, x_mean: 'Tensor', x_log_var: 'Tensor', p_mean: 'Tensor', p_log_var: 'Tensor') ->Tensor:
        return 0.5 * (-1.0 + p_log_var - x_log_var + (x_log_var - p_log_var).exp() + (x_mean - p_mean).pow(2) * (-p_log_var).exp())

    def forward(self, pred_mean: 'Tensor', pred_log_var: 'Tensor', x0: 'Tensor', xt: 'Tensor', t: 'Tensor') ->Tensor:
        mean, log_variance = self.schedule.q_posterior(x0, xt, t)
        nat = 1.0 / math.log(2.0)
        kl = self.normal_kl(mean, log_variance, pred_mean, pred_log_var)
        kl = self.meanflat(kl) * nat
        decoder_nll = -self.discretized_gaussian_log_likelihood(x0, mean=pred_mean, log_scale=0.5 * pred_log_var)
        decoder_nll = self.meanflat(decoder_nll) * nat
        losses = torch.where(t == 0, decoder_nll, kl)
        return losses.mean()


class DiffusionHybridLoss(nn.Module):
    """
    Combines both simple loss (typically MSE) and VLB loss weighted by lambda, as described in Eq. 16 of
    "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672).
    VLB loss is only used to train the model learned variance.

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time
        simple_loss (nn.Module): loss function computed on prediction of diffusion model and desired target
            (typically noise). Default is nn.MSELoss.
        lmbda (float): lambda weight for vlb loss. Default is 0.001.

    Args:
        input (Tensor): prediction of diffusion model of shape [b, c, ...]
        target (Tensor): desired target of shape [b, c, ...]
        mean (Tensor): predicted mean of posterior/xt of shape [b, c, ...]
        log_variance (Tensor): predicted log variance of posterior/xt of shape [b, c, ...]
        x0 (Tensor): data sample of shape [b, c,...]
        xt (Tensor): noised data sample from diffusion process of shape [b, c, ...]
        t (Tensor): diffusion timesteps of shape [b, ]

    """

    def __init__(self, schedule: 'DiscreteGaussianSchedule', simple_loss: 'nn.Module'=nn.MSELoss(), lmbda: 'float'=0.001):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.simple_loss = simple_loss
        self.vlb_loss = VLBLoss(schedule)
        self.lmbda = lmbda

    def forward(self, input: 'Tensor', target: 'Tensor', mean: 'Tensor', log_variance: 'Tensor', x0: 'Tensor', xt: 'Tensor', t: 'Tensor') ->Tensor:
        return self.simple_loss(input, target) + self.lmbda * self.vlb_loss(mean.detach(), log_variance, x0, xt, t)


class RandomDiffusionSteps(nn.Module):
    """Data Transform to randomly sample noised data from the diffusion schedule.
    During diffusion training, random diffusion steps are sampled per model update.
    This transform samples steps and returns the steps (t), seed noise (noise), and transformed
    data at time t (xt).

    Attributes:
        schedule (DiffusionSchedule): defines diffusion of noise through time
        batched (bool): if True, transform expects a batched input
        data_field (str): key name for the data to noise
        time_field (str): key name for the diffusion timestep
        noise_field (str): key name for the random noise
        noised_data_field (str): key name for the noised data

    Args:
        x (Dict): data containing tensor "x". This represents x0, the artifact being learned.
                  The 0 represents zero diffusion steps.
    """

    def __init__(self, schedule: 'DiffusionSchedule', batched: 'bool'=True, data_field: 'str'='x', time_field: 'str'='t', noise_field: 'str'='noise', noised_data_field: 'str'='xt'):
        super().__init__()
        self.schedule = schedule
        self.batched = batched
        self.x0 = data_field
        self.t = time_field
        self.noise = noise_field
        self.xt = noised_data_field

    def forward(self, x: 'Dict[str, Any]') ->Dict[str, Any]:
        assert self.x0 in x, f'{type(self).__name__} expects key {self.x0}'
        x0 = x[self.x0]
        if not self.batched:
            t = self.schedule.sample_steps(x0.unsqueeze(0))
            t = t.squeeze(0)
        else:
            t = self.schedule.sample_steps(x0)
        noise = self.schedule.sample_noise(x0)
        xt = self.schedule.q_sample(x0, noise, t)
        x.update({self.t: t, self.noise: noise, self.xt: xt})
        return x


MAX_BRUSH_STROKE = 35


MAX_NUM_VERTEX = 3


MIN_BRUSH_STROKE = 20


MIN_IMAGE_SIZE = 64


MIN_NUM_VERTEX = 1


def draw_strokes(mask: 'Image.Image', vertexes: 'List[Tuple[int, int]]', width: 'int') ->None:
    """
    Draws the brush strokes on the mask using the provided vertexes and width.

    Args:
        mask (Image.Image): The mask image.
        vertexes (List[Tuple[int, int]]): List of vertexes.
        width (int): Width of the brush strokes.
    """
    draw = ImageDraw.Draw(mask)
    draw.line(vertexes, fill=1, width=width)
    for v in vertexes:
        draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)


def generate_vertexes(mask: 'Image.Image', num_vertexes: 'int', img_width: 'int', img_height: 'int') ->List[Tuple[int, int]]:
    """
    Generates a list of vertexes based on the mask, number of vertexes, image width, and image height.

    Args:
        mask (Image.Image): The mask image.
        num_vertexes (int): Number of vertexes.
        img_width (int): Image width.
        img_height (int): Image height.

    Returns:
        List[Tuple[int, int]]: List of vertexes.
    """
    vertex = []
    vertex.append((int(np.random.randint(img_width // 2, img_width - img_width // 4)), int(np.random.randint(img_height // 2, img_height - img_height // 4))))
    average_radius = math.sqrt(img_height * img_width + img_width * img_height) / 8
    angles = []
    mean_angle = 2 * math.pi / 2
    angle_range = 2 * math.pi / 8
    angle_min = mean_angle - np.random.uniform(0, angle_range)
    angle_max = mean_angle + np.random.uniform(0, angle_range)
    for i in range(num_vertexes):
        if i % 2 == 0:
            angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
        else:
            angles.append(np.random.uniform(angle_min, angle_max))
    for i in range(num_vertexes):
        r = np.clip(np.random.normal(loc=average_radius, scale=average_radius // 2), 0, 2 * average_radius)
        new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, img_width)
        new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, img_height)
        vertex.append((int(new_x), int(new_y)))
    return vertex


def brush_stroke_mask_image(im: 'Tensor') ->Tensor:
    """
    Generates a brush stroke mask for an input image.

    Args:
        im (Tensor): The input image tensor of shape (C, H, W), where C is the number of channels,
                     H is the image height, and W is the image width.

    Returns:
        Tensor: The generated brush stroke mask tensor of shape (1, H, W), where H is the image height
                and W is the image width. The brush stroke mask has a value of 1.0 within the brush stroke regions
                and 0.0 outside the brush stroke regions.
    """
    img_height, img_width = im.shape[-2], im.shape[-1]
    min_width = int(MIN_BRUSH_STROKE * img_width / MIN_IMAGE_SIZE)
    max_width = int(MAX_BRUSH_STROKE * img_width / MIN_IMAGE_SIZE)
    mask = Image.new('1', (img_width, img_height), 0)
    for _ in range(np.random.randint(1, 3)):
        num_vertexes = np.random.randint(MIN_NUM_VERTEX, MAX_NUM_VERTEX)
        vertex = generate_vertexes(mask, num_vertexes, img_width, img_height)
        width = int(np.random.uniform(min_width, max_width))
        draw_strokes(mask, vertex, width)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.reshape(mask, (1, img_height, img_width))
    mask = torch.tensor(mask, dtype=im.dtype).permute(0, 1, 2)
    return mask


def mask_full_image(im: 'Tensor') ->Tensor:
    """
    Create a mask covering the entire image.

    Args:
        image (Tensor): Input image tensor.

    Returns:
        Tensor: Mask covering the entire image.

    """
    img_height, img_width = im.shape[-2], im.shape[-1]
    mask = torch.ones(size=(1, img_height, img_width), dtype=im.dtype)
    return mask


MASK_VALUE = 1.0


MAX_BBOX = 48


def random_inpaint_mask_image(im: 'Tensor', vertical_margin: 'int'=0, horizontal_margin: 'int'=0) ->Tensor:
    """
    Generate a random inpainting mask for an image.

    Args:
        im (Tensor): The input image tensor of shape (C, H, W), where C is the number of channels,
                     H is the image height, and W is the image width.
        vertical_margin (int): Vertical margin to exclude from the mask. Defaults to 0.
        horizontal_margin (int): Horizontal margin to exclude from the mask. Defaults to 0.

    Returns:
        Tensor: The generated inpainting mask tensor of shape (1, H, W), where H is the image height
                and W is the image width. The inpainting mask has a value of 1.0 within the masked region
                and 0.0 outside the masked region.

    """
    img_height, img_width = im.shape[-2], im.shape[-1]
    height = int(MAX_BBOX * img_height / MIN_IMAGE_SIZE)
    width = int(MAX_BBOX * img_width / MIN_IMAGE_SIZE)
    max_top = img_height - vertical_margin - height
    max_left = img_width - horizontal_margin - width
    top = int(torch.randint(low=vertical_margin, high=max_top, size=(1,)).item())
    left = int(torch.randint(low=horizontal_margin, high=max_left, size=(1,)).item())
    bbox = BBox(top=top, left=left, height=height, width=width)
    mask = torch.zeros(size=(1, img_height, img_width), dtype=im.dtype)
    h = int(torch.randint(low=0, high=height // 3 + 1, size=(1,)).item())
    w = int(torch.randint(low=0, high=width // 3 + 1, size=(1,)).item())
    mask[:, bbox.top + h:bbox.top + bbox.height - h, bbox.left + w:bbox.left + bbox.width - w] = MASK_VALUE
    return mask


def random_outpaint_mask_image(im: 'Tensor', min_delta: 'int'=0) ->Tensor:
    """
    Generates a random outpaint mask for an input image.

    Args:
        im (Tensor): The input image tensor of shape (C, H, W), where C is the number of channels,
                     H is the image height, and W is the image width.
        min_delta (int): The minimum size of the outpaint mask. Default is 0.

    Returns:
        Tensor: The generated outpaint mask tensor of shape (1, H, W), where H is the image height
                and W is the image width. The outpaint mask has a value of MASK_VALUE within the masked region
                and 0.0 outside the masked region.
    """
    img_height, img_width = im.shape[-2], im.shape[-1]
    bbox = BBox(top=0, left=0, height=img_height, width=img_width)
    side = int(torch.randint(low=0, high=4, size=(1,)).item())
    max_delta = img_height // 2 if side in [0, 1] else img_width // 2
    size = int(torch.randint(low=min_delta, high=max_delta, size=(1,)).item())
    if side == 0:
        bbox.top = img_height - size
    elif side == 1:
        bbox.height = size
    elif side == 2:
        bbox.left = img_width - size
    elif side == 3:
        bbox.width = size
    mask = torch.zeros(size=(1, img_height, img_width), dtype=im.dtype)
    mask[:, bbox.top:bbox.height, bbox.left:bbox.width] = MASK_VALUE
    return mask


class RandomInpaintingMask(nn.Module):
    """Data transform to generate mask for training with inpainting. This approach
    is based on "Palette: Image-to-Image Diffusion Models" (https://arxiv.org/abs/2111.05826)
    and "GLIDE: Towards Photorealistic Image Generation and Editing with
    Text-Guided Diffusion Models" (https://arxiv.org/abs/2112.10741). A random mask is generated
    and concatenated with the original image and masked image.

    Attributes:
        prob_masking_threshold (float): Probability of fully masking each image.
        batched (bool): if True, transform expects a batched input
        data_field (str): key name for data
        mask_field (str): key name add mask

    Args:
        x (Dict): data containing tensor "x".

    """

    def __init__(self, prob_masking_threshold: 'float'=0.25, batched: 'bool'=True, data_field: 'str'='x', mask_field: 'str'='mask'):
        super().__init__()
        self.prob_masking_threshold = prob_masking_threshold
        self.batched = batched
        self.data = data_field
        self.mask = mask_field

    def _random_mask(self, image: 'Tensor') ->Tensor:
        prob_masking = random.random()
        if prob_masking < self.prob_masking_threshold:
            mask = mask_full_image(image)
        else:
            chosen_mask = random.randint(0, 3)
            if chosen_mask == 0:
                mask = random_inpaint_mask_image(image)
            elif chosen_mask == 1:
                mask = brush_stroke_mask_image(image)
            else:
                mask = random_outpaint_mask_image(image)
        return 1 - mask

    def forward(self, x: 'Dict[str, Any]') ->Dict[str, Any]:
        assert self.data in x, f'{type(self).__name__} expects key {self.data}'
        data = x[self.data]
        if self.batched:
            x[self.mask] = torch.stack([self._random_mask(i) for i in data])
        else:
            x[self.mask] = self._random_mask(data)
        return x


class SuperResolutionTransform(nn.Module):
    """Data transform to generate small image for training with super resolution.

    Attributes:
        size (int): expected size of data
        low_res_size (int): size of scaled down data
        data_field (str): key name for data
        low_res_field (str): key name for low resolution data
        min_val (Optional[float]): min value for data
        max_val (Optional[float]): max value for data
        antilias (bool): whether to apply anti-aliasing when downsampling.
        mode (str): Interpolation mode to resizing
        align_corners (bool): align corners based on pixel pixel order instead of center.

    Args:
        x (Dict): data containing tensor "x".

    """

    def __init__(self, size: 'int', low_res_size: 'int', data_field: 'str'='x', low_res_field: 'str'='low_res', min_val: 'Optional[float]'=None, max_val: 'Optional[float]'=None, antialias: 'bool'=True, mode: 'str'='bicubic', align_corners: 'bool'=False, augmentation_func: 'Optional[Callable]'=None):
        super().__init__()
        self.data_field = data_field
        self.low_res_field = low_res_field
        self.low_res_size = low_res_size
        self.size = size
        self.min_val = min_val
        self.max_val = max_val
        self.antialias = antialias
        self.mode = mode
        self.align_corners = align_corners
        self.augmentation_func = augmentation_func

    def forward(self, x: 'Dict[str, Any]') ->Dict[str, Any]:
        assert self.data_field in x, f'{type(self).__name__} expects key {self.data_field}'
        data = x[self.data_field]
        down_scaled = F.interpolate(data, size=self.low_res_size, mode=self.mode, antialias=self.antialias, align_corners=self.align_corners)
        if self.min_val or self.max_val:
            down_scaled = down_scaled.clamp(min=self.min_val, max=self.max_val)
        if self.augmentation_func:
            down_scaled = self.augmentation_func(down_scaled)
        up_scaled = F.interpolate(down_scaled, size=self.size, mode=self.mode, antialias=self.antialias, align_corners=self.align_corners)
        if self.min_val or self.max_val:
            up_scaled = up_scaled.clamp(min=self.min_val, max=self.max_val)
        x[self.low_res_field] = up_scaled
        return x


class ComputeV(nn.Module):
    """Data transform to compute v prediction target from x0 and noise. This transfrom
    is meant to be used with the VPredictor. V is first proposed
    in "Progressive Distillation for Fast Sampling of Diffusion Models" by Salimans
    and Ho (https://arxiv.org/abs/2202.00512).

    Attributes:
        schedule (DiscreteGaussianSchedule): defines diffusion of noise through time
        data_field (str): key name for the data to noise
        time_field (str): key name for the diffusion timestep
        noise_field (str): key name for the random noise
        v (str): key name for computed v tensor

    Args:
        x (Dict): data containing tensors "x", "t", and "noise".
    """

    def __init__(self, schedule: 'DiscreteGaussianSchedule', data_field: 'str'='x', time_field: 'str'='t', noise_field: 'str'='noise', v_field: 'str'='v'):
        super().__init__()
        self.schedule = schedule
        self.x0 = data_field
        self.t = time_field
        self.noise = noise_field
        self.v = v_field

    def forward(self, x: 'Dict[str, Any]') ->Dict[str, Any]:
        assert self.x0 in x, f'{type(self).__name__} expects key {self.x0}'
        assert self.t in x, f'{type(self).__name__} expects key {self.t}'
        assert self.noise in x, f'{type(self).__name__} expects key {self.noise}'
        x0, t, noise = x[self.x0], x[self.t], x[self.noise]
        shape, dtype = x0.shape, x0.dtype
        e_coef = self.schedule('sqrt_alphas_cumprod', t, shape)
        x_coef = self.schedule('sqrt_compliment_alphas_cumprod', t, shape)
        v = e_coef * noise - x_coef * x0
        x[self.v] = v
        return x


class ALBEFVisionEncoder(nn.Module):
    """
    Modified VisionTransformer used by ALBEF.

    This class returns the output of the encoder ('encoder.ln'), without passing it to the heads.

    Args:
        image_size (int): The size (resolution) of each image.
            Default is 256.
        patch_size (int) The size (resolution) of each patch.
            Default is 16.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            Default is 12.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            Default is 12.
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
            Default is 768.
        mlp_dim (int): Dimensionality of the MLP Block in the encoder layers.
            Default is 3072.
        dropout (float): The dropout ratio for the encoder probabilities.
            Default is 0.
        attention_dropout (float): The dropout ratio for the attention probabilities.
            Default is 0.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
            Default is 1e-6.

    Inputs:
        x (Tensor): Tensor of size (n, c, image_size, image_size) containing image features
    """

    def __init__(self, image_size: 'int'=256, patch_size: 'int'=16, num_hidden_layers: 'int'=12, num_attention_heads: 'int'=12, hidden_size: 'int'=768, mlp_dim: 'int'=3072, dropout: 'float'=0.0, attention_dropout: 'float'=0.0, layer_norm_eps: 'float'=1e-06) ->None:
        super().__init__()
        vision_transformer = VisionTransformer(image_size, patch_size, num_hidden_layers, num_attention_heads, hidden_size, mlp_dim, dropout, attention_dropout, norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps))
        self.encoder_layer_name = 'encoder.ln'
        self.encoder = create_feature_extractor(vision_transformer, [self.encoder_layer_name])

    def forward(self, x: 'Tensor') ->Tensor:
        return self.encoder(x)[self.encoder_layer_name]


ALBEFOutput = namedtuple('ALBEFOutput', ['image_embeddings', 'image_embeddings_m', 'text_embeddings', 'text_embeddings_m', 'multimodal_embeddings', 'multimodal_embeddings_m'], defaults=(None, None, None, None, None, None))


class ALBEFModel(nn.Module):
    """
    ALBEF is a model to ALign the image and text representations BEfore Fusing
    (ALBEF) them through cross-modal attention, which enables more grounded vision
    and language representation learning. (https://arxiv.org/pdf/2107.07651.pdf)

    Args:   vision_encoder (nn.Module): Instantiated vision encoder.
            text_encoder (nn.Module): Instantiated text encoder.
            multimodal_encoder (nn.Module): Instantiated multimodal encoder.
            momentum (float): Momentum parameter. Default is 0.995.

    Inputs: image (Tensor): Tensor of shape (B, C, H, W) containing image features.
            text (Tensor): Tensor of shape (B, L) containing text features.
            text_atts (Tensor): Tensor of shape (B, L) containing text attention mask.
    """

    def __init__(self, vision_encoder: 'nn.Module', text_encoder: 'nn.Module', multimodal_encoder: 'nn.Module', momentum: 'float'=0.995) ->None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.multimodal_encoder = multimodal_encoder
        self.vision_encoder_m = copy.deepcopy(vision_encoder)
        self.text_encoder_m = copy.deepcopy(text_encoder)
        self.multimodal_encoder_m = copy.deepcopy(multimodal_encoder)
        remove_grad(self.vision_encoder_m)
        remove_grad(self.text_encoder_m)
        remove_grad(self.multimodal_encoder_m)
        self.momentum = momentum

    def forward(self, image: 'Tensor', text: 'Tensor', text_atts: 'Tensor') ->ALBEFOutput:
        image_embeds = self.vision_encoder(image)
        text_embeds = self.text_encoder(text, text_atts)
        multimodal_embeddings = self.multimodal_encoder(hidden_states=text_embeds.last_hidden_state, attention_mask=text_atts, encoder_hidden_states=image_embeds)
        with torch.no_grad():
            momentum_update(self.vision_encoder, self.vision_encoder_m, self.momentum)
            momentum_update(self.text_encoder, self.text_encoder_m, self.momentum)
            momentum_update(self.multimodal_encoder, self.multimodal_encoder_m, self.momentum)
            image_embeds_m = self.vision_encoder_m(image)
            text_embeds_m = self.text_encoder_m(text, text_atts)
            multimodal_embeddings_m = self.multimodal_encoder_m(hidden_states=text_embeds_m.last_hidden_state, attention_mask=text_atts, encoder_hidden_states=image_embeds_m)
        return ALBEFOutput(image_embeddings=image_embeds, image_embeddings_m=image_embeds_m, text_embeddings=text_embeds.last_hidden_state, text_embeddings_m=text_embeds_m.last_hidden_state, multimodal_embeddings=multimodal_embeddings, multimodal_embeddings_m=multimodal_embeddings_m)


ALBEFSimilarity = namedtuple('ALBEFSimilarity', ['sim_i2t', 'sim_t2i', 'sim_i2t_m', 'sim_t2i_m'], defaults=(None, None, None, None))


ALBEFWithSimilarityOutput = namedtuple('ALBEFWithSimilarityOutput', ['image_embeddings', 'text_embeddings', 'multimodal_embeddings', 'multimodal_embeddings_neg', 'similarity', 'sim_targets'], defaults=(None, None, None, None, None, None))


def _gather_embeddings(embeddings: 'torch.Tensor') ->torch.Tensor:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return embeddings
    embeddings_all_gpus = [torch.zeros_like(embeddings) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(embeddings_all_gpus, embeddings)
    return torch.cat(embeddings_all_gpus)


class ALBEFModelWithSimilarity(nn.Module):
    """
    ALBEFModelWithSimilarity outputs image embeddings, text embeddings, multimodal embeddings,
    negative image-text pairs multimodal embeddings, and image-text similarity, as used in ITC
    and ITM losses.

    Args:   albef_model (ALBEFModel): Instantiated ALBEF model.
            vision_proj (nn.Module): Instantiated vision projection layer.
            text_proj (nn.Module): Instantiated text projection layer.
            embed_size (int): Embedding size of the vision and text projection layers. Default is 256.
            queue_size (int): Size of image and text queues for momentum distillation. Default is 65536.
            masked_token_id (int): The token id indicating a masked token. Default is -100.
            temp (float): Temperature parameter. Default is 0.07.

    Inputs: image (Tensor): Tensor of shape (B, C, H, W) containing image features.
            text (Tensor): Tensor of shape (B, L) containing text features.
            text_atts (Tensor): Tensor of shape (B, L) containing text attention mask.
            idx (Tensor): Tensor of shape (B) containing unique identifiers for each sample.
    """

    def __init__(self, albef_model: 'ALBEFModel', vision_proj: 'nn.Module', text_proj: 'nn.Module', embed_size: 'int'=256, queue_size: 'int'=65536, mask_token_id: 'int'=-100, temp: 'float'=0.07) ->None:
        super().__init__()
        self.albef_model = albef_model
        self.vision_proj = vision_proj
        self.text_proj = text_proj
        self.vision_proj_m = copy.deepcopy(vision_proj)
        self.text_proj_m = copy.deepcopy(text_proj)
        remove_grad(self.vision_proj_m)
        remove_grad(self.text_proj_m)
        self.queue_size = queue_size
        self.temp = nn.Parameter(torch.ones([]) * temp)
        self.register_buffer('image_queue', torch.randn(embed_size, queue_size))
        self.register_buffer('text_queue', torch.randn(embed_size, queue_size))
        self.register_buffer('idx_queue', torch.full((1, self.queue_size), mask_token_id))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.image_queue: 'Tensor'
        self.text_queue: 'Tensor'
        self.idx_queue: 'Tensor'
        self.queue_ptr: 'Tensor'
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, image: 'Tensor', text: 'Tensor', text_atts: 'Tensor', idx: 'Tensor') ->ALBEFWithSimilarityOutput:
        outputs = self.albef_model(image, text, text_atts)
        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.detach().clone()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        similarity = self._similarity(outputs.image_embeddings, outputs.image_embeddings_m, outputs.text_embeddings, outputs.text_embeddings_m, idx)
        image_embeds_neg, text_embeds_neg, text_atts_neg = self._neg_embeddings(outputs.image_embeddings, outputs.text_embeddings, text_atts, similarity)
        multimodal_embeddings_neg = self.albef_model.multimodal_encoder(torch.cat([outputs.text_embeddings, text_embeds_neg], dim=0), torch.cat([text_atts, text_atts_neg], dim=0), torch.cat([image_embeds_neg, outputs.image_embeddings], dim=0))
        return ALBEFWithSimilarityOutput(image_embeddings=outputs.image_embeddings, text_embeddings=outputs.text_embeddings, multimodal_embeddings=outputs.multimodal_embeddings, multimodal_embeddings_neg=multimodal_embeddings_neg, similarity=similarity, sim_targets=sim_targets)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat_m: 'Tensor', text_feat_m: 'Tensor', idx: 'Tensor') ->None:
        image_feats = _gather_embeddings(image_feat_m)
        text_feats = _gather_embeddings(text_feat_m)
        idxs = _gather_embeddings(idx)
        batch_size = image_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, 'queue_size should be divisible by batch_size'
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _similarity(self, image_embeds: 'Tensor', image_embeds_m: 'Tensor', text_embeds: 'Tensor', text_embeds_m: 'Tensor', idx: 'Tensor') ->ALBEFSimilarity:
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        with torch.no_grad():
            momentum_update(self.vision_proj, self.vision_proj_m, self.albef_model.momentum)
            momentum_update(self.text_proj, self.text_proj_m, self.albef_model.momentum)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.detach().clone()], dim=1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.detach().clone()], dim=1)
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)
        return ALBEFSimilarity(sim_i2t=sim_i2t, sim_t2i=sim_t2i, sim_i2t_m=sim_i2t_m, sim_t2i_m=sim_t2i_m)

    def _neg_embeddings(self, image_embeds: 'Tensor', text_embeds: 'Tensor', text_atts: 'Tensor', similarity: 'ALBEFSimilarity') ->Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            bs = image_embeds.size(0)
            weights_i2t = F.softmax(similarity.sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(similarity.sim_t2i[:, :bs], dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)
        image_embeds_neg, text_embeds_neg, text_atts_neg = [], [], []
        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_t2i[b], 1).item())
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        for b in range(bs):
            neg_idx = int(torch.multinomial(weights_i2t[b], 1).item())
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        return image_embeds_neg, text_embeds_neg, text_atts_neg


class MLP(nn.Module):
    """A multi-layer perceptron module.

    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(self, in_dim: 'int', out_dim: 'int', hidden_dims: 'Optional[Union[int, List[int]]]'=None, dropout: 'float'=0.5, activation: 'Callable[..., nn.Module]'=nn.ReLU, normalization: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        layers = nn.ModuleList()
        if hidden_dims is None:
            hidden_dims = []
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self.model(x)


class TransformerCrossAttentionLayer(nn.Module):
    """Transformer layer with self-attention on inputs and cross-attention on an encoder's outputs.
    Can be used in a transformer decoder or an encoder with cross-attention. Similar to
    ``nn.TransformerDecoderLayer``, but generalized for use in an encoder with cross-attention as well.
    Uses a custom ``MultiHeadAttention`` that supports n-dimensional inputs including sequences,
    images, video.

    Attributes:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention, cross-attention,
            and feedforward. Otherwise, layer norm is done after.

    Args:
        hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate self-attention on.
        encoder_hidden_states (Tensor): input tensor of shape [b, d1, ..., dn, c] to calculate
            cross-attention on.
        attention_mask (Tensor, optional): mask to be applied to self-attention inputs, ``hidden_states``.
            See ``MultiHeadAttention`` for shape requirements.
        cross_attention_mask (Tensor, optional): mask to be applied to cross-attention inputs,
            ``encoder_hidden_states``. See ``MultiHeadAttention`` for shape requirements.
    """

    def __init__(self, d_model: 'int', n_head: 'int', dim_feedforward: 'int', dropout: 'float'=0.0, activation: 'Callable[..., nn.Module]'=nn.ReLU, layer_norm_eps: 'float'=1e-12, norm_first: 'bool'=False) ->None:
        super().__init__()
        self.attention = MultiHeadAttention(dim_q=d_model, dim_kv=d_model, n_head=n_head, attn_module=SelfAttention(dropout))
        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention = MultiHeadAttention(dim_q=d_model, dim_kv=d_model, n_head=n_head, attn_module=SelfAttention(dropout))
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feedforward = MLP(d_model, d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _self_attention_block(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        output = self.attention(hidden_states, attention_mask=attention_mask, return_attn_weights=False)
        output = self.attention_dropout(output)
        return output

    def _cross_attention_block(self, hidden_states: 'Tensor', encoder_hidden_states: 'Tensor', cross_attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        output = self.cross_attention(hidden_states, encoder_hidden_states, attention_mask=cross_attention_mask, return_attn_weights=False)
        output = self.cross_attention_dropout(output)
        return output

    def _feedforward_block(self, hidden_states: 'Tensor') ->Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(self, hidden_states: 'Tensor', encoder_hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None, cross_attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        inputs = self.attention_layernorm(x)
        attn_output = self._self_attention_block(inputs, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.cross_attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(attn_norm_output, kv, cross_attention_mask)
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.feedforward_layernorm(cross_attention_residual)
        ff_residual = cross_attention_norm_output + self._feedforward_block(cross_attention_norm_output)
        return ff_residual

    def _forward_postnorm(self, hidden_states: 'Tensor', encoder_hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None, cross_attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        x = hidden_states
        kv = encoder_hidden_states
        attn_output = self._self_attention_block(x, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_norm_output = self.attention_layernorm(attn_residual)
        cross_attention_output = self._cross_attention_block(attn_norm_output, kv, cross_attention_mask)
        cross_attention_residual = cross_attention_output + attn_norm_output
        cross_attention_norm_output = self.cross_attention_layernorm(cross_attention_residual)
        ff_residual = cross_attention_norm_output + self._feedforward_block(cross_attention_norm_output)
        outputs = self.feedforward_layernorm(ff_residual)
        return outputs

    def forward(self, hidden_states: 'Tensor', encoder_hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None, cross_attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        if self.norm_first:
            return self._forward_prenorm(hidden_states, encoder_hidden_states, attention_mask, cross_attention_mask)
        else:
            return self._forward_postnorm(hidden_states, encoder_hidden_states, attention_mask, cross_attention_mask)


class ALBEFMultimodalEncoder(nn.Module):
    """
    Construct multimodal embeddings from image embeddings, text embeddings, and text attention mask.

    The ALBEFMultimodalEncoder is similar to ALBEFTextEncoder, with the addition of image-text cross attention in encoder layers.

    Args:
        hidden_size (int): Dimensionality of the encoder layers.
            Default is 768.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
            Default is 6.
        num_attention_heads (int): Number of attention heads for each attention layer in the Transformer encoder.
            Default is 12.
        intermediate_size (int): Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
            Default is 3072.
        layer_norm_eps (float): The epsilon used by the layer normalization layers.
            Default is 1e-12.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function for the Transformer encoder layer.
            Default is GELU.

    Inputs:
        hidden_states (Tensor of shape (batch_size, seq_len, hidden_size)):
            Unimodal input hidden states.
        attention_mask (Tensor of shape (batch_size, seq_len)):
            Input attention mask to avoid performing attention on padding token indices.
        encoder_hidden_states (Tensor of shape (batch_size, encoder_seq_len, hidden_size)):
            The encoder hidden states.
        encoder_attention_mask (Optional[Tensor] of shape (batch_size, encoder_seq_len)):
            The attention mask for encoder hidden states.
        is_decoder (bool): Whether this module is used as a decoder. Default is False.
    """

    def __init__(self, hidden_size: 'int'=768, num_hidden_layers: 'int'=6, num_attention_heads: 'int'=12, intermediate_size: 'int'=3072, layer_norm_eps: 'float'=1e-12, transform_act_fn: 'Callable[..., nn.Module]'=nn.GELU) ->None:
        super().__init__()
        self.layer = nn.ModuleList([TransformerCrossAttentionLayer(d_model=hidden_size, n_head=num_attention_heads, dim_feedforward=intermediate_size, activation=transform_act_fn, layer_norm_eps=layer_norm_eps) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states: 'Tensor', attention_mask: 'Tensor', encoder_hidden_states: 'Tensor', encoder_attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        attention_mask = get_extended_attention_mask(attention_mask)
        if encoder_attention_mask is not None:
            encoder_attention_mask = get_extended_attention_mask(encoder_attention_mask)
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, cross_attention_mask=encoder_attention_mask)
        return hidden_states


class Blip2Output(NamedTuple):
    """
    BLIP2 model output for loss computation.

    image_embeddings(Tensor): normalized image embeddings returned by the visual encoder
        with shape [bsz x seq_len x embed_dim].
    image_features(Tensor): Image features after qformer and projection (for stage 1 training)
        with shape [bsz, num_query_tokens, embed_dim]
    image_qformer_output(Tensor) : last hidden state for qformer output by given image input
    text_features(Optional[Tensor]): Text features after qformer and projection if text input is provided
        with shape [bsz, embed_dim]
    prediction_scores (Optional[Tensor]): computed for next word prediction
        with shape of [bsz, seq_len, vocab_size]
    """
    image_embeddings: 'Tensor'
    image_features: 'Tensor'
    image_qformer_output: 'Tensor'
    text_features: 'Optional[Tensor]' = None
    prediction_scores: 'Optional[Tensor]' = None


class BLIP2(nn.Module):
    """
    BLIP2(https://arxiv.org/pdf/2301.12597.pdf) provides a pre-training strategy to bootstrap vision-language
    pre-training from frozen image encoders and frozen large language models(LLM). BLIP-2 bridges the modality gap
    and facilitates cross-modal alignment via Querying Transformer (Q-former). Q-former is a lightweight transformer
    which has a set of learnable query vectors to extract visual features from the frozen image encoder.

    Args:
        qformer(nn.Module): Querying Transformer (Q-former)
        visual_encoder(nn.Module): Frozen image encoder
        dim_q(int) : Dimension of query tensor, this value should be the same as dim_q in qformer.
        image_encoder_embedding_dim(int): Embedding dimension for image encoder,
            this value should be the same as dim_kv in qformer.
        freeze_visual_encoder(bool): Whether to freeze the visual encoder, default to True
        cross_attention_freq(int): Frequency of adding cross-attention block in Qformer, default to 2
        embedding_dim(int): Embedding dimension
        num_query_token(int): Number of query tokens in Qformer, default to 32
        init_query_tokens(bool): whether init query token params, default to True
        decoder_bos_token_id(Optional[int]): bos_token_id used in decoder, default to None
    """

    def __init__(self, qformer: 'nn.Module', vision_encoder: 'nn.Module', dim_q: 'int', image_encoder_embedding_dim: 'int', freeze_vision_encoder: 'bool'=True, cross_attention_freq: 'int'=2, embedding_dim: 'int'=256, num_query_token: 'int'=32, init_query_tokens: 'bool'=True, decoder_bos_token_id: 'Optional[int]'=None):
        super().__init__()
        self.vision_encoder = vision_encoder
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder = self.vision_encoder.eval()
        self.qformer = qformer
        self.decoder_bos_token_id = decoder_bos_token_id
        self.dim_q = dim_q
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_token, self.dim_q))
        if init_query_tokens:
            self.query_tokens.data.normal_(mean=0.0, std=0.02)
        self.vision_proj = nn.Linear(self.dim_q, embedding_dim)
        self.text_proj = nn.Linear(self.dim_q, embedding_dim)
        self.ln_vision = nn.LayerNorm(image_encoder_embedding_dim)

    def forward(self, image: 'Tensor', input_ids: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None) ->Blip2Output:
        """
        Args:
            image(Tensor): Image input tensor with shape [B, C, H, W]
            input_ids(Optional[Tensor]): Text input tensor with shape [bsz, seq_len]
            attention_mask(Optional[Tensor]): Attention mask tensor with shape [bsz, seq_len]

        Returns:
            return BLIP2 model output(Blip2Output).
        """
        vision_encoder_output = self.vision_encoder(image)
        if isinstance(vision_encoder_output, TransformerOutput):
            vision_encoder_output = vision_encoder_output.last_hidden_state
        assert vision_encoder_output is not None
        image_embeds = self.ln_vision(vision_encoder_output)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.qformer.model(query_embeds=query_tokens, encoder_hidden_states=image_embeds, use_cache=True)
        image_feats = F.normalize(self.vision_proj(query_output[0]), dim=-1)
        text_feats: 'Optional[Tensor]' = None
        prediction_scores: 'Optional[Tensor]' = None
        if input_ids is not None:
            text_output = self.qformer.model(input_ids, attention_mask=attention_mask, use_cache=False)
            text_feats = F.normalize(self.text_proj(text_output[0][:, 0, :]), dim=-1)
            decoder_input_ids = input_ids.clone()
            if self.decoder_bos_token_id is not None:
                decoder_input_ids[:, 0] = self.decoder_bos_token_id
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long)
            if attention_mask is not None:
                attention_mask = torch.cat([query_atts, attention_mask], dim=1)
            prediction_scores = self.qformer(input_ids=decoder_input_ids, attention_mask=attention_mask, past_key_values=query_output[1], use_cache=False)
        return Blip2Output(image_embeddings=image_embeds, image_features=image_feats, image_qformer_output=query_output[0], text_features=text_feats, prediction_scores=prediction_scores)


class MHAWithCacheOutput(NamedTuple):
    attn_output: 'Tensor'
    past_key_value: 'Tuple[Tensor, Tensor]'


class MultiHeadAttentionWithCache(nn.Module):
    """
    MultiHeadAttention module for both self-attention(SA) and cross-attention(CA).
    This class supports a cache mechanism for decoders to store previous states through
    "past_key_value". Key/value states should be only cached for self-attention cases.
    q, k, v share the same dimension for self-attention,
    but different for cross-attention, CA requires encoder hidden states dim as k, v dims.

    Args:
        dim_q (int): query embedding dimension
        dim_kv (int): key, value embedding dimension,
            same as dim_q for SA; equals to encoder dimension for cross-attention
        num_heads (int): number of attention heads
        dropout (float): dropout rate
        add_bias (bool): if true, adds a learnable bias to query, key, value.
            Defaults to True.
    """

    def __init__(self, dim_q: 'int', dim_kv: 'int', num_heads: 'int', dropout: 'float'=0.0, add_bias: 'bool'=True) ->None:
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dim_q, dim_q, bias=add_bias)
        self.k_proj = nn.Linear(dim_kv, dim_q, bias=add_bias)
        self.v_proj = nn.Linear(dim_kv, dim_q, bias=add_bias)
        self.output_proj = nn.Linear(dim_q, dim_q)
        self.dropout = dropout

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor', attn_mask: 'Optional[Tensor]'=None, past_key_value: 'Optional[Tuple[Tensor, Tensor]]'=None, is_causal: 'bool'=False, use_cache: 'bool'=False) ->Union[Tensor, MHAWithCacheOutput]:
        """
        Args:
            query (Tensor): input query of shape bsz x target_seq_len x embed_dim
            key (Tensor): key of shape bsz x source_seq_len x embed_dim
            value (Tensor): value of shape bsz x source_seq_len x embed_dim
            attn_mask (optional Tensor): Attention mask of shape bsz x num_heads x target_seq_len x source_seq_len.
                Note that the num_heads dimension can equal 1 and the mask will be broadcasted to all heads.
                Two types of masks are supported. A boolean mask where a value of True
                indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            past_key_value (optional tuple of tensors): cached key and value with the same shape of key, value inputs.
                The size of tuple should be 2, where the first entry is for cached key and second entry is for cached value.
            is_causal (bool): If true, does causal attention masking, attn_mask should be set to None if this is set to True
                 is_causal is a hint that the mask is a causal mask, providing incorrect hints can result in incorrect execution.
            use_cache (bool): whether to use cache for key and value tensors

        Returns:
            if use_cache is off, return attn_output tensor of shape bsz x seq_len x embed_dim;
            otherwise return namedtuple with attn_output, cached key and value.
        """
        bsz = query.size(0)
        embed_dim = query.size(-1)
        head_dim = embed_dim // self.num_heads
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        query = query.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        if key.size(0) != bsz or value.size(0) != bsz:
            raise ValueError('key and value should have the same bsz as query.')
        key = key.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask, dropout, is_causal)
        attn = attn.transpose(1, 2).reshape(bsz, -1, embed_dim)
        attn_output = self.output_proj(attn)
        if use_cache:
            return MHAWithCacheOutput(attn_output, (key, value))
        return attn_output


class QformerLayer(nn.Module):
    """
    Qformer layer module.

    This module is designed with a self-attention (SA) block and optionally includes a cross-attention (CA) block for queries.
    The inputs for this module, referred to as hidden_states, can consist of either a query, text, or a combination of both.
    Cross-attention is exclusively activated for queries (query_length > 0) with encoder_hidden_states derived from image inputs.

    The feedforward(ff) block will project the hidden states output by the layer before,
    query output and text output are concatenated as overall output after separated handling for CA and ff.

    Args:
        dim_q (int): dimensionality of the query tensor
        dim_feedforward (int): dimensionality of the feedforward layer
        num_heads (int): number of attention heads
        attn_dropout (float): dropout probability for attention weights
        dropout (float): dropout probability for the densen layer after attention and feedforward layer
        layer_norm_eps (float): the epsilon used by the layer normalization layers
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
        has_cross_attention (bool): whether a cross-attention layer is included
        dim_kv (Optional[int]): dimensionality of the key and value tensors, this value is only used in CA.

    """

    def __init__(self, dim_q: 'int', dim_feedforward: 'int', num_heads: 'int', attn_dropout: 'float'=0.0, dropout: 'float'=0.0, layer_norm_eps: 'float'=1e-12, activation: 'Callable[..., nn.Module]'=nn.ReLU, has_cross_attention: 'bool'=False, dim_kv: 'Optional[int]'=None):
        super().__init__()
        self.self_attention = MultiHeadAttentionWithCache(dim_q, dim_q, num_heads, attn_dropout)
        self.self_attn_layernorm = Fp32LayerNorm(dim_q, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.has_cross_attention = has_cross_attention
        self.cross_attention: 'Optional[MultiHeadAttentionWithCache]' = None
        if has_cross_attention:
            if dim_kv is None:
                raise ValueError('key and value dim should be provided for cross attention.')
            self.cross_attention = MultiHeadAttentionWithCache(dim_q=dim_q, dim_kv=dim_kv, num_heads=num_heads, dropout=attn_dropout)
            self.cross_attn_layernorm = Fp32LayerNorm(dim_q, eps=layer_norm_eps)
            self.cross_attn_dropout = nn.Dropout(dropout)
        self.feedforward = MLP(dim_q, dim_q, dim_feedforward, dropout=0.0, activation=activation)
        self.feedforward_layernorm = Fp32LayerNorm(dim_q, eps=layer_norm_eps)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_query = MLP(dim_q, dim_q, dim_feedforward, dropout=0.0, activation=activation)
        self.feedforward_layernorm_query = Fp32LayerNorm(dim_q, eps=layer_norm_eps)
        self.feedforward_dropout_query = nn.Dropout(dropout)

    def _self_attention_block(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None, past_key_value: 'Optional[Tuple[Tensor, Tensor]]'=None, use_cache: 'bool'=False) ->Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        x = hidden_states
        attn_output = self.self_attention(x, x, x, attn_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        present_key_value: 'Optional[Tuple[Tensor, Tensor]]' = None
        if use_cache:
            assert isinstance(attn_output, MHAWithCacheOutput)
            attn_output_value = attn_output.attn_output
            present_key_value = attn_output.past_key_value
        else:
            assert isinstance(attn_output, Tensor)
            attn_output_value = attn_output
        attn_output = self.dropout(attn_output_value)
        attn_residual = attn_output + x
        attn_residual = self.self_attn_layernorm(attn_residual)
        return attn_residual, present_key_value

    def _cross_attention_block(self, hidden_states: 'Tensor', encoder_hidden_states: 'Tensor') ->Tensor:
        x = hidden_states
        assert self.cross_attention is not None
        cross_attn_output = self.cross_attention(query=x, key=encoder_hidden_states, value=encoder_hidden_states, use_cache=False)
        if not torch.jit.isinstance(cross_attn_output, Tensor):
            raise ValueError('cross-attention output must be Tensor.')
        cross_attn_output = self.cross_attn_dropout(cross_attn_output)
        cross_attn_residual = cross_attn_output + x
        cross_attn_residual = self.cross_attn_layernorm(cross_attn_residual)
        return cross_attn_residual

    def _feedforward_block(self, hidden_states: 'Tensor') ->Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        h = self.feedforward_layernorm(h + hidden_states)
        return h

    def _feedforward_query_block(self, hidden_states: 'Tensor') ->Tensor:
        h = self.feedforward_query(hidden_states)
        h = self.feedforward_dropout_query(h)
        h = self.feedforward_layernorm_query(h + hidden_states)
        return h

    def forward(self, hidden_states: 'Tensor', encoder_hidden_states: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, past_key_value: 'Optional[Tuple[Tensor, Tensor]]'=None, query_length: 'int'=0, use_cache: 'bool'=False) ->Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Inputs:
            hidden_states (Tensor): input query of shape bsz x seq_len x embed_dim
            encoder_hidden_states (Optional[Tensor]): input key/values of shape bsz x seq_len x embed_dim, only used in CA case
            attention_mask (Optional[Tensor]): attention mask, supported mask type is described in MultiHeadAttentionWithCache class
            past_key_value (Optional[Tuple[Tensor, Tensor]]): cached key/value tuple for self-attention
            query_length (Optional[int]): length of query embedding, used as condition
                to determine query attention output and check text existance.
            use_cache (bool): whether to use cache for key and value tensors

        Return:
            A tuple includes:
                layer_output (Tensor): layer output of shape bsz x seq_len x embed_dim
                present_key_value (Optional[Tuple[Tensor, Tensor]]): key/value tuple for self-attention
        """
        if past_key_value is not None and len(past_key_value) != 2:
            raise ValueError('past_key_value should be 2-element tuple to represent self-attention cached key/values.')
        attn_residual, present_key_value = self._self_attention_block(hidden_states=hidden_states, attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        if query_length > 0:
            query_attn_output = attn_residual[:, :query_length, :]
            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError('encoder_hidden_states must be given for cross-attention layers')
                cross_attn_output = self._cross_attention_block(hidden_states=query_attn_output, encoder_hidden_states=encoder_hidden_states)
                query_attn_output = cross_attn_output
            layer_output = self._feedforward_query_block(query_attn_output)
            if attn_residual.shape[1] > query_length:
                layer_output_text = self._feedforward_block(attn_residual[:, query_length:, :])
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = self._feedforward_block(attn_residual)
        return layer_output, present_key_value


class QformerEncoder(nn.Module):
    """
    Qformer encoder module including multiple Qformer layers.

    Args:
        num_hidden_layers (int): number of Qformer layers inside encoder
        dim_q (int): dimensionality of the query tensor
        dim_feedforward (int): dimensionality of the feedforward layer
        num_heads (int): number of attention heads
        attn_dropout (float): dropout probability for attention weights
        dropout (float): dropout probability for the densen layer after attention and feedforward layer in each Qformer layer
        layer_norm_eps (float): the epsilon used by the layer normalization layers
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
        cross_attention_freq (int): frequency of adding cross attention in QFormer layers, default to 2.
        dim_kv (Optional[int]): dimensionality of the key and value tensors, this value is only used in CA.

    """

    def __init__(self, num_hidden_layers: 'int', dim_q: 'int', dim_feedforward: 'int', num_heads: 'int', attn_dropout: 'float'=0.0, dropout: 'float'=0.0, layer_norm_eps: 'float'=1e-12, activation: 'Callable[..., nn.Module]'=nn.ReLU, cross_attention_freq: 'int'=2, dim_kv: 'Optional[int]'=None):
        super().__init__()
        layers = []
        for i in range(num_hidden_layers):
            has_cross_attention = i % cross_attention_freq == 0
            layers.append(QformerLayer(dim_q=dim_q, dim_feedforward=dim_feedforward, num_heads=num_heads, attn_dropout=attn_dropout, dropout=dropout, layer_norm_eps=layer_norm_eps, activation=activation, has_cross_attention=has_cross_attention, dim_kv=dim_kv))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: 'Tensor', encoder_hidden_states: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, past_key_values: 'Optional[List[Tuple[Tensor, Tensor]]]'=None, query_length: 'int'=0, use_cache: 'bool'=False) ->Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Inputs:
            hidden_states (Tensor): input query of shape bsz x seq_len x embed_dim
            encoder_hidden_states (Optional[Tensor]): input key/values of shape bsz x seq_len x embed_dim, only used in CA case
            attention_mask (Optional[Tensor]): attention mask, supported mask type is described in MultiHeadAttentionWithCache class
            past_key_values (Optional[List[Tuple[Tensor, Tensor]]]): cached key/value tuple for self-attention
            query_length (int): the length of input query, used for cross-attention
            use_cache (bool): whether to use cache for key and value tensors

        Return:
            A tuple includes:
                the last hidden state: Tensor of shape bsz x seq_len x embed_dim
                past_key_values (List[Optional[Tuple[Tensor, Tensor]]]]): cached key/values from Qformer layers
        """
        current_key_values = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        for i, layer_module in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, current_key_value = layer_module(hidden_states=hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, past_key_value=past_key_value, query_length=query_length, use_cache=use_cache)
            if use_cache:
                assert isinstance(current_key_value, tuple)
                current_key_values.append(current_key_value)
        return hidden_states, current_key_values


class QformerEmbedding(nn.Module):
    """
    Qformer embedding module.

    Args:
        embedding_dim (int): dim of embedding space
        max_position_embeddings (int): max sequence length allowed for positional embeddings
        vocab_size (int): size of vocabulary
        pad_token_id (int): id used for padding token, default is 0.
        dropout (float): dropout probability after embedding layers and layernorm.
        layer_norm_eps (float): the epsilon used by the layer normalization layers.
    """

    def __init__(self, embedding_dim: 'int', max_position_embeddings: 'int', vocab_size: 'int', pad_token_id: 'int'=0, layer_norm_eps: 'float'=1e-12, dropout: 'float'=0.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        self.layernorm = Fp32LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('position_ids', torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids: 'Optional[Tensor]'=None, position_ids: 'Optional[Tensor]'=None, query_embeddings: 'Optional[Tensor]'=None, past_seq_length: 'int'=0) ->Tensor:
        """
        Inputs:
            input_ids (Optional[Tensor]): input token ids
            position_ids (Optional[Tensor]): batches of of 1D integer tensors used to identify each token's position,
                if no position_ids is provided, the IDs are automatically created as absolute positional embeddings.
            query_embeddings (Optional[Tensor]): query embeddings for QFormer
            past_seq_length (Optional[int]): sequence length cached by past_key_values.

        Returns:
            embeddings (Tensor): concatenated embeddings of shape (bsz, num tokens, embedding dim), concatenation is along
            the token dimension.
        """
        if input_ids is None and query_embeddings is None:
            raise ValueError('Either input_ids or query_embeddings must be passed.')
        seq_length = input_ids.size(1) if input_ids is not None else 0
        embeddings = query_embeddings
        if input_ids is not None:
            if position_ids is None:
                position_ids = self.position_ids[:, past_seq_length:seq_length + past_seq_length].clone()
            word_embeddings = self.token_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids.long())
            embeddings = word_embeddings + position_embeddings
            if query_embeddings is not None:
                embeddings = torch.cat((query_embeddings, embeddings), dim=1)
        assert isinstance(embeddings, Tensor)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def get_causal_mask(attention_mask: 'Tensor', input_shape: 'Tuple[int, int]', has_query: 'bool'=False) ->Tensor:
    """A causal mask in addition to the padding mask for Q-Former for generation task.
       when input seq_len is shorter than attn_mask, increasing causal_mask by prefix_seq_len with 1s;
       if query is available, apply causal self-attention mask to control query-text interaction;

    Arguments:
        attention_mask (Tensor) is a binary mask with 1 for unmasked and 0 for masked positions.
            Attention_mask has size of [batch_size, attn_seq_len]. attn_seq_len can be only seq_len for text_token
            or query_len + seq_len.
        input_shape (tuple[int, int]): indicates input shape of (batch_size, input_seq_len) from embedding output.
            If query_emb is used, input_seq_len is query_len + seq_len.
            Input shape can be different from attention_mask shape for image caption and text generation tasks.
        has_query (bool) indicating whether query is available in qformer input.

    Returns:
        causal_mask (Tensor): mask size of [bsz, attn_seq_len, attn_seq_len] with query,
            [bsz, input_seq_len, attn_seq_len] without query

    """
    device = attention_mask.device
    batch_size, seq_len = input_shape
    causal_mask = get_causal_attention_mask(seq_len)
    causal_mask = causal_mask.repeat(batch_size, 1).view(batch_size, seq_len, seq_len)
    if causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        if has_query:
            causal_mask = torch.cat([torch.zeros((batch_size, prefix_seq_len, seq_len), device=device, dtype=causal_mask.dtype), causal_mask], dim=1)
        causal_mask = torch.cat([torch.ones((batch_size, causal_mask.shape[1], prefix_seq_len), device=device, dtype=causal_mask.dtype), causal_mask], dim=-1)
    return causal_mask


class QformerModel(nn.Module):
    """
    Qformer model including Qformer embedding and Qformer encoder.

    Args:
        num_hidden_layers (int): number of Qformer layers inside encoder
        dim_q (int): dimensionality of the query tensor
        dim_feedforward (int): dimensionality of the feedforward layer
        num_heads (int): number of attention heads
        max_position_embeddings (int): max sequence length allowed for positional embeddings
        vocab_size (int): size of vocabulary
        pad_token_id (int): id used for padding token, default is 0.
        query_length(int): query length in Qformer, used to compute cached query length.
            default value is the same as num_query_token for Blip2 case (https://fburl.com/316803mo).
        dim_kv (Optional[int]): dimensionality of the key and value tensors, this value is only used in CA, default is None.
        layer_norm_eps (float): the epsilon used by the layer normalization layers
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
        attn_dropout (float): dropout probability for attention weights
        dropout (float): dropout probability for the densen layer after attention and feedforward layer in each Qformer layer
        cross_attention_freq (int): frequency of adding cross attention in QFormer layers, default to 2.
    """

    def __init__(self, num_hidden_layers: 'int', dim_q: 'int', dim_feedforward: 'int', num_heads: 'int', max_position_embeddings: 'int', vocab_size: 'int', pad_token_id: 'int'=0, query_length: 'int'=32, dim_kv: 'Optional[int]'=None, layer_norm_eps: 'float'=1e-12, activation: 'Callable[..., nn.Module]'=nn.ReLU, attn_dropout: 'float'=0.0, dropout: 'float'=0.0, cross_attention_freq: 'int'=2) ->None:
        super().__init__()
        self.query_length = query_length
        self.embeddings = QformerEmbedding(embedding_dim=dim_q, max_position_embeddings=max_position_embeddings, vocab_size=vocab_size, pad_token_id=pad_token_id, layer_norm_eps=layer_norm_eps, dropout=dropout)
        self.encoder = QformerEncoder(num_hidden_layers=num_hidden_layers, dim_q=dim_q, dim_feedforward=dim_feedforward, num_heads=num_heads, attn_dropout=attn_dropout, dropout=dropout, layer_norm_eps=layer_norm_eps, activation=activation, cross_attention_freq=cross_attention_freq, dim_kv=dim_kv)

    def forward(self, input_ids: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, position_ids: 'Optional[Tensor]'=None, query_embeds: 'Optional[Tensor]'=None, encoder_hidden_states: 'Optional[Tensor]'=None, past_key_values: 'Optional[List[Tuple[Tensor, Tensor]]]'=None, use_cache: 'bool'=False, use_causal_mask: 'bool'=False) ->Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Inputs:
            input_ids (Optional[Tensor]): input token ids for QFormer
            attention_mask (Optional[Tensor]): attention mask for QFormer
            position_ids (Optional[Tensor]): position ids for QFormer
            query_embeds (Optional[Tensor]): query embeddings for QFormer
            encoder_hidden_states (Optional[Tensor]): input key/values of shape bsz x seq_len x embed_dim, only used in CA case
            past_key_values: (Optional[List[Tuple[Tensor, Tensor]]]):  a list of num_layers elements,
                each element is a 2-element tuple for cached key/value.
                key/value is tensor with shape of (bsz x source_seq_len x embed_dim).
            use_cache (bool): whether to use cache for key and value tensors
            use_causal_mask (bool): apply causal mask if true, default to False

        Returns:
            Qformer encoder output with a tuple of last hidden states and past_key_values if use_cache.
        """
        past_seq_length = past_key_values[0][0].shape[2] - self.query_length if past_key_values is not None else 0
        query_length = query_embeds.shape[1] if query_embeds is not None else 0
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, query_embeddings=query_embeds, past_seq_length=past_seq_length)
        bsz, seq_len = embedding_output.size()[:-1]
        if attention_mask is not None:
            if use_causal_mask:
                causal_mask = get_causal_mask(attention_mask, (bsz, seq_len), has_query=query_embeds is not None)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                attention_mask = extended_attention_mask
            else:
                attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        return self.encoder(hidden_states=embedding_output, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values, use_cache=use_cache, query_length=query_length)


class QformerPredictionHead(nn.Module):
    """
    MLP head for computinng prediction score from QformerModel output

    Args:
        dim_q (int): dimensionality of the query tensor
        vocab_size (int): the size of vocabulary used by QFormer
        layer_norm_eps (float): the epsilon used by the layer normalization layers, default is 1e-12
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
    """

    def __init__(self, dim_q: 'int', vocab_size: 'int', layer_norm_eps: 'float'=1e-12, activation: 'Callable[..., nn.Module]'=nn.GELU) ->None:
        super().__init__()
        self.linear_1 = nn.Linear(dim_q, dim_q)
        self.activation = activation()
        self.layernorm = nn.LayerNorm(dim_q, eps=layer_norm_eps)
        self.linear_2 = nn.Linear(dim_q, vocab_size)

    def forward(self, sequence_output: 'Tensor') ->Tensor:
        """
        Inputs (Tensor):
            sequence_output of shape bsz x seq_len x embed_dim
        Returns:
            prediction scores (Tensor) of shape: bsz x seq_len x vocab_size
        """
        hidden_states = self.linear_1(sequence_output)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        predictions = self.linear_2(hidden_states)
        return predictions


class QformerForCLM(nn.Module):
    """
    A QformerModel wrapper class for causal language modeling(clm).

    Args:
        num_hidden_layers (int): number of Qformer layers inside encoder
        dim_q (int): dimensionality of the query tensor
        dim_feedforward (int): dimensionality of the feedforward layer
        num_heads (int): number of attention heads
        max_position_embeddings (int): max sequence length allowed for positional embeddings
        vocab_size (int): size of vocabulary
        pad_token_id (int): id used for padding token, default is 0.
        query_length(int): query length in Qformer, details see QformerModel class.
        dim_kv (Optional[int]): dim_kv (Optional[int]): dimensions of the key and value tensors, this value is only used in CA.
            Default is None.
        layer_norm_eps (float): the epsilon used by the layer normalization layers
        activation (Callable[..., nn.Module]): the activation function applied to the feedforward layer
        attn_dropout (float): dropout probability for attention weights
        dropout (float): dropout probability for the densen layer after attention and feedforward layer in each Qformer layer
        cross_attention_freq (int): frequency of adding cross attention in QFormer layers, default to 2
    """

    def __init__(self, num_hidden_layers: 'int', dim_q: 'int', dim_feedforward: 'int', num_heads: 'int', max_position_embeddings: 'int', vocab_size: 'int', pad_token_id: 'int'=0, query_length: 'int'=32, dim_kv: 'Optional[int]'=None, layer_norm_eps: 'float'=1e-12, activation: 'Callable[..., nn.Module]'=nn.GELU, attn_dropout: 'float'=0.0, dropout: 'float'=0.0, cross_attention_freq: 'int'=2) ->None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.head = QformerPredictionHead(dim_q=dim_q, activation=activation, layer_norm_eps=layer_norm_eps, vocab_size=vocab_size)
        self.model = QformerModel(num_hidden_layers=num_hidden_layers, dim_q=dim_q, dim_feedforward=dim_feedforward, num_heads=num_heads, max_position_embeddings=max_position_embeddings, vocab_size=vocab_size, pad_token_id=pad_token_id, query_length=query_length, dim_kv=dim_kv, layer_norm_eps=layer_norm_eps, activation=activation, attn_dropout=attn_dropout, dropout=dropout, cross_attention_freq=cross_attention_freq)

    def forward(self, input_ids: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, position_ids: 'Optional[Tensor]'=None, query_embeds: 'Optional[Tensor]'=None, encoder_hidden_states: 'Optional[Tensor]'=None, past_key_values: 'Optional[List[Tuple[Tensor, Tensor]]]'=None, use_cache: 'bool'=False) ->Tensor:
        """
        Inputs:
            input_ids (Optional[Tensor]): input token ids for QFormer
            attention_mask (Optional[Tensor]): attention mask for QFormer
            position_ids (Optional[Tensor]): position ids for QFormer
            query_embeds (Optional[Tensor]): query embeddings for QFormer
            encoder_hidden_states (Optional[Tensor]): input key/values of shape bsz x seq_len x embed_dim, only used in CA case
            past_key_values: (Optional[List[Tuple[Tensor, Tensor]]]): cached key/value tuple for self-attention
            use_cache (bool): whether to use cache for key and value tensors,
                default to False for generation as cached values should be computed in previous training tasks.

        Returns:
            prediction score (Tensor) computed for next word prediction of shape
                bsz x seq_len x vocab_size
        """
        if past_key_values is not None:
            assert query_embeds is None
        sequence_output, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, query_embeds=query_embeds, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values, use_cache=use_cache, use_causal_mask=True)
        if query_embeds is not None:
            sequence_output = sequence_output[:, query_embeds.shape[1]:, :]
        prediction_scores = self.head(sequence_output)
        return prediction_scores


class SiLU(nn.Module):
    """Sigmoid Linear Unit

    .. math:: \\text{SiLU}(x) = x * \\sigma(1.702 * x)

    where :math:`\\sigma(x)` is the cumulative distribution function for Logistic Distribution.

    Approximation of the exact GeLU for greater forward speed. Note that this is different from
    ``torch.nn.SiLU`` by the coefficient ``1.702`` from the paper:
    `"Gaussian error linear units"<https://arxiv.org/pdf/1606.08415.pdf>`_.
    """

    def forward(self, x: 'Tensor') ->Tensor:
        return torch.sigmoid(1.702 * x) * x


class CLIPViTEncoder(nn.Module):
    """
    Vision transformer encoder for CLIP.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders projections.
        patch_size (int): The dimension of each patch
        image_size(int): The size (width==height) of input image
        width (int): Dimensionality of the encoder layers and the pooler layer
        heads (int): Number of attention heads for each attention layer in the Transformer encoder
        layers (int): Number of hidden layers in the Transformer encoder

    Inputs:
        x (Tensor): image tensor with dimensions B x C(3) x image_size x image_size
    """

    def __init__(self, embedding_dim: 'int', patch_size: 'int', image_size: 'int', width: 'int', heads: 'int', layers: 'int'):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.conv = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.image_size = image_size
        scale = width ** -0.5
        self.cls_token_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_size // patch_size) ** 2 + 1, width))
        self.ln_pre = Fp32LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dropout=0.0, activation=SiLU(), norm_first=True, dim_feedforward=4 * width, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln_post = Fp32LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, embedding_dim))

    def forward(self, x: 'Tensor') ->Tensor:
        if x.size(2) != self.image_size or x.size(3) != self.image_size:
            raise ValueError(f'Expected input with width and height as {self.image_size}, found {x.size(2)} by {x.size(3)} ')
        if x.size(1) != 3:
            raise ValueError(f'Expected 3 channels found {x.size(1)}')
        x = self.conv(x)
        x = torch.flatten(x, start_dim=2)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.cls_token_embedding.unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.encoder(x)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.projection
        return x


EXPANSION = 4


class ResNetForCLIPBottleneck(nn.Module):

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * EXPANSION, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * EXPANSION)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * EXPANSION:
            self.downsample = nn.Sequential(OrderedDict([('-1', nn.AvgPool2d(stride)), ('0', nn.Conv2d(inplanes, planes * EXPANSION, 1, stride=1, bias=False)), ('1', nn.BatchNorm2d(planes * EXPANSION))]))

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self, spacial_dim: 'int', embed_dim: 'int', num_heads: 'int', output_dim: 'int'=None):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: 'Tensor') ->Tensor:
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1], num_heads=self.num_heads, q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight, in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias, use_separate_proj_weight=True, training=self.training, need_weights=False)
        return x[0]


class ResNetForCLIP(nn.Module):
    """Modified ResNet used by CLIP.

    Based on https://github.com/openai/CLIP/blob/main/clip/model.py#L93, this class
    differs from Torchvision's ResNet in the following ways:
    - There are now 3 "stem" convolutions as opposed to 1, with an
        average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is
        prepended to convolutions with stride > 1.
    - The final pooling layer is a QKV attention instead of an average pool.

    Args:
        layers (Tuple[int]): number of residual blocks in each stage.
            of the ResNet architecture
        output_dim (int): dimension of output tensor
        heads (int): number of heads in the attention pooling layer
        input_resolution (int): resolution of image input to encoder
        width (int): ResNet width
        use_clip_init (bool): Whether to use CLIP-specific initialization.

    Inputs:
        x (Tensor): Tensor containing image features
    """

    def __init__(self, layers: 'Tuple[int, int, int, int]'=(3, 4, 6, 3), output_dim: 'int'=512, heads: 'int'=1024, input_resolution: 'int'=224, width: 'int'=64, use_clip_init: 'bool'=True):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        if use_clip_init:
            self.initialize_parameters()

    def _make_layer(self, planes: 'int', blocks: 'int', stride: 'int'=1) ->nn.Module:
        layers = [ResNetForCLIPBottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * EXPANSION
        for _ in range(1, blocks):
            layers.append(ResNetForCLIPBottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def initialize_parameters(self) ->None:
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)
        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith('bn3.weight'):
                    nn.init.zeros_(param)

    def forward(self, x: 'Tensor') ->Tensor:

        def stem(x: 'Tensor') ->Tensor:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class CLIPOutput(NamedTuple):
    embeddings_a: 'torch.Tensor'
    embeddings_b: 'torch.Tensor'


class CLIP(nn.Module):
    """CLIP is a model for contrastive pretraining between two modalities.

    CLIP (https://arxiv.org/pdf/2103.00020.pdf) jointly trains an image encoder
    (either ResNet or ViT) and a text encoder (Transformer) to predict correct
    (image, text) pairings via a contrastive loss function. This module contains the
    encoders, while the loss is implemented in ContrastiveLossWithTemperature.


    Args:   encoder_a (nn.Module): Instantiated encoder for modality A.
                See e.g. ResNetForCLIP class.
            encoder_b (nn.Module): Instantiated encoder for modality B.
                See e.g. CLIPTextEncoder class.

    Inputs: features_a (Tensor): Tensor containing features of modality A.
            features_b (Tensor): Tensor containing features of modality B.
    """

    def __init__(self, encoder_a: 'nn.Module', encoder_b: 'nn.Module'):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b

    def forward(self, features_a: 'torch.Tensor', features_b: 'torch.Tensor') ->CLIPOutput:
        embeddings_a = self.encoder_a(features_a)
        embeddings_b = self.encoder_b(features_b)
        embeddings_a = F.normalize(embeddings_a)
        embeddings_b = F.normalize(embeddings_b)
        return CLIPOutput(embeddings_a=embeddings_a, embeddings_b=embeddings_b)


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder class. Should be instantiated and passed to
    CLIP (models/clip.py)

    As in CLIP, the text encoder follows a Transformer architecture.

    Args:
        embedding_dim (int): Embedding dimension for text and image encoders projections.
        context_length (int): Maximum sequence length for Transformer.
        vocab_size (int): Vocab size.
        width (int): Embedding dimension for Transformer encoder.
        dim_feedforward (int): Dimension of the feedfoward networks.
        heads (int): Number of heads in Transformer encoder.
        layers (int): Number of layers in Transformer encoder.
        use_clip_init (bool): Whether to use CLIP-specific initialization.

    Inputs:
        text (Tensor): Tensor containing text features.
        return_hidden_state (bool): If ``True``, returns the last hidden state
            instead of the final projected embeddings. Defaults to ``False``.
    """
    TOKEN_EMBEDDING_INIT_STD = 0.02
    POS_EMBEDDING_INIT_STD = 0.01

    def __init__(self, embedding_dim: 'int'=512, context_length: 'int'=77, vocab_size: 'int'=49408, width: 'int'=512, dim_feedforward: 'int'=2048, heads: 'int'=8, layers: 'int'=12, use_clip_init: 'bool'=True):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.token_embedding = torch.nn.Embedding(vocab_size, width)
        self.positional_embedding = torch.nn.Parameter(torch.empty(context_length, width))
        encoder_layer = TransformerEncoderLayer(d_model=width, dim_feedforward=dim_feedforward, nhead=heads, dropout=0.0, activation=SiLU(), norm_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=layers)
        self.width = width
        self.context_length = context_length
        self.ln_final = Fp32LayerNorm(width)
        self.projection = nn.Linear(width, embedding_dim, bias=False)
        self.mask = torch.full((self.context_length, self.context_length), float('-inf')).triu(1)
        if use_clip_init:
            self.initialize_parameters()

    def initialize_parameters(self) ->None:
        nn.init.normal_(self.token_embedding.weight, std=self.TOKEN_EMBEDDING_INIT_STD)
        nn.init.normal_(self.positional_embedding, std=self.POS_EMBEDDING_INIT_STD)
        proj_std = self.width ** -0.5 * (2 * self.encoder.num_layers) ** -0.5
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for layer in self.encoder.layers:
            nn.init.normal_(layer.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(layer.self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(layer.linear1.weight, std=fc_std)
            nn.init.normal_(layer.linear2.weight, std=proj_std)
        nn.init.normal_(self.projection.weight, std=self.width ** -0.5)

    def build_attention_mask(self) ->Tensor:
        mask = torch.full((self.context_length, self.context_length), float('-inf')).triu(1)
        return mask

    def forward(self, text: 'Tensor', return_hidden_state: 'bool'=False) ->Tensor:
        if text.size(1) != self.context_length:
            raise ValueError(f'length of input should be {self.context_length} but found {text.size(1)}')
        embeddings = self.token_embedding(text)
        embeddings = embeddings + self.positional_embedding
        embeddings = embeddings.permute(1, 0, 2)
        embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        embeddings = torch.permute(embeddings, (1, 0, 2))
        hidden_state = self.ln_final(embeddings)
        if return_hidden_state:
            return hidden_state
        projected_embeddings = self.projection(hidden_state[torch.arange(hidden_state.shape[0]), text.argmax(dim=-1)])
        return projected_embeddings


class MultimodalOutput(NamedTuple):
    image_pooled_output: 'Tensor'
    text_pooled_output: 'Tensor'
    multimodal_embeddings: 'Tensor'
    multimodal_pooled_embeddings: 'Optional[Tensor]' = None


class CoCaModel(nn.Module):
    """
    CoCa model class containing vision encoder, text decoder, and multimodal decoder.
    Reference: https://arxiv.org/abs/2205.01917
    Args:
        vision_encoder (nn.Module): Instantiated vision encoder. Should return either
            TransformerOutput or Tensor.
        text_decoder (CoCaTextDecoder): Instantiated CoCaTextDecoder returning a
            Tuple[Tensor, Tensor], where the first element is the normalized CLS
            embedding, and the second element is the full set of token embeddings.
        multimodal_decoder (nn.Module): Instantiated CoCaMultimodalDecoder returning a
            Tensor of multimodal embeddings.
        vision_pooler (nn.Module): Pooler for vision outputs (see e.g. AttentionPooler).
        vision_proj (nn.Module): Projection layer for vision encoder. Note that the
            projections for the text_decoder and multimodal_decoder are handled inside
            the CoCaTextDecoder and CoCaMultimodalDecoder classes, respectively, but
            for vision we apply attentional pooling first so the vision projection
            is separated from the vision_encoder class.
    """

    def __init__(self, vision_encoder: 'nn.Module', text_decoder: 'CoCaTextDecoder', multimodal_decoder: 'CoCaMultimodalDecoder', vision_pooler: 'nn.Module', vision_proj: 'nn.Module'):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_decoder = text_decoder
        self.multimodal_decoder = multimodal_decoder
        self.vision_pooler = vision_pooler
        self.vision_proj = vision_proj

    def forward(self, images: 'Tensor', texts: 'Tensor', text_padding_mask: 'Optional[Tensor]'=None) ->MultimodalOutput:
        """
        Args:
            images (Tensor): Tensor of size (bsz, c, h, w) containing image pixels.
            texts (Tensor): Tensor of size (bsz, seq_len) containing text tokens.
            text_padding_mask (Optional[Tensor]): Boolean mask indicating padded tokens.
                True for unpadded tokens, False for padded tokens. Default: None
        Returns:
            MultimodalOutput containing pooled image embeddings, text embeddings,
                and multimodal embeddings.
        """
        vision_encoder_outs = self.vision_encoder(images)
        if isinstance(vision_encoder_outs, TransformerOutput):
            image_embeddings = vision_encoder_outs.last_hidden_state
        elif isinstance(vision_encoder_outs, tuple):
            vision_encoder_outs = vision_encoder_outs[0]
            assert isinstance(vision_encoder_outs, Tensor)
            image_embeddings = vision_encoder_outs
        else:
            assert isinstance(vision_encoder_outs, Tensor)
            image_embeddings = vision_encoder_outs
        assert isinstance(image_embeddings, Tensor), 'Image embeddings must be Tensor'
        pooled_outputs = self.vision_pooler(image_embeddings)
        if torch.jit.isinstance(pooled_outputs, List[Tensor]):
            assert len(pooled_outputs) == 2
            captioning_image_embeddings, contrastive_image_embeddings = pooled_outputs
        else:
            assert isinstance(pooled_outputs, Tensor), 'Pooled image embeddings must be Tensor'
            contrastive_image_embeddings, captioning_image_embeddings = pooled_outputs[:, 0], pooled_outputs[:, 1:]
        contrastive_image_embeddings = self.vision_proj(contrastive_image_embeddings)
        contrastive_image_embeddings = F.normalize(contrastive_image_embeddings, dim=-1)
        pooled_text_embeddings, text_tokens = self.text_decoder(texts, text_padding_mask)
        contrastive_text_embeddings = F.normalize(pooled_text_embeddings, dim=-1)
        multimodal_embeddings = self.multimodal_decoder(text_tokens, captioning_image_embeddings)
        return MultimodalOutput(contrastive_image_embeddings, contrastive_text_embeddings, multimodal_embeddings)


DEFAULT_LOGIT_SCALE = math.log(1 / 0.07)


class ContrastiveLossWithTemperature(nn.Module):
    """Contrastive loss with a temperature parameter, as used in CLIP and FLAVA.
    CLIP: https://arxiv.org/pdf/2103.00020.pdf
    FLAVA: https://arxiv.org/pdf/2112.04482.pdf


    A contrastive loss over pairs of input embeddings a and b. For each input_a
    embedding, we compute a weighted cosine similarity with all input_b embeddings,
    then calculate the cross entropy loss against the true (input_a, input_b) pairing.
    Each input_b embedding is evaluated against all input_a embeddings similarly.
    The batch's loss is the average cross entropy over all input_a and input_b embeddings
    in the batch.

    Temperature is a learned parameter clamped to ``[1, 100]`` and
    initialized to 1 / 0.07 as in the CLIP paper.


    Args:
        logit_scale (Union[float, nn.Module]): Log of the learnable temperature parameter value
            A nn.Parameter instantiation can also be passed directly in case parent class
            is handling the initialization.
            Defaults to ``ln(1/0.07)``, as in the CLIP paper.
        logit_scale_min (Optional[float]): Log of the minimum temperature value.
            If ``None``, then temperature will not be clamped to a minimum value.
            Defaults to ``ln(1)``, as in the CLIP paper.
        logit_scale_max (Optional[float]): Log of the maximum temperature value.
            If ``None``, then temperature will not be clamped to a maximum value.
            Defaults to ``ln(100)``, as in the CLIP paper.

    Inputs: embeddings_a (Tensor): Tensor containing features from the first input or modality.
                (In the CLIP model, these are the outputs of the image encoder.)
            embeddings_b (Tensor): Tensor containing features from the second input or modality.
                (In the CLIP model, these are the outputs of the text encoder.)
            backprop_type (BackpropType): whether to backpropagate gradients to all
                workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
                Default: BackpropType.GLOBAL
            cross_entropy_kwargs (Optional[Dict[str, Any]]): Any additional inputs to cross entropy loss (ex: label_smoothing)
            mask (Optional[Tensor], optional): If certain elements of the inputs shouldn't
                be considered in the loss calculation use this option to pass a boolean
                mask. Size is (BatchSize,). Defaults to None.
    """

    def __init__(self, logit_scale: 'Union[float, nn.Parameter]'=DEFAULT_LOGIT_SCALE, logit_scale_min: 'Optional[float]'=math.log(1), logit_scale_max: 'Optional[float]'=math.log(100)):
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        if not logit_scale_min and not logit_scale_max:
            raise ValueError('Only one of `logit_scale_min` and `logit_scale_max` can be None.')
        self.logit_scale_min = logit_scale_min
        self.logit_scale_max = logit_scale_max
        if isinstance(logit_scale, nn.Parameter):
            self.logit_scale = logit_scale
        else:
            self.logit_scale = nn.Parameter(logit_scale * torch.ones([]))

    def forward(self, embeddings_a: 'Tensor', embeddings_b: 'Tensor', backprop_type: 'BackpropType'=BackpropType.GLOBAL, cross_entropy_kwargs: 'Optional[Dict[str, Any]]'=None, mask: 'Optional[Tensor]'=None) ->Tensor:
        self.logit_scale.data.clamp_(self.logit_scale_min, self.logit_scale_max)
        return contrastive_loss_with_temperature(embeddings_a=embeddings_a, embeddings_b=embeddings_b, logit_scale=self.logit_scale, backprop_type=backprop_type, cross_entropy_kwargs=cross_entropy_kwargs, mask=mask).loss


class CoCaForPretraining(nn.Module):
    """
    CoCa pretraining model class.
    Ties CoCa model to captioning and contrastive losses.
    Args:
        model (CoCaModel): Instantiated CoCa model.
        pad_idx (int): Index of padding tokens (used to filter captioning
        loss indices). Default: 0
        contrastive_logit_scale_min (Optional[float]): Min clamp value for contrastive
            temperature. Default: 0.0
        contrastive_logit_scale_max (Optional[float]): Max clamp value for contrastive
            temperature. Default: log(100)
    """

    def __init__(self, model: 'CoCaModel', pad_idx: 'int'=0, contrastive_logit_scale_min: 'Optional[float]'=math.log(1.0), contrastive_logit_scale_max: 'Optional[float]'=math.log(100.0)):
        super().__init__()
        self.model = model
        self.contrastive_loss = ContrastiveLossWithTemperature(logit_scale_min=contrastive_logit_scale_min, logit_scale_max=contrastive_logit_scale_max)
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, images: 'Tensor', texts: 'Tensor', text_padding_mask: 'Optional[Tensor]'=None) ->Dict[str, Tensor]:
        """
        Args:
            images (Tensor): Tensor of size (bsz, c, h, w) containing image pixels.
            texts (Tensor): Tensor of size (bsz, seq_len) containing text tokens.
            text_padding_mask (Optional[Tensor]): Boolean mask indicating padded tokens.
                True for unpadded tokens, False for padded tokens. Default: None
        Returns:
            Dict[str, Tensor]: Dict containing contrastive and captioning losses with
                respective keys 'contrastive' and 'captioning'.
        """
        model_outs = self.model(images, texts, text_padding_mask)
        captioning_labels = texts[:, 1:].contiguous()
        contrastive_loss = self.contrastive_loss(model_outs.image_pooled_output, model_outs.text_pooled_output)
        vocab_size = model_outs.multimodal_embeddings.shape[-1]
        captioning_loss = self.caption_loss(model_outs.multimodal_embeddings.view(-1, vocab_size), captioning_labels.view(-1))
        return {'contrastive': contrastive_loss, 'captioning': captioning_loss}


default_coca_cls_pooler = partial(torch.select, dim=1, index=-1)


class CoCaModelWithHeads(nn.Module):
    """
    CoCa model with heads.
    Args:
        model (CoCaModel): Instantiated CoCa model.
        heads (nn.ModuleDict): Dictionary of heads, taking either unimodal or
            multimodal embeddings as inputs
        pad_idx (int): Index of padding tokens (used to filter captioning
        loss indices). Default: 0
        pooler (Callable): how to extract the the multimodal embeddings. some examples
            [default] partial(torch.select, dim=1, index=-1)
            partial(torch.mean, dim=1)
            partial(torch.max, dim=1)
            torchmultimodal.fb.modules.layers.attention_pooler.AttentionPooler
    """

    def __init__(self, model: 'CoCaModel', heads: 'nn.ModuleDict', pad_idx: 'int'=0, pooler: 'Callable'=default_coca_cls_pooler):
        super().__init__()
        self.model = model
        self.heads = heads
        self.pooler = pooler

    def forward(self, images: 'Tensor', texts: 'Tensor', text_padding_mask: 'Optional[Tensor]'=None) ->Dict[str, Tensor]:
        model_out = self.model(images, texts, text_padding_mask)
        mm_out = model_out.multimodal_embeddings
        bsz = mm_out.shape[0]
        pooled_output = self.pooler(mm_out).view((bsz, -1))
        head_outputs = {}
        for k, head in self.heads.items():
            head_outputs[k] = head(pooled_output)
        return head_outputs


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer consisting of multihead self-attention, optional
    cross-attention, and feedforward network.

    Args:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network.
            Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention,
            (optional) cross-attention, and feedforward. Otherwise, layer norm is done
            after. Defaults to False.
        use_cross_attention (bool): if True, cross-attention is applied before
            feedforward network. If False, no cross-attention is applied.
            Defaults to True.
        dim_kv (Optional[int]): dimension for key and value tensors in cross-attention.
            If None, K and V are assumed to have dimension d_model. Defaults to None.
    """

    def __init__(self, d_model: 'int', n_head: 'int', dim_feedforward: 'int', dropout: 'float'=0.0, activation: 'Callable[..., nn.Module]'=nn.ReLU, layer_norm_eps: 'float'=1e-12, norm_first: 'bool'=False, use_cross_attention: 'bool'=True, dim_kv: 'Optional[int]'=None) ->None:
        super().__init__()
        if dim_kv is not None:
            dim_kv = dim_kv
        else:
            dim_kv = d_model
        self.attention = MultiHeadAttentionWithCache(dim_q=d_model, dim_kv=d_model, num_heads=n_head, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention: 'Optional[MultiHeadAttentionWithCache]' = None
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attention = MultiHeadAttentionWithCache(dim_q=d_model, dim_kv=dim_kv, num_heads=n_head, dropout=dropout)
            self.cross_attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
            self.cross_attention_dropout = nn.Dropout(dropout)
        self.feedforward = MLP(d_model, d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _self_attention_block(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None, past_key_value: 'Optional[Tuple[Tensor, Tensor]]'=None, use_cache: 'bool'=False) ->Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        attn_output = self.attention(query=hidden_states, key=hidden_states, value=hidden_states, attn_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        present_key_value: 'Optional[Tuple[Tensor, Tensor]]' = None
        if use_cache:
            assert isinstance(attn_output, MHAWithCacheOutput)
            attn_output_value = attn_output.attn_output
            present_key_value = attn_output.past_key_value
        else:
            assert isinstance(attn_output, Tensor)
            attn_output_value = attn_output
        output = self.attention_dropout(attn_output_value)
        return output, present_key_value

    def _cross_attention_block(self, hidden_states: 'Tensor', encoder_hidden_states: 'Tensor', cross_attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        assert self.cross_attention is not None, """
            Cannot use cross-attention unless self.cross_attention and
            self.cross_attention_dropout are defined.
        """
        output = self.cross_attention(query=hidden_states, key=encoder_hidden_states, value=encoder_hidden_states, attn_mask=cross_attention_mask, use_cache=False)
        assert isinstance(output, Tensor), 'cross-attention output must be Tensor.'
        attention_output = self.cross_attention_dropout(output)
        return attention_output

    def _feedforward_block(self, hidden_states: 'Tensor') ->Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(self, hidden_states: 'Tensor', encoder_hidden_states: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, cross_attention_mask: 'Optional[Tensor]'=None, past_key_value: 'Optional[Tuple[Tensor, Tensor]]'=None, use_cache: 'bool'=False) ->Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        self_attn_input = self.attention_layernorm(hidden_states)
        attn_output, present_key_value = self._self_attention_block(self_attn_input, attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        self_attn_output = attn_output + hidden_states
        if self.use_cross_attention and encoder_hidden_states is not None:
            assert hasattr(self, 'cross_attention_layernorm'), 'Cross-attention layernorm not initialized'
            cross_attn_input = self.cross_attention_layernorm(self_attn_output)
            cross_attn_output = self._cross_attention_block(cross_attn_input, encoder_hidden_states, cross_attention_mask=cross_attention_mask)
            attn_output = cross_attn_output + self_attn_output
        else:
            attn_output = self_attn_output
        ff_input = self.feedforward_layernorm(attn_output)
        output = attn_output + self._feedforward_block(ff_input)
        return output, present_key_value

    def _forward_postnorm(self, hidden_states: 'Tensor', encoder_hidden_states: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, cross_attention_mask: 'Optional[Tensor]'=None, past_key_value: 'Optional[Tuple[Tensor, Tensor]]'=None, use_cache: 'bool'=False) ->Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        attn_output, present_key_value = self._self_attention_block(hidden_states, attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        attn_residual = attn_output + hidden_states
        self_attn_output = self.attention_layernorm(attn_residual)
        if self.use_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError('encoder_hidden_states must be provided for cross attention')
            assert hasattr(self, 'cross_attention_layernorm'), 'Cross-attention layernorm not initialized'
            cross_attn_output = self._cross_attention_block(self_attn_output, encoder_hidden_states, cross_attention_mask)
            cross_attn_residual = cross_attn_output + self_attn_output
            attn_output = self.cross_attention_layernorm(cross_attn_residual)
        else:
            attn_output = self_attn_output
        ff_residual = attn_output + self._feedforward_block(attn_output)
        output = self.feedforward_layernorm(ff_residual)
        return output, present_key_value

    def forward(self, hidden_states: 'Tensor', encoder_hidden_states: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, cross_attention_mask: 'Optional[Tensor]'=None, past_key_value: 'Optional[Tuple[Tensor, Tensor]]'=None, use_cache: 'bool'=False) ->Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Inputs:
            hidden_states (Tensor): input query of shape bsz x seq_len x embed_dim
            encoder_hidden_states (Optional[Tensor]): input key/values of shape
                bsz x seq_len x embed_dim, only used for cross-attention.
                Default is None.
            attention_mask (Optional[Tensor]): attention mask for self-attention,
                supported mask type is described in MultiHeadAttentionWithCache class.
                Default is None.
            cross_attention_mask (Optional[Tensor]): attention mask for cross-attention,
                similar to attention_mask. Default is None.
            past_key_value (Optional[Tuple[Tensor, Tensor]]): cached key/value tuple
                for self-attention. Default is None.
            use_cache (bool): whether to use cache for key and value tensors.
                Can be used for faster autoregressive decoding during inference.
                    Default is False.

        Returns:
            A tuple including
                output (Tensor): layer output of shape bsz x seq_len x embed_dim
                present_key_value (Optional[Tuple[Tensor, Tensor]]): key/value tuple for
                    self-attention if use_cache set to True else None
        """
        if self.norm_first is True:
            return self._forward_prenorm(hidden_states, encoder_hidden_states, attention_mask, cross_attention_mask, past_key_value, use_cache)
        else:
            return self._forward_postnorm(hidden_states, encoder_hidden_states, attention_mask, cross_attention_mask, past_key_value, use_cache)


class TransformerDecoder(nn.Module):
    """
    Transformer decoder: n transformer decoder layers and an optional final LN

    Args:
        n_layer (int): number of transformer decoder layers
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network.
            Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention,
            (optional) cross-attention, and feedforward. Otherwise, layer norm is done
            after. Defaults to False.
        use_cross_attention (bool): if True, cross-attention is applied before
            feedforward network. If False, no cross-attention is applied.
            Defaults to True.
        dim_kv (Optional[int]): dimension for key and value tensors in cross-attention.
            If None, K and V are assumed to have dimension d_model. Defaults to None.
        final_layer_norm_eps (Optional[float]): epsilon used in final layer norm.
            Defaults to None (no final layer norm).
        cross_attention_interval: interval layers to apply cross attention. Not used if
            use_cross_attention = False
    """

    def __init__(self, n_layer: 'int', d_model: 'int', n_head: 'int', dim_feedforward: 'int', dropout: 'float'=0.0, activation: 'Callable[..., nn.Module]'=nn.ReLU, layer_norm_eps: 'float'=1e-12, norm_first: 'bool'=False, use_cross_attention: 'bool'=True, dim_kv: 'Optional[int]'=None, final_layer_norm_eps: 'Optional[float]'=None, cross_attention_interval: 'int'=1):
        super().__init__()
        self.layer = nn.ModuleList([TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, activation, layer_norm_eps, norm_first, use_cross_attention and i % cross_attention_interval == 0, dim_kv) for i in range(n_layer)])
        self.final_layer_norm = None
        if final_layer_norm_eps:
            self.final_layer_norm = Fp32LayerNorm(d_model, eps=final_layer_norm_eps)

    def forward(self, hidden_states: 'Tensor', encoder_hidden_states: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None, cross_attention_mask: 'Optional[Tensor]'=None, past_key_values: 'Optional[List[Tuple[Tensor, Tensor]]]'=None, use_cache: 'bool'=False, return_hidden_states: 'bool'=False) ->TransformerOutput:
        """
        Inputs:
            hidden_states (Tensor): input query of shape bsz x seq_len x embed_dim
            encoder_hidden_states (Optional[Tensor]): input key/values of shape
                bsz x seq_len x embed_dim, only used for cross-attention.
                Default is None.
            attention_mask (Optional[Tensor]): attention mask for self-attention,
                supported mask type is described in MultiHeadAttentionWithCache class.
                Default is None.
            cross_attention_mask (Optional[Tensor]): attention mask for cross-attention,
                similar to attention_mask. Default is None.
            past_key_values (Optional[List[Tuple[Tensor, Tensor]]]): cached key/value
                tuples for self-attention in each layer. Default is None.
            use_cache (bool): whether to use cache for key and value tensors.
                Default is False.
            return_hidden_states (bool): if True, return output from each layer of
                transformer including the input to first layer. Default is False.

        Returns:
            output of TransformerOutput type with fields
                last_hidden_state (Tensor): layer output of shape bsz x seq_len x embed_dim
                hidden_states (List[Tensor]): all hidden states from decoder layers
                present_key_value (Optional[Tuple[Tensor, Tensor]]): key/value tuple
                    for self-attention.
        """
        all_hidden_states = []
        current_key_values = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        for i, layer_module in enumerate(self.layer):
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs, current_key_value = layer_module(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache)
            if use_cache:
                assert isinstance(current_key_value, tuple)
                current_key_values.append(current_key_value)
            hidden_states = layer_outputs
        if return_hidden_states:
            all_hidden_states.append(hidden_states)
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        return TransformerOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, current_key_values=current_key_values)


class CoCaMultimodalDecoder(nn.Module):
    """
    Multimodal decoder for CoCa model.
    Uses a transformer decoder with causal mask for text embeddings
    that cross-attends to image embeddings, followed by output projection.
    Based on the implementation in open_clip: https://tinyurl.com/mn35vdmd

    Args:
        input_seq_len (int): Number of text positions (used to construct
            causal mask)
        text_embedding_dim (int): Dimension of text embeddings
            inside transformer decoder.
        n_layer (int): Number of transformer layers
        n_head (int): Number of heads in multi-head attention
        dim_feedforward (int): Dimension of FFN in transformer decoder
        dropout (float): Dropout probability in transformer decoder. Default: 0.0
        activation (Callable[..., nn.Module]): Activation function of transformer
            decoder. Default: nn.GELU
        layer_norm_eps (float): Epsilon value for transformer decoder layer norms.
            Default: 1e-5
        norm_first (bool): Whether to apply layer normalization before or after
            self-attention in transformer decoder. Default: True
        final_layer_norm_eps (Optional[float]): Regularization value for final layer norm
            in transformer decoder. Default: 1e-5
        visual_embedding_dim (Optional[int]): Dimension of visual embeddings inside
            transformer decoder (used for cross-attention). Default: None (visual
            embeddings assumed to be same dimension as text embeddings)
    """

    def __init__(self, input_seq_len: 'int', text_embedding_dim: 'int', n_layer: 'int', n_head: 'int', dim_feedforward: 'int', output_dim: 'Optional[int]'=None, dropout: 'float'=0.0, activation: 'Callable[..., nn.Module]'=nn.GELU, layer_norm_eps: 'float'=1e-05, norm_first: 'bool'=True, final_layer_norm_eps: 'Optional[float]'=1e-05, visual_embedding_dim: 'Optional[int]'=None):
        super().__init__()
        self.transformer_decoder = TransformerDecoder(n_layer=n_layer, d_model=text_embedding_dim, n_head=n_head, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first, use_cross_attention=True, final_layer_norm_eps=final_layer_norm_eps, dim_kv=visual_embedding_dim)
        if output_dim is not None:
            self.output_projection = nn.Linear(text_embedding_dim, output_dim, bias=False)
        else:
            self.output_projection = None
        self.register_buffer('causal_mask', get_causal_attention_mask(input_seq_len), persistent=False)

    def forward(self, texts: 'Tensor', images: 'Tensor') ->Tensor:
        """
        Args:
            texts (Tensor): Tensor containing text embeddings of shape [batch_size, text_seq_length, embeddings_dim]
            images (Tensor): Tensor containing image embeddings of shape [batch_size, image_seq_length, embeddings_dim]
            text_causal_mask (Tensor): Tensor containing causal mask of shape [text_seq_length, text_seq_length]
        Returns:
        Tensor: Tensor containing output embeddings of shape [batch_size, text_seq_length, output_dim]
        """
        seq_len = texts.shape[1]
        assert self.causal_mask.shape == (seq_len, seq_len)
        decoder_outputs = self.transformer_decoder(hidden_states=texts, encoder_hidden_states=images, attention_mask=self.causal_mask)
        hidden_states = decoder_outputs.last_hidden_state
        assert hidden_states is not None, 'hidden states must not be None'
        if self.output_projection is not None:
            out = self.output_projection(hidden_states)
        else:
            out = hidden_states
        return out


class CoCaTextEmbeddings(nn.Module):
    """
    Text embeddings for CoCa model. Includes token embeddings, positional embeddings,
    and optional CLS embedding.

    Args:
        vocab_size (int): Size of the vocab
        num_positions (int): Number of token positions for positional embeddings
            not including cls.
        embedding_dim (int): Output embedding dimension
        pad_idx (Optional[int]): Padding index to be ignored by token embeddings.
            Default: 0
        embed_cls (bool): Whether to include CLS embedding. Default: True
    """

    def __init__(self, vocab_size: 'int', num_positions: 'int', embedding_dim: 'int', pad_idx: 'Optional[int]'=0, embed_cls: 'bool'=True):
        super().__init__()
        self.num_positions = num_positions
        if embed_cls:
            self.cls_embedding = nn.Parameter(torch.empty(embedding_dim))
        else:
            self.cls_embedding = None
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim, pad_idx)
        self.position_embeddings = nn.Parameter(torch.empty(num_positions, embedding_dim))
        self.init_parameters()

    def init_parameters(self) ->None:
        nn.init.normal_(self.token_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.01)
        if self.cls_embedding is not None:
            nn.init.constant_(self.cls_embedding, 0.01)

    def forward(self, input_ids: 'Tensor') ->Tensor:
        """
        Args:
            input_ids (Tensor of size (batch_size, seq_length)):
                Indices of input sequence tokens.
        Returns:
            Tensor of size (batch_size, seq_length, embedding_dim)
        """
        assert input_ids.shape[1] == (self.num_positions if self.cls_embedding is None else self.num_positions - 1)
        embeddings = self.token_embeddings(input_ids)
        if self.cls_embedding is not None:
            cls_embed = self.cls_embedding.reshape(1, 1, -1).repeat(input_ids.shape[0], 1, 1)
            embeddings = torch.cat([embeddings, cls_embed], dim=1)
        embeddings = embeddings + self.position_embeddings
        return embeddings


class CoCaTextDecoder(nn.Module):
    """
    Text decoder for CoCa model.
    Based on the implementation in open_clip: https://tinyurl.com/2jswrb9h

    Args:
        vocab_size (int): Size of the vocab
        num_positions (int): Number of token positions for positional embeddings.
        embedding_dim (int): Embedding dimension for transformer
        n_layer (int): Number of transformer layers
        n_head (int): Number of attention heads
        dim_feedforward (int): Hidden dimension in transformer FFN
        output_dim (int): Output dimension of decoder cls / eos projection
        pad_idx (Optional[int]): Padding index (will be masked from CLS token).
            Default: 0
        embed_cls (bool): Whether to append CLS embedding. Default: True
        dropout (float): Dropout probability in transformer decoder
        activation (Callable[..., nn.Module]): Activation function of transformer
            decoder. Default: nn.GELU
        layer_norm_eps (float): Epsilon value for transformer decoder layer norms.
            Default: 1e-5
        norm_first (bool): Whether to apply layer normalization before or after
            self-attention in transformer decoder. Default: True
        final_layer_norm_eps (Optional[float]): Final layer norm epsilon. Only applied
            to CLS token if embed_cls=True. Default: 1e-5
    """

    def __init__(self, vocab_size: 'int', num_positions: 'int', embedding_dim: 'int', n_layer: 'int', n_head: 'int', dim_feedforward: 'int', output_dim: 'int', pad_idx: 'Optional[int]'=0, embed_cls: 'bool'=True, dropout: 'float'=0.0, activation: 'Callable[..., nn.Module]'=nn.GELU, layer_norm_eps: 'float'=1e-05, norm_first: 'bool'=True, final_layer_norm_eps: 'Optional[float]'=1e-05):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed_cls = embed_cls
        self.num_positions = num_positions
        self.embeddings = CoCaTextEmbeddings(vocab_size=vocab_size, num_positions=num_positions, embedding_dim=embedding_dim, pad_idx=pad_idx, embed_cls=embed_cls)
        self.transformer_decoder = TransformerDecoder(n_layer=n_layer, d_model=embedding_dim, n_head=n_head, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, norm_first=norm_first, use_cross_attention=False)
        if final_layer_norm_eps is not None:
            self.ln_final = nn.LayerNorm(normalized_shape=embedding_dim, eps=final_layer_norm_eps)
        self.text_projection = nn.Linear(embedding_dim, output_dim, bias=False)
        self.register_buffer('causal_mask', get_causal_attention_mask(num_positions), persistent=False)
        self.init_parameters(embedding_dim, n_layer)

    def init_parameters(self, embedding_dim: 'int', n_layer: 'int') ->None:
        attn_std = embedding_dim ** -0.5
        proj_std = (2 * embedding_dim * n_layer) ** -0.5
        fc_std = (2 * embedding_dim) ** -0.5
        for layer in self.transformer_decoder.layer:
            nn.init.normal_(layer.attention.q_proj.weight, std=attn_std)
            nn.init.normal_(layer.attention.k_proj.weight, std=attn_std)
            nn.init.normal_(layer.attention.v_proj.weight, std=attn_std)
            nn.init.normal_(layer.attention.output_proj.weight, std=proj_std)
            nn.init.normal_(layer.feedforward.model[0].weight, std=fc_std)
            nn.init.normal_(layer.feedforward.model[2].weight, std=proj_std)
        nn.init.normal_(self.text_projection.weight, std=embedding_dim ** 0.5)

    def build_mask(self, input_ids: 'Tensor', padding_mask: 'Optional[Tensor]'=None) ->Tensor:
        if not self.embed_cls or self.pad_idx is None:
            return self.causal_mask
        if padding_mask is None:
            padding_mask = input_ids != self.pad_idx
        assert padding_mask is not None
        padding_mask = padding_mask.unsqueeze(1)
        padding_mask = F.pad(padding_mask, (1, 0, padding_mask.shape[2], 0), value=1.0)
        mask = (padding_mask * self.causal_mask).unsqueeze(1)
        return mask

    def forward(self, input_ids: 'Tensor', padding_mask: 'Optional[Tensor]'=None) ->Tuple[Tensor, Tensor]:
        """
        Args:
            input_ids (Tensor of size (batch_size, seq_length)):
                Indices of input sequence tokens.
            padding_mask (Optional[Tensor] of size (batch_size, seq_length)):
                Boolean tensor: True for unpadded tokens, False for padded tokens.
        Returns:
            A tuple including
                pooled (Tensor): Normalized CLS embedding of shape
                    (batch_size, output_dim) (for use in contrastive loss).
                tokens (Tensor): Embeddings for all non-CLS tokens. Shape:
                    (batch_size, num_positions, output_dim).
        """
        if self.embed_cls:
            if input_ids.shape[1] == self.num_positions:
                input_ids = input_ids[:, :-1]
            if padding_mask is not None and padding_mask.shape[1] == self.num_positions:
                padding_mask = padding_mask[:, :-1]
        target_shape = self.num_positions - 1 if self.embed_cls else self.num_positions
        assert input_ids.shape[1] == target_shape, f"{input_ids.shape} doesn't match ({target_shape},*)"
        embeddings = self.embeddings(input_ids)
        mask = self.build_mask(input_ids, padding_mask)
        decoder_out = self.transformer_decoder(embeddings, attention_mask=mask)
        hidden_states = decoder_out.last_hidden_state
        assert hidden_states is not None, 'hidden states must not be None'
        if self.embed_cls:
            pooled, tokens = hidden_states[:, -1], hidden_states[:, :-1]
            if self.ln_final is not None:
                pooled = self.ln_final(pooled)
        else:
            hidden_states = self.ln_final(hidden_states)
            pooled, tokens = hidden_states[torch.arange(hidden_states.shape[0]), input_ids.argmax(dim=-1)], hidden_states
        if self.text_projection is not None:
            pooled = self.text_projection(pooled)
        return pooled, tokens


class ImageTransformerWithVAE(nn.Module):

    def __init__(self, image_transformer: 'nn.Module', vae: 'nn.Module', **kwargs: Dict[str, Any]) ->None:
        super().__init__()
        self.image_transformer = image_transformer
        self.vae = vae

    def forward(self, pixel_values: 'Optional[Tensor]'=None, image_patches_mask: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None) ->TransformerOutput:
        image_labels = self.vae(pixel_values).flatten(1)
        image_patches_mask = image_patches_mask.flatten(1)
        image_labels[image_patches_mask == False] = -1
        output = self.image_transformer(pixel_values=pixel_values, image_patches_mask=image_patches_mask, attention_mask=attention_mask)
        return TransformerOutput(last_hidden_state=output.last_hidden_state, pooler_output=output.pooler_output, hidden_states=output.hidden_states, attentions=output.attentions)


class MultiHeadSelfAttention(nn.Module):
    """
    Multihead self attention.
    Similar to the self attention variant of MHA in attention.py but uses the scaled_dot_product_attention from PyTorch
    (which uses flash or memory efficient version for certain conditions).
    TODO: merge this into attention.py once other models are ready to use it.

    Args:
        embed_dim (int): embedding dimension of the input
        num_heads (int): number of attn heads
        dropout (float): dropout rate
    """

    def __init__(self, embed_dim: 'int', num_heads: 'int', dropout: 'float'=0.0):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, query: 'Tensor', attn_mask: 'Optional[Tensor]'=None, is_causal: 'bool'=False) ->Tensor:
        """
        Args:
            query (Tensor): input query of shape bsz x seq_len x embed_dim
            attn_mask (optional Tensor): attention mask of shape bsz x num_heads x seq_len x seq_len.
            Note that the num_heads dimension can equal 1 and the mask will be broadcasted to all heads.
            Two types of masks are supported.
            A boolean mask where a value of True indicates that the element should take part in attention.
            A float mask of the same type as query that is added to the attention score.
            is_causal (bool): If true, does causal attention masking. attn_mask should be set to None if this is set to True

        Returns:
            attention output Tensor of shape bsz x seq_len x embed_dim
        """
        bsz = query.size(0)
        embed_dim = query.size(-1)
        projected_query = self.input_proj(query)
        query, key, value = projected_query.chunk(3, dim=-1)
        head_dim = embed_dim // self.num_heads
        query = query.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        dropout = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(query, key, value, attn_mask, dropout, is_causal)
        attn = attn.transpose(1, 2).reshape(bsz, -1, embed_dim)
        attn_out = self.output_proj(attn)
        return attn_out


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer: transformer block consisting of multihead self-attention and feedforward blocks,
    based on "Attention Is All You Need" (Vaswani et al. 2017).

    Args:
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention
            and feedforward. Otherwise, layer norm is done after. Defaults to False
        drop_path_rate (Optional[float]): use stochastic drop path instead of dropout for attn and feedforward dropout
        in transformer block as used by vision transformers https://arxiv.org/pdf/1603.09382.pdf. Defaults to None.
    """

    def __init__(self, d_model: 'int', n_head: 'int', dim_feedforward: 'int', dropout: 'float'=0.0, activation: 'Callable[..., nn.Module]'=nn.ReLU, layer_norm_eps: 'float'=1e-12, norm_first: 'bool'=False, drop_path_rate: 'Optional[float]'=None) ->None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim=d_model, num_heads=n_head)
        if drop_path_rate is not None:
            self.attention_dropout = self.feedforward_dropout = StochasticDepth(drop_path_rate, mode='row')
        else:
            self.attention_dropout = nn.Dropout(dropout)
            self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward = MLP(d_model, d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.attention_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.feedforward_layernorm = Fp32LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_first = norm_first

    def _attention_block(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        output = self.attention(hidden_states, attn_mask=attention_mask)
        output = self.attention_dropout(output)
        return output

    def _feedforward_block(self, hidden_states: 'Tensor') ->Tensor:
        h = self.feedforward(hidden_states)
        h = self.feedforward_dropout(h)
        return h

    def _forward_prenorm(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        x = hidden_states
        inputs = self.attention_layernorm(x)
        attn_output = self._attention_block(inputs, attention_mask=attention_mask)
        attn_residual = attn_output + x
        ff_residual = attn_residual + self._feedforward_block(self.feedforward_layernorm(attn_residual))
        return ff_residual

    def _forward_postnorm(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        x = hidden_states
        attn_output = self._attention_block(x, attention_mask=attention_mask)
        attn_residual = attn_output + x
        attn_residual = self.attention_layernorm(attn_residual)
        ff_residual = attn_residual + self._feedforward_block(attn_residual)
        outputs = self.feedforward_layernorm(ff_residual)
        return outputs

    def forward(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None) ->Tensor:
        """
        Args:
            hidden_states (Tensor): input to the transformer encoder layer of shape bsz x seq_len x d_model
            attention_mask (Optional[Tensor]): attention mask of shape bsz x seq_len x seq_len.
            Same format as MultiHeadSelfAttention class.

        Returns:
            output tensor of shape bsz x seq_len x d_model
        """
        if self.norm_first is True:
            return self._forward_prenorm(hidden_states, attention_mask)
        else:
            return self._forward_postnorm(hidden_states, attention_mask)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of n Transformer encoder layers and an optional final LN

    Args:
        n_layer (int): number of Transformer encoder layers
        d_model (int): size of hidden dimension of input
        n_head (int): number of attention heads
        dim_feedforward (int): size of hidden dimension of feedforward network
        dropout (float): dropout probability for all dropouts. Defaults to 0.
        activation (Callable): activation function in feedforward network. Defaults to ``nn.ReLU``.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-12.
        norm_first (bool): if True, layer norm is done prior to each of self-attention
            and feedforward. Otherwise, layer norm is done after. Defaults to False
        final_layer_norm_eps (Optional[float]): eps for final layer norm. Defaults to None.
        drop_path_rate (Optional[float]): use stochastic drop path instead of dropout for attn and feedforward dropout
        in transformer block sometimes used by vision transformers https://arxiv.org/pdf/1603.09382.pdf. Defaults to None.
    """

    def __init__(self, n_layer: 'int', d_model: 'int', n_head: 'int', dim_feedforward: 'int', dropout: 'float'=0.0, activation: 'Callable[..., nn.Module]'=nn.ReLU, layer_norm_eps: 'float'=1e-12, norm_first: 'bool'=False, final_layer_norm_eps: 'Optional[float]'=None, drop_path_rate: 'Optional[float]'=None):
        super().__init__()
        if drop_path_rate is not None:
            drop_rate = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        else:
            drop_rate = [None for _ in range(n_layer)]
        self.layer = nn.ModuleList([TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation, layer_norm_eps, norm_first, drop_rate[i]) for i in range(n_layer)])
        self.final_layer_norm = None
        if final_layer_norm_eps:
            self.final_layer_norm = Fp32LayerNorm(d_model, eps=final_layer_norm_eps)

    def forward(self, hidden_states: 'Tensor', attention_mask: 'Optional[Tensor]'=None, return_hidden_states: 'bool'=False) ->TransformerOutput:
        """
        Args:
            hidden_states (Tensor): input to the transformer encoder of shape bsz x seq_len x d_model
            attention_mask (Optional[Tensor]): attention mask of shape bsz x seq_len x seq_len.
            Same format as MultiHeadSelfAttention class.
            return_hidden_states (bool): if True, return output from each layer of transformer including the input to first layer.
            Defaults to False.

        Returns:
            output of TransformerOutput type with the final output in last_hidden_state field.
            If return_hidden_states is set to True, the hidden_states field contains list of n_layer + 1 layer outputs.
            The last entry in the list is the output from last encoder block before final ln has been applied.
        """
        all_hidden_states = []
        for layer_module in self.layer:
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs
        if return_hidden_states:
            all_hidden_states.append(hidden_states)
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        return TransformerOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states if return_hidden_states else None)


class LateFusion(nn.Module):
    """A generic architecture for late fusion multimodal models.

    A late fusion model contains separate encoders for each modality,
    followed by a fusion layer and then a head module. For an example of a
    late fusion model, see the TorchMultimodal implementation of the cnn-lstm
    multimodal classifier (cnn_lstm.py)

    Args:
        encoders (ModuleDict): Dictionary mapping modalities to their respective
            encoders.

    Inputs:
        modalities (Dict[str, Tensor]): A dictionary mapping modalities to
            their tensor representations.
    """

    def __init__(self, encoders: 'nn.ModuleDict', fusion_module: 'nn.Module', head_module: 'nn.Module'):
        super().__init__()
        self.encoders = nn.ModuleDict({k: encoders[k] for k in sorted(encoders.keys())})
        self.fusion_module = fusion_module
        self.head_module = head_module

    def forward(self, modalities: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        embeddings = {}
        for key, encoder in self.encoders.items():
            assert key in modalities, f'{key} missing in input'
            embeddings[key] = encoder(modalities[key])
        fused = self.fusion_module(embeddings)
        return self.head_module(fused)


class DecoderEmbeddings(nn.Module):
    """
    Construct the decoder embeddings from encoder embeddings.
    Args:
        encoder_embed_dim (int): Input dim for decoder embedding i.e. output dim of the encoder.
        decoder_embed_dim (int): output dim for decoder embedding.
        image_size (Union[int, Tuple[int, int]]): Size of the original input image. If set to an int, we assume a square input.
         Defaults to 224.
        patch_size (int): Patch size for the decoder.
    """

    def __init__(self, encoder_embed_dim: 'int', decoder_embed_dim: 'int', image_size: 'Union[int, Tuple[int, int]]'=224, patch_size: 'int'=16) ->None:
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        if isinstance(image_size, int):
            image_size = image_size, image_size
        num_patches = image_size[0] // patch_size * (image_size[1] // patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

    def forward(self, x: 'Tensor', ids_restore: 'Tensor') ->Tensor:
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.position_embeddings
        return x


class MAEOutput(NamedTuple):
    encoder_output: 'Union[TransformerOutput, Tensor]'
    decoder_pred: 'Optional[Tensor]' = None
    label_patches: 'Optional[Tensor]' = None
    mask: 'Optional[Tensor]' = None


def get_1d_sin_cos_embeddings(embed_dim: 'int', positions: 'Tensor') ->Tensor:
    """
    1d position sin cos embeddings.
    Args:
        embed_dim (int): embedding dimension of the position embedding
        positions (Tensor): 1d tensor with the position ids
    """
    omega = 1 / 10000 ** (torch.arange(embed_dim // 2, dtype=torch.float) / (embed_dim / 2.0))
    out = torch.einsum('i,j->ij', positions, omega)
    sin_embed = torch.sin(out)
    cos_embed = torch.cos(out)
    embed = torch.cat([sin_embed, cos_embed], dim=1)
    return embed


def get_2d_sin_cos_embeddings(embed_dim: 'int', input_size: 'Tuple[int, int]', include_cls_embed: 'bool'=True) ->Tensor:
    """
    2d position sin cos embeddings.
    Args:
        embed_dim (int): embedding dimension of the position embedding
        input_size (Tuple[int, int]): input dimensions of the grid
        include_cls_embed (bool): Whether to include positional embedding for [CLS] token. Defaults to True.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f'embed_dim must be divisible by 4, got {embed_dim}')
    h, w = input_size
    pos_h = torch.arange(h)
    pos_w = torch.arange(w)
    pos_grid = torch.meshgrid(pos_w, pos_h, indexing='xy')
    embed_w = get_1d_sin_cos_embeddings(embed_dim // 2, pos_grid[0].flatten())
    embed_h = get_1d_sin_cos_embeddings(embed_dim // 2, pos_grid[1].flatten())
    embed = torch.cat([embed_w, embed_h], dim=1)
    if include_cls_embed:
        embed = torch.cat([torch.zeros(1, embed_dim), embed], dim=0)
    embed = embed.unsqueeze(0)
    return embed


class MaskedAutoEncoder(nn.Module):
    """
    MAE (https://arxiv.org/abs/2111.06377) is a pretraining technique to mask out patches of the input
    before passing through the encoder and then using a decoder to predict the masked patches
    The code has been adapted from the original implementation https://github.com/facebookresearch/mae

    Args:
        encoder_transformer (nn.Module): instance of encoder transformer
        decoder_transformer (nn.Module): instance of decoder transformer
        input_size (Union[int, Tuple[int,int]): size of the input. if tuple, the format should be height,width.
        If an int, a square input is assumed. Default: 224
        patch_size (int): size of the patches. Default: 16
        num_channels (int): number of input channels. Default: 3
        embed_dim (int): embedding dim of input to the encoder transformer (or output dim of patch embedding). Default: 768
        masking_ratio (float): ratio of patches to mask. Default: 0.75
        decoder_embed_dim (int): embedding dim of the input to the decoder transformer. Default: 512
    """

    def __init__(self, encoder_transformer: 'nn.Module', decoder_transformer: 'nn.Module', input_size: 'Union[int, Tuple[int, int]]'=224, patch_size: 'int'=16, num_channels: 'int'=3, embed_dim: 'int'=768, masking_ratio: 'float'=0.75, decoder_embed_dim: 'int'=512, use_cls_in_decoder: 'bool'=True):
        super().__init__()
        self.patch_size = patch_size
        self.embeddings = PatchEmbeddings(image_size=input_size, patch_size=patch_size, num_channels=num_channels, hidden_size=embed_dim, patch_drop_rate=masking_ratio)
        self.embeddings.position_embeddings.requires_grad = False
        self.encoder = encoder_transformer
        self.decoder_embed = DecoderEmbeddings(encoder_embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim, image_size=input_size, patch_size=patch_size)
        self.decoder_embed.position_embeddings.requires_grad = False
        self.decoder_transformer = decoder_transformer
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * num_channels)
        self.use_cls_in_decoder = use_cls_in_decoder
        self._initialize_weights(input_size, embed_dim, decoder_embed_dim)

    def _initialize_weights(self, input_size: 'Union[int, Tuple[int, int]]', encoder_embed_dim: 'int', decoder_embed_dim: 'int') ->None:
        if isinstance(input_size, int):
            input_h = input_w = input_size
        else:
            input_h, input_w = input_size
        num_patches_h = input_h // self.patch_size
        num_patches_w = input_w // self.patch_size
        self.embeddings.position_embeddings.data = get_2d_sin_cos_embeddings(encoder_embed_dim, (num_patches_w, num_patches_h))
        self.decoder_embed.position_embeddings.data = get_2d_sin_cos_embeddings(decoder_embed_dim, (num_patches_w, num_patches_h))
        w = self.embeddings.conv_projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.embeddings.cls_token, std=0.02)
        torch.nn.init.normal_(self.decoder_embed.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: 'nn.Module') ->None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _patchify_input(self, x: 'Tensor') ->Tensor:
        bsz, channels, height, width = x.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        label_patches = x.reshape(bsz, channels, num_patches_h, self.patch_size, num_patches_w, self.patch_size)
        label_patches = torch.einsum('nchpwq->nhwpqc', label_patches)
        label_patches = label_patches.reshape(bsz, num_patches_h * num_patches_w, self.patch_size ** 2 * channels)
        return label_patches

    def forward(self, x: 'Tensor') ->MAEOutput:
        """
        Args:
            x (Tensor): input tensor with shape bsz x channels x height x width
        Returns:
            output of MAEOutput type where encoder_output gives the output from the encoder,
            decoder_pred gives prediction from the decoder followed by linear head,
            mask indicates the masked out patches i.e. 1 refers to masked patches and 0 refers to unmasked patches
            label_patches indicates the patchified ground truth pixels

        """
        embedding_out = self.embeddings(x)
        encoder_out = self.encoder(embedding_out.embeddings)
        if not self.training:
            return MAEOutput(encoder_out)
        decoder_embedding = self.decoder_embed(encoder_out.last_hidden_state, embedding_out.ids_restore)
        decoder_input = decoder_embedding
        if not self.use_cls_in_decoder:
            decoder_input = decoder_input[:, 1:, :]
        decoder_out = self.decoder_transformer(decoder_input)
        pred = self.decoder_pred(decoder_out.last_hidden_state)
        if self.use_cls_in_decoder:
            pred = pred[:, 1:, :]
        label_patches = self._patchify_input(x)
        return MAEOutput(encoder_output=encoder_out, decoder_pred=pred, label_patches=label_patches, mask=embedding_out.random_mask)


class WindowMultiHeadAttention(nn.Module):
    """
    Window based attention as used by swin v2 https://arxiv.org/pdf/2111.09883.pdf

    Args:
        input_dim (int): input feature dimension
        num_heads (int): number of attention heads
        window_size (Tuple[int, int]): dimension of the window for local attention.
        attn_dropout (float): dropout probability for attention weights. Defaults to 0.
        proj_dropout (float): dropout probability for attention output projection. Defaults to 0.
        meta_hidden_dim (int): hidden dim for the mlp for relative position bias. Default is 384.
    """

    def __init__(self, input_dim: 'int', num_heads: 'int', window_size: 'Tuple[int, int]', attn_dropout: 'float'=0.0, proj_dropout: 'float'=0.0, meta_hidden_dim: 'int'=384, meta_mlp_dropout: 'float'=0.1) ->None:
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(in_features=input_dim, out_features=input_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.meta_mlp = MLP(in_dim=2, hidden_dims=meta_hidden_dim, out_dim=num_heads, activation=nn.ReLU, dropout=meta_mlp_dropout)
        self.register_parameter('tau', nn.Parameter(torch.ones(num_heads)))
        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) ->None:
        device = self.tau.device
        coordinates = torch.stack(torch.meshgrid([torch.arange(self.window_size[0], device=device), torch.arange(self.window_size[1], device=device)]), dim=0).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(1.0 + relative_coordinates.abs())
        self.register_buffer('relative_coordinates_log', relative_coordinates_log, persistent=False)

    def _relative_positional_encodings(self) ->Tensor:
        window_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(self.num_heads, window_area, window_area)
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def forward(self, x: 'torch.Tensor', mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Args:
            x (Tensor): input to the attention block of shape bsz x seq_len x input_dim.
            seq_len should match number of patches in the window
            mask (Optional[Tensor]): attention mask of shape total_num_window x seq_len x seq_len. Defaults to None.

        Returns:
            Tensor of shape bsz x seq_len x input_dim
        """
        bsz, seq_len, embed_dim = x.shape
        if seq_len != self.window_size[0] * self.window_size[1]:
            raise ValueError(f'Input sequence length {seq_len} needs to match window area')
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        denom = torch.linalg.vector_norm(query, dim=-1, keepdim=True) @ torch.linalg.vector_norm(key, dim=-1, keepdim=True).transpose(-2, -1)
        attn = query @ key.transpose(-2, -1) / denom.clamp(min=1e-06)
        attn = attn / self.tau.clamp(min=0.01).reshape(1, self.num_heads, 1, 1)
        attn = attn + self._relative_positional_encodings()
        if mask is not None:
            num_win: 'int' = mask.shape[0]
            attn = attn.view(bsz // num_win, num_win, self.num_heads, seq_len, seq_len)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, seq_len, seq_len)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(bsz, seq_len, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin transformer block customized for audio mae and loosely following swin v2 https://arxiv.org/pdf/2111.09883.pdf

    Args:
        input_dim (int): input feature dimension
        num_heads (int): number of attention heads
        input_size (Tuple[int, int]): dimension of the original input before patchification
        window_size (Tuple[int, int]): dimension of the window for local attention
        feedforward_dim (int): size of hidden dimension of feedforward network in transformer block
        shift_size (Tuple[int, int]): dimension of shift to be applied to the window. Defaults to (0, 0)
        mlp_dropout (float): dropout probability for mlp in transformer block and projection in SA block.
            Defaults to 0
        attn_dropout (float): dropout probability for attention weights. Defaults to 0.
        drop_path (float): Drop path probability in transformer. Defaults to 0.0.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-5.
    """

    def __init__(self, input_dim: 'int', num_heads: 'int', input_size: 'Tuple[int, int]', window_size: 'Tuple[int, int]', feedforward_dim: 'int', shift_size: 'Tuple[int, int]'=(0, 0), mlp_dropout: 'float'=0.0, attn_dropout: 'float'=0.0, drop_path: 'float'=0.0, layer_norm_eps: 'float'=1e-05) ->None:
        super().__init__()
        self.input_size = input_size
        self.window_size, self.shift_size = self._get_effective_window_shift(window_size, shift_size)
        self.num_heads = num_heads
        self.attn = WindowMultiHeadAttention(input_dim=input_dim, num_heads=num_heads, window_size=self.window_size, attn_dropout=attn_dropout, proj_dropout=mlp_dropout)
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.drop_path1 = StochasticDepth(drop_path, 'row') if drop_path > 0.0 else nn.Identity()
        self.mlp = MLP(in_dim=input_dim, hidden_dims=feedforward_dim, dropout=mlp_dropout, out_dim=input_dim, activation=nn.GELU)
        self.norm2 = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.drop_path2 = StochasticDepth(drop_path, 'row') if drop_path > 0.0 else nn.Identity()
        self._make_attention_mask()

    def _get_effective_window_shift(self, target_window_size: 'Tuple[int, int]', target_shift_size: 'Tuple[int, int]') ->Tuple[Tuple[int, ...], Tuple[Any, ...]]:
        window_size: 'List[int]' = [(f if f <= w else w) for f, w in zip(self.input_size, target_window_size)]
        shift_size = [(0 if f <= w else s) for f, w, s in zip(self.input_size, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def _make_attention_mask(self) ->None:
        if any(self.shift_size):
            input_h, input_w = self.input_size
            img_mask = torch.zeros((1, input_h, input_w, 1))
            cnt = 0
            for h in (slice(0, -self.window_size[0]), slice(-self.window_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None)):
                for w in (slice(0, -self.window_size[1]), slice(-self.window_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self._window_partition(img_mask)
            window_area = self.window_size[0] * self.window_size[1]
            mask_windows = mask_windows.view(-1, window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask, persistent=False)

    def _window_partition(self, x: 'Tensor') ->Tensor:
        bsz, h, w, channels = x.shape
        window_h, window_w = self.window_size
        x = x.view(bsz, h // window_h, window_h, w // window_w, window_w, channels)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_h, window_w, channels)
        return windows

    def _window_reverse(self, windows: 'Tensor', bsz: 'int') ->Tensor:
        input_h, input_w = self.input_size
        window_h, window_w = self.window_size
        x = windows.view(bsz, input_h // window_h, input_w // window_w, window_h, window_w, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(bsz, input_h, input_w, -1)
        return x

    def _shifted_window_attn(self, x: 'Tensor') ->Tensor:
        h, w = self.input_size
        bsz, seq_len, channels = x.shape
        if seq_len != h * w:
            raise ValueError(f'Input sequence length {seq_len} needs to match input size')
        x = x.view(bsz, h, w, channels)
        sh, sw = self.shift_size
        do_shift: 'bool' = any(self.shift_size)
        if do_shift:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
        x_windows = self._window_partition(x)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], channels)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], channels)
        x = self._window_reverse(attn_windows, bsz)
        if do_shift:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
        x = x.view(bsz, seq_len, channels)
        return x

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): input to the transformer block of shape bsz x seq_len x input_dim.
            seq_len shoud match number of patches as per input_size

        Returns:
            Tensor of shape bsz x seq_len x input_dim
        """
        x = x + self.drop_path1(self.norm1(self._shifted_window_attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class SwinTransformer(nn.Module):
    """
    Swin transformer that stacks layers of the swin block

    Args:
        n_layer (int): number of layers
        input_dim (int): input feature dimension
        num_heads (int): number of attention heads
        input_size (Tuple[int, int]): dimension of the original input before patchification
        window_size (Tuple[int, int]): dimension of the window for local attention
        feedforward_dim (int): size of hidden dimension of feedforward network in transformer block
        shift_size (Tuple[int, int]): dimension of shift to be applied to the window. Defaults to (0, 0)
        mlp_dropout (float): dropout probability for mlp in transformer block and projection in SA block.
            Defaults to 0
        attn_dropout (float): dropout probability for attention weights. Defaults to 0.
        drop_path (float): Drop path probability in transformer. Defaults to 0.0.
        layer_norm_eps (float): the eps value in layer norms. Default is 1e-5.
        final_layer_norm_eps (float): the eps value in layer norms. Default is 1e-5
    """

    def __init__(self, n_layer: 'int', input_dim: 'int', num_heads: 'int', input_size: 'Tuple[int, int]', window_size: 'Tuple[int, int]', feedforward_dim: 'int', mlp_dropout: 'float'=0.0, attn_dropout: 'float'=0.0, drop_path: 'float'=0.0, layer_norm_eps: 'float'=1e-05, final_layer_norm_eps: 'float'=1e-05):
        super().__init__()
        layers = []
        for idx in range(n_layer):
            if idx % 2 == 0:
                shift_size = 0, 0
            else:
                shift_size = 2, 0
            layers.append(SwinTransformerBlock(input_dim=input_dim, num_heads=num_heads, input_size=input_size, window_size=window_size, shift_size=shift_size, feedforward_dim=feedforward_dim, mlp_dropout=mlp_dropout, attn_dropout=attn_dropout, drop_path=drop_path, layer_norm_eps=layer_norm_eps))
        self.layers = nn.ModuleList(layers)
        self.final_layer_norm = nn.LayerNorm(input_dim, eps=final_layer_norm_eps)

    def forward(self, x: 'Tensor') ->TransformerOutput:
        """
        Args:
            x (Tensor): input to the transformer block of shape bsz x seq_len x input_dim.
            seq_len shoud match number of patches as per input_size

        Returns:
            Output of type TransformerOutput with last_hidden_state field contain tensor of shape bsz x seq_len x input_dim
            representing output from final layer
        """
        hidden_states = x
        for layer_module in self.layers:
            layer_outputs = layer_module(hidden_states)
            hidden_states = layer_outputs
        hidden_states = self.final_layer_norm(hidden_states)
        return TransformerOutput(last_hidden_state=hidden_states)


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copied from torchvision.ops.misc with added eps before rsqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans. This module is a useful replacement for BatchNorm2d in the
    case of very small batches, see https://bit.ly/3xQvmiJ.


    Args:   n (int): Number of features ``C`` from expected input size ``(N, C, H, W)``
            eps (float): Value added to denominator for numerical stability.
                Default = 1e-5

    Inputs: x (Tensor): Tensor to be normalized
    """

    def __init__(self, n: 'int', eps: 'float'=1e-05):
        super().__init__()
        self.eps = eps
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def forward(self, x: 'Tensor') ->Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class PositionEmbedding2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper (https://arxiv.org/abs/1706.03762),
    generalized to work on images.

    Args:   num_pos_feats (int): Number of positional features
                (should be half the output embedding size). Default = 64
            temperature (int): Base for generating frequency mesh. Default = 10000
            scale (float): Scaling factor when performing normalization. Setting
                scale = s will rescale values to fall in [0, s].
                Default = None (no normalization)

    Inputs: mask (Tensor): Padding mask (used to infer size of each image in batch).
                Input size: (batch_size, height, width)

    Returns: Tensor of size (batch_size, 2 * num_pos_feats, height, width)
    """

    def __init__(self, num_pos_feats: 'int'=64, temperature: 'int'=10000, scale: 'float'=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale

    def forward(self, mask: 'Tensor') ->Tensor:
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.scale is not None:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class MaskedIntermediateLayer(nn.Module):
    """
    This class wraps a backbone returning an intermediate layer (e.g. a ResNet
    where we do not want to perform pooling) while casting masks to the appropriate
    sizes.

    Note: for simplicity we only support returning a single intermediate layer.

    Args:   body (nn.Module): The module to return the intermediate layer from.
            intermediate_layer (str): Name of the layer to return from body.

    Inputs: images (Tensor): Batch of images to pass to the backbone.
            image_masks (Tensor): Masks to cast to backbone output size.
    """

    def __init__(self, body: 'nn.Module', intermediate_layer: 'str'):
        super().__init__()
        self.body = IntermediateLayerGetter(body, return_layers={intermediate_layer: 0})

    def forward(self, images: 'torch.Tensor', image_masks: 'torch.Tensor') ->Tuple[Tensor, Tensor]:
        out = self.body(images)
        tensor = out[next(iter(out))]
        mask = F.interpolate(image_masks[None].float(), size=tensor.shape[-2:]).bool()[0]
        return tensor, mask


class MDETRModelOutput(NamedTuple):
    transformer_output: 'MDETRTransformerOutput'
    pred_logits: 'torch.Tensor'
    pred_boxes: 'torch.Tensor'
    extra_embeddings: 'Optional[torch.Tensor]'


class MDETR(nn.Module):
    """
    MDETR (https://arxiv.org/abs/2104.12763) is a modulated detection model
    used to detect objects in an image conditioned on text or captions.
    This class contains the entire MDETR architecture, including the
    image backbone, text encoder, and multimodal transformer. (Note that the
    matcher and losses are provided elsewhere.)

    Args:   image_backbone (nn.Module): Torch module of the backbone to be used.
                See image_encoder.py.
            text_encoder (nn.Module): Torch module of the text encoder to be used.
                See text_encoder.py.
            transformer (nn.Module): The multimodal transformer module. See the
                Transformer class in this file.
            pos_embed (nn.Module): Module for positional embedding of images.
            text_projection (nn.Module): Module to resize text encoder outputs before feeding
                them to the multimodal transformer.
            image_projection (nn.Module): Projection module applied to image embeddings
                prior to the multimodal transformer.
            query_embed (nn.Module): Learned object query embeddings (used in
                transformer decoder).
            bbox_embed (nn.Module): Embedding mapping transformer outputs to
                bounding boxes.
            class_embed (nn.Module): Embedding mapping transformer outputs to classes.
            extra_query_embeddings (Optional[nn.Embedding]): Additional query embeddings,
                as used in e.g. VQA. Default: None

    Inputs: images (List[Tensor]): A list of image Tensors (possibly of different sizes).
            text (List[Tensor]): A list of Tensors of tokenized texts (possibly of different lengths).

    Returns:
        A dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
    """

    def __init__(self, image_backbone: 'nn.Module', text_encoder: 'nn.Module', transformer: 'nn.Module', pos_embed: 'nn.Module', text_projection: 'nn.Module', image_projection: 'nn.Module', query_embed: 'nn.Embedding', bbox_embed: 'nn.Module', class_embed: 'nn.Module', extra_query_embeddings: 'Optional[nn.Embedding]'=None):
        super().__init__()
        self.image_backbone = image_backbone
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.transformer = transformer
        self.pos_embed = pos_embed
        self.image_projection = image_projection
        self.query_embed = query_embed
        self.bbox_embed = bbox_embed
        self.class_embed = class_embed
        self.extra_query_embeddings = extra_query_embeddings

    def _pad_images(self, images: 'List[Tensor]') ->Tuple[Tensor, Tensor]:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
        batch_shape = (len(images),) + max_size
        b, _, h, w = batch_shape
        dtype = images[0].dtype
        device = images[0].device
        padded_images = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(images, padded_images, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
        return padded_images, mask

    def _pad_text(self, text: 'List[Tensor]', padding_idx: 'int'=1) ->Tuple[Tensor, Tensor]:
        padded_text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=padding_idx)
        mask = padded_text == padding_idx
        return padded_text, mask

    def forward(self, images: 'List[Tensor]', text: 'List[Tensor]') ->MDETRModelOutput:
        images, image_mask = self._pad_images(images)
        text, text_attention_mask = self._pad_text(text)
        encoded_text = self.text_encoder(text, text_attention_mask)
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        image_embeddings, image_mask = self.image_backbone(images, image_mask)
        pos = self.pos_embed(image_mask)
        query_embed = self.query_embed.weight
        if self.extra_query_embeddings is not None:
            n_extra_embeddings = self.extra_query_embeddings.num_embeddings
            query_embed = torch.cat([query_embed, self.extra_query_embeddings.weight])
        text_memory_resized = self.text_projection(text_memory)
        transformer_output = self.transformer(self.image_projection(image_embeddings), image_mask, query_embed, pos, text_memory=text_memory_resized, text_attention_mask=text_attention_mask)
        if self.extra_query_embeddings is not None:
            extra_embeddings = transformer_output.decoder_hidden_states[0, :, -n_extra_embeddings:]
            decoder_hidden_states_truncated = transformer_output.decoder_hidden_states[:, :, :-n_extra_embeddings]
            transformer_output = transformer_output._replace(decoder_hidden_states=decoder_hidden_states_truncated)
        else:
            extra_embeddings = None
        final_hidden_state = transformer_output.decoder_hidden_states[-1]
        outputs_class = self.class_embed(final_hidden_state)
        outputs_coord = self.bbox_embed(final_hidden_state).sigmoid()
        return MDETRModelOutput(transformer_output, outputs_class, outputs_coord, extra_embeddings)


class ContrastiveEmbeddingsOutput(NamedTuple):
    query_embeddings: 'Tensor'
    token_embeddings: 'Tensor'


class MDETRVQAOutput(NamedTuple):
    model_output: 'MDETRModelOutput'
    vqa_preds: 'Dict[str, Tensor]'
    contrastive_embeddings: 'ContrastiveEmbeddingsOutput'


class MDETRForVQA(nn.Module):

    def __init__(self, model: 'MDETR', vqa_heads: 'nn.ModuleDict', contrastive_alignment_image_projection: 'nn.Module', contrastive_alignment_text_projection: 'nn.Module'):
        super().__init__()
        self.model = model
        self.vqa_heads = vqa_heads
        if self.model.extra_query_embeddings is None:
            raise ValueError('MDETRForVQA requires extra query embeddings ')
        if self.model.extra_query_embeddings.num_embeddings != len(self.vqa_heads.keys()):
            raise ValueError('Number of heads must match number of QA embeddings')
        self.contrastive_alignment_image_projection = contrastive_alignment_image_projection
        self.contrastive_alignment_text_projection = contrastive_alignment_text_projection

    def forward(self, images: 'List[Tensor]', text: 'List[Tensor]') ->MDETRVQAOutput:
        model_output = self.model(images, text)
        final_hidden_state = model_output.transformer_output.decoder_hidden_states[-1]
        contrastive_query_embeddings = F.normalize(self.contrastive_alignment_image_projection(final_hidden_state), p=2, dim=-1)
        contrastive_token_embeddings = F.normalize(self.contrastive_alignment_text_projection(model_output.transformer_output.text_memory).transpose(0, 1), p=2, dim=-1)
        contrastive_outputs = ContrastiveEmbeddingsOutput(contrastive_query_embeddings, contrastive_token_embeddings)
        answer_preds = OrderedDict()
        vqa_embeddings = model_output.extra_embeddings.transpose(0, 1)
        for (head_name, head), embedding in zip(self.vqa_heads.items(), vqa_embeddings):
            answer_preds[head_name] = head(embedding)
        return MDETRVQAOutput(model_output, answer_preds, contrastive_outputs)


class MDETRPhraseGroundingOutput(NamedTuple):
    model_output: 'MDETRModelOutput'
    contrastive_embeddings: 'ContrastiveEmbeddingsOutput'


class MDETRForPhraseGrounding(nn.Module):

    def __init__(self, model: 'MDETR', contrastive_alignment_image_projection: 'nn.Module', contrastive_alignment_text_projection: 'nn.Module'):
        super().__init__()
        self.model = model
        self.contrastive_alignment_image_projection = contrastive_alignment_image_projection
        self.contrastive_alignment_text_projection = contrastive_alignment_text_projection

    def forward(self, images: 'List[Tensor]', text: 'List[Tensor]') ->MDETRPhraseGroundingOutput:
        model_output = self.model(images, text)
        final_hidden_state = model_output.transformer_output.decoder_hidden_states[-1]
        contrastive_query_embeddings = F.normalize(self.contrastive_alignment_image_projection(final_hidden_state), p=2, dim=-1)
        contrastive_token_embeddings = F.normalize(self.contrastive_alignment_text_projection(model_output.transformer_output.text_memory).transpose(0, 1), p=2, dim=-1)
        contrastive_outputs = ContrastiveEmbeddingsOutput(contrastive_query_embeddings, contrastive_token_embeddings)
        return MDETRPhraseGroundingOutput(model_output, contrastive_outputs)


class ModifiedTransformerEncoder(nn.Module):
    """
    Modified version of TorchText's RoBERTa transformer encoder
    taking in embeddings instead of input IDs.

    Args:   embedding_dim (int): Number of features in the input.
            num_encoder_layers  (int): Number of layers in the encoder.
            num_attention_heads (int): Number of heads in multi-head attention.
            ffn_dimension (int): Dimension of feedforward network inside
                attention layers.
            dropout (float): dropout value in each layer. Default: 0.1.
            normalize_before (bool): Whether to do PreNorm in encoder layers.
                Default: False
            return_all_layers (bool) Whether to return all layers (or just the last
                one). Default: False

    Inputs: embeddings (Tensor): Tensor of embeddings of a batch of input IDs.
            attention_mask (Optional[Tensor]) Batch attention mask returned from
                tokenizer (applied as padding mask inside self-attention).
                Default: None
    """

    def __init__(self, embedding_dim: 'int', num_encoder_layers: 'int', num_attention_heads: 'int', ffn_dimension: 'int', dropout: 'float'=0.1, normalize_before: 'bool'=False):
        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_attention_heads, dim_feedforward=ffn_dimension, dropout=dropout, activation='gelu', batch_first=True, norm_first=normalize_before)
        self.layers = torch.nn.TransformerEncoder(encoder_layer=layer, num_layers=num_encoder_layers)
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: 'Tensor', attention_mask: 'Optional[Tensor]'=None, return_attn_weights: 'bool'=False, return_hidden_states: 'bool'=False) ->TransformerOutput:
        encoded = embeddings
        batch_size, seq_len = embeddings.size()[:2]
        mask = attention_mask.reshape(batch_size, seq_len)
        for layer in self.layers.layers:
            encoded = layer(encoded, src_key_padding_mask=mask)
        return TransformerOutput(last_hidden_state=encoded)


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    Args:   input_feat_size (int): Dimension of input features.
            output_feat_size (int): Dimension of output features.
            dropout (float): Dropout probability for final features. Default: 0.1
            do_ln (bool): Whether to perform layer normalization after the linear layer.
    Inputs: encoder_features (Tensor): Features to be resized.
    """

    def __init__(self, input_feat_size: 'int', output_feat_size: 'int', dropout: 'float'=0.1, do_ln: 'bool'=True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12) if do_ln else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features: 'Tensor') ->Tensor:
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class MDETRTransformerOutput(NamedTuple):
    decoder_hidden_states: 'torch.Tensor'
    text_memory: 'torch.Tensor'


class MDETRTransformer(nn.Module):
    """
    Transformer class for MDETR model.

    Args:   d_model (int): Number of features in the input.
            num_heads (int): Number of heads in multi-head attention.
            num_encoder_layers (int): Number of layers in the encoder. Default: 6
            num_decoder_layers (int): Number of layers in the decoder. Default: 6
            dim_feedforward (int): Dimension of feedforward network. Default: 2048
            dropout (float): Dropout value. Default: 0.1.
            activation (Callable[..., nn.Module]): The activation function of the
                intermediate layer. Default: nn.ReLU
            normalize_before (bool): Whether to do PreNorm. Default: False
            return_intermediate_dec (bool): Whether to return intermediate decoder outputs.
                Default: True

    Inputs: image_embeddings Tensor: The image input.
            image_mask (Tensor) The mask for the image sequence.
            query_embed (Tensor): Positional embeddings applied to Q
                cross-attention matrix in decoder.
            pos_embed (Tensor): Positional embeddings applied to Q and K
                self-attention matrices in decoder.
            text_memory (Tensor): Text input.
            text_attention_mask (Tensor): Attention mask for text input.
    """

    def __init__(self, d_model: 'int'=512, num_heads: 'int'=8, num_encoder_layers: 'int'=6, num_decoder_layers: 'int'=6, dim_feedforward: 'int'=2048, dropout: 'float'=0.1, activation: 'Callable[..., nn.Module]'=nn.ReLU, normalize_before: 'bool'=False, return_intermediate_dec: 'bool'=True):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout, activation, normalize_before)
        encoder_final_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_final_norm)
        decoder_layer = TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self.d_model = d_model
        self._init_parameters()

    def _init_parameters(self) ->None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image_embeddings: 'Tensor', image_mask: 'Tensor', query_embed: 'Tensor', pos_embed: 'Tensor', text_memory: 'Tensor', text_attention_mask: 'Tensor') ->MDETRTransformerOutput:
        bs = image_embeddings.size(0)
        image_embeddings = image_embeddings.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        image_mask = image_mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        mm_embeddings = torch.cat([image_embeddings, text_memory], dim=0)
        image_mask = torch.cat([image_mask, text_attention_mask], dim=1)
        pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory)], dim=0)
        mm_memory = self.encoder(mm_embeddings, src_key_padding_mask=image_mask, pos=pos_embed)
        text_memory = mm_memory[-len(text_memory):]
        assert mm_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
        assert mm_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
        hs = self.decoder(tgt, mm_memory, memory_key_padding_mask=image_mask, pos=pos_embed, query_pos=query_embed)
        return MDETRTransformerOutput(decoder_hidden_states=hs.transpose(1, 2), text_memory=text_memory)


class Omnivore(nn.Module):
    """Omnivore is a model that accept multiple vision modality.

    Omnivore (https://arxiv.org/abs/2201.08377) is a single model that able to do classification
    on images, videos, and single-view 3D data using the same shared parameters of the encoder.

    Args:
        encoder (nn.Module): Instantiated encoder. It generally accept a video backbone.
            The paper use SwinTransformer3d for the encoder.
        heads (Optional[nn.ModuleDict]): Dictionary of multiple heads for each dataset type

    Inputs:
        x (Tensor): 5 Dimensional batched video tensor with format of B C D H W
            where B is batch, C is channel, D is time, H is height, and W is width.
        input_type (str): The dataset type of the input, this will used to choose
            the correct head.
    """

    def __init__(self, encoder: 'nn.Module', heads: 'nn.ModuleDict'):
        super().__init__()
        self.encoder = encoder
        self.heads = heads

    def forward(self, x: 'torch.Tensor', input_type: 'str') ->torch.Tensor:
        x = self.encoder(x)
        assert input_type in self.heads, f'Unsupported input_type: {input_type}, please use one of {list(self.heads.keys())}'
        x = self.heads[input_type](x)
        return x


class PatchEmbedOmnivore(nn.Module):
    """Patch Embedding strategy for Omnivore model
    It will use common PatchEmbed3d for image and video,
    for single view depth image it will have separate embedding for the depth channel
    and add the embedding result with the RGB channel
    reference: https://arxiv.org/abs/2201.08377

    Args:
        patch_size (Tuple[int, int, int]): Patch token size. Default: ``(2, 4, 4)``
        embed_dim (int): Number of linear projection output channels. Default: ``96``
        norm_layer (nn.Module, optional): Normalization layer. Default: ``None``
    """

    def __init__(self, patch_size: 'List[int]', embed_dim: 'int'=96, norm_layer: 'Optional[Callable[..., nn.Module]]'=None):
        super().__init__()
        self.patch_embed = PatchEmbed3d(patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer)
        self.depth_patch_embed = PatchEmbed3d(patch_size=patch_size, in_channels=1, embed_dim=embed_dim, norm_layer=norm_layer)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        assert x.ndim == 5
        has_depth = x.shape[1] == 4
        if has_depth:
            x_rgb = self.patch_embed(x[:, :3, ...])
            x_d = self.depth_patch_embed(x[:, 3:, ...])
            x = x_rgb + x_d
        else:
            x = self.patch_embed(x)
        return x


class TwoTowerOutput(NamedTuple):
    output: 'Tensor'
    tower_embeddings: 'Dict[str, Tensor]'


class TwoTower(nn.Module):
    """
    A two tower architecture with a pair of late fusion models
    (for now, can be extended) followed by a fusion for output of each tower.
    Args:
        tower_id_to_tower (Dict[str, LateFusion]): mapping of tower id
        to tower model. Size should be 2, same tower should be passed in
        for shared towers
        tower fusion (nn.Module): Module fusing list of tensors (tower outputs)
        into single output
        shared_tower_id_to_channel_mapping (Optional[Dict[str, Dict[str, str]]]): Dict
        of shared tower id to mapping of channel names of the shared tower
         to the original input channel name
    Inputs:
        channel_to_input (Dict[str,Tensor]) : Channel name to input tensor dict
    """

    def __init__(self, tower_id_to_tower: 'Dict[str, LateFusion]', tower_fusion: 'nn.Module', shared_tower_id_to_channel_mapping: 'Optional[Dict[str, Dict[str, str]]]'=None):
        super().__init__()
        if len(tower_id_to_tower) != 2:
            raise ValueError(f'Two tower needs 2 towers but found                 {len(tower_id_to_tower)} towers')
        self.tower_id_to_tower = nn.ModuleDict(tower_id_to_tower)
        self.tower_fusion = tower_fusion
        if shared_tower_id_to_channel_mapping is not None:
            towers = list(tower_id_to_tower.values())
            if towers[0] != towers[1]:
                raise ValueError('Towers should be shared if channel mapping is passed in')
        self.shared_tower_id_to_channel_mapping: 'Optional[Dict[str, Dict[str, str]]]' = shared_tower_id_to_channel_mapping

    def forward(self, channel_to_input: 'Dict[str, Tensor]') ->TwoTowerOutput:
        tower_embeddings = OrderedDict()
        for tower_id, tower in self.tower_id_to_tower.items():
            tower_input = self._get_tower_input(tower_id, list(tower.encoders.keys()), channel_to_input)
            tower_embeddings[tower_id] = tower(tower_input)
        final_out = self.tower_fusion(list(tower_embeddings.values()))
        return TwoTowerOutput(output=final_out, tower_embeddings=tower_embeddings)

    def _get_tower_input(self, tower_id: 'str', tower_channels: 'List[str]', channel_to_input: 'Dict[str, Tensor]') ->Dict[str, Tensor]:
        tower_input = {}
        channel_name_mapping: 'Dict[str, str]' = {}
        if self.shared_tower_id_to_channel_mapping is not None:
            if self.shared_tower_id_to_channel_mapping.get(tower_id) is not None:
                channel_name_mapping = self.shared_tower_id_to_channel_mapping[tower_id]
        for channel in tower_channels:
            if channel_name_mapping.get(channel) is not None:
                input_channel_name = channel_name_mapping[channel]
            else:
                input_channel_name = channel
            tower_input[channel] = channel_to_input[input_channel_name]
        return tower_input


class MultimodalGPTOutput(NamedTuple):
    """Outputs from :meth:`~torchmultimodal.models.video_gpt.gpt.MultimodalGPT.forward`.

    Attributes:
        decoder_output (TransformerDeocoderOutput): Contains output from the multimodal transformer decoder.
            See :class:`MultimodalTransformerDecoder`.
        logits (Tensor): Logits computed from the last hidden state of the multimodal transformer decoder.
    """
    decoder_output: 'TransformerDecoderOutput'
    logits: 'Tensor'


class TransformerDecoderOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.models.video_gpt.gpt.TransformerDecoder`.

    Attributes:
        last_hidden_states (Tensor): Output from the last layer of the transformer.
        hidden_states (Tuple[Tensor, ...], optional): Outputs from all layers of the transformer.
            Defaults to ``None``.
        attention_weights (Tuple[Tensor, ...], optional): Attention probabilities from all layers of the
            transformer. Defaults to ``None``.
        past_key_values (Tuple[Dict[str, Tensor], ...]], optional): If ``use_cache`` is on, contains
            key/value tensors prior to the current step along the sequence. Defaults to ``None``.
    """
    last_hidden_states: 'Tensor'
    hidden_states: 'Optional[Tuple[Tensor, ...]]' = None
    attention_weights: 'Optional[Tuple[Tensor, ...]]' = None
    past_key_values: 'Optional[Tuple[Dict[str, Tensor], ...]]' = None


class MultimodalGPT(nn.Module):
    """Extends the GPT (Generative Pre-Training) model for cross-modality generation.

    This module implements the GPT model for generation of one modality given another
    following the paper `"Improving Language Understanding by Generative Pre-Training
    "<https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>`_.

    Args:
        d_model (int): Embedding dimension of the transformer decoder.
        num_in_tokens (int): Number of unique token states for the input modality.
        num_out_tokens (int): Number of unique token states for the output modality.
        latent_shape ([Tuple[int, ...]): Shape of the latent space of the output modality tokenizer. Used to reshape
            sequence of generated tokens to be decoded back to data.
        in_tokenizer (nn.Module): Tokenizer for the input modality. Must have methods ``encode``, ``lookup``.
        out_tokenizer (nn.Module): Tokenizer for the output modality. Must have methods ``encode``, ``decode``.
        mm_decoder (nn.Module): Multimodal transformer decoder. An instace of
            :py:class:`MultimodalTransformerDecoder`.
        in_projection (nn.Module, optional): Projects the input modality token embeddings to match size of the
            transformer decoder. Defaults to ``None``.
        out_projection (nn.Module, optional): Projects the output modality token embeddings to match size of the
            transformer decoder. Defaults to ``None``.
        norm_layer (Callable[..., nn.Module], optional): Which normalization layer to use. Supports ``nn.Module`` or
            partial. If ``None``, ``nn.LayerNorm`` will be used as the default.
        use_gpt_init (bool): Whether to use GPT model specific initialization. Defaults to ``True``.

    Raises:
        AttributeError: If input tokenizer does not implement methods ``encode`` and ``lookup`` or if output
        tokenizer does not implement methods ``encode``, ``lookup`` and ``decode``.
    """

    def __init__(self, d_model: 'int', num_in_tokens: 'int', num_out_tokens: 'int', latent_shape: 'Tuple[int, ...]', in_tokenizer: 'nn.Module', out_tokenizer: 'nn.Module', mm_decoder: 'nn.Module', in_projection: 'Optional[nn.Module]'=None, out_projection: 'Optional[nn.Module]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, use_gpt_init: 'bool'=True) ->None:
        super().__init__()
        if not all([hasattr(in_tokenizer, attr_name) for attr_name in ['encode', 'lookup']]):
            raise AttributeError("Input modality tokenizer must have methods 'encode' and 'lookup'.")
        if not all([hasattr(out_tokenizer, attr_name) for attr_name in ['encode', 'lookup', 'decode']]):
            raise AttributeError("Output modality tokenizer must have methods 'encode', 'lookup' and 'decode'.")
        num_tokens = num_in_tokens + num_out_tokens
        self.num_in_tokens = num_in_tokens
        self.num_out_tokens = num_out_tokens
        self.latent_shape = latent_shape
        self.in_tokenizer = in_tokenizer
        self.out_tokenizer = out_tokenizer
        self.mm_decoder = mm_decoder
        self.in_projection = in_projection
        self.out_projection = out_projection
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-05)
        self.norm = norm_layer(normalized_shape=d_model)
        self.to_logit = nn.Linear(d_model, num_tokens, bias=False)
        self.to_logit.weight.data.copy_(torch.zeros(num_tokens, d_model))
        if use_gpt_init:
            self.initialize_parameters()

    def initialize_parameters(self) ->None:
        if hasattr(self.in_projection, 'weight'):
            self.in_projection.weight.data.normal_(std=0.02)
        if hasattr(self.out_projection, 'weight'):
            self.out_projection.weight.data.normal_(std=0.02)

    def forward(self, in_tokens: 'Optional[Tensor]'=None, out_tokens: 'Optional[Tensor]'=None, in_pos_ids: 'Optional[Tensor]'=None, out_pos_ids: 'Optional[Tensor]'=None, attn_mask: 'Optional[Tensor]'=None, head_mask: 'Optional[Tensor]'=None, logits_mask: 'Optional[Tensor]'=None, use_cache: 'bool'=False, causal: 'bool'=False, right_shift: 'bool'=False, return_attn_weights: 'bool'=False, return_hidden_states: 'bool'=False) ->MultimodalGPTOutput:
        """
        Args:
            in_tokens (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing tokens
                for the input modality. Defaults to ``None``.
            out_tokens (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing tokens
                for the output modality. Defaults to ``None``.
            in_pos_ids (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing indices for the
                input modality position embeddings. Defaults to ``None``.
            out_pos_ids (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing indices for the
                output modality position embeddings. Defaults to ``None``.
            attn_mask (Tensor, optional): Tensor of dimension ``(q_seq_len, k_seq_len)`` or
                ``(b, q_seq_len, k_seq_len)`` where prefixes ``q`` and ``k`` stand for query and key.
                Contains 1s for positions to attend to and 0s for masked positions. Defaults to ``None``.
            head_mask (Tensor, optional): Tensor of dimension ``(h, q_seq_len, k_seq_len)`` or
                ``(b, h, q_seq_len, k_seq_len)``. Masks need to be specified for each attention head.
                Defaults to ``None``.
            logits_mask (Tensor, optional): Tensor of dimension ``(seq_len, num_tokens)`` or
                ``(b, seq_len, num_tokens)`` to ensure we only calculate probabilities from tokens of the
                corresponding modality sequence.
            use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding. If ``False``,
                recomputes key and value for each decoding step. Defaults to ``False``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            right_shift (bool): If ``True``, shifts the embedding vectors to the right and prepends it with start of
                sentence token. Defaults to ``False``. This option is disregarded during training mode
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
                layer. Defaults to ``False``.
            return_hidden_states (bool, optional): If ``True``, returns the embeddings of each transformer layer.
                Defaults to ``False``.

        Returns:
            An instance of :class:`~torchmultimodal.models.video_gpt.gpt.MultimodalGPTOutput`.
        """
        decoder_output = self.fwd(in_tokens=in_tokens, out_tokens=out_tokens, in_pos_ids=in_pos_ids, out_pos_ids=out_pos_ids, attn_mask=attn_mask, head_mask=head_mask, use_cache=use_cache, causal=causal, right_shift=right_shift, return_attn_weights=return_attn_weights, return_hidden_states=return_hidden_states)
        hidden_states = decoder_output.last_hidden_states
        logits = self.logit_projection(hidden_states, logits_mask)
        return MultimodalGPTOutput(decoder_output, logits)

    def fwd(self, in_tokens: 'Optional[Tensor]'=None, out_tokens: 'Optional[Tensor]'=None, in_pos_ids: 'Optional[Tensor]'=None, out_pos_ids: 'Optional[Tensor]'=None, attn_mask: 'Optional[Tensor]'=None, head_mask: 'Optional[Tensor]'=None, use_cache: 'bool'=False, causal: 'bool'=False, right_shift: 'bool'=False, return_attn_weights: 'bool'=False, return_hidden_states: 'bool'=False) ->TransformerDecoderOutput:
        if in_tokens is None and out_tokens is None:
            raise ValueError('input-modality token and output-modality token sequences cannot be both empty')
        in_modality = out_modality = None
        if in_tokens is not None:
            in_modality = self.lookup(in_tokens, 'in')
            if self.in_projection is not None:
                in_modality = self.in_projection(in_modality)
        if out_tokens is not None:
            out_modality = self.lookup(out_tokens, 'out')
            if self.out_projection is not None:
                out_modality = self.out_projection(out_modality)
        return self.mm_decoder(in_modality=in_modality, out_modality=out_modality, in_pos_ids=in_pos_ids, out_pos_ids=out_pos_ids, attn_mask=attn_mask, head_mask=head_mask, use_cache=use_cache, causal=causal, right_shift=right_shift, return_attn_weights=return_attn_weights, return_hidden_states=return_hidden_states)

    def logit_projection(self, hidden_states: 'Tensor', logits_mask: 'Optional[Tensor]'=None) ->Tensor:
        if logits_mask is not None and logits_mask.dim() == 2:
            logits_mask = logits_mask.unsqueeze(0)
        hidden_states = self.norm(hidden_states)
        logits = self.to_logit(hidden_states)
        max_neg_value = -torch.finfo(logits.dtype).max
        if logits_mask is not None:
            logits.masked_fill_(logits_mask == 0, max_neg_value)
        return logits

    def encode(self, x: 'Any', modality: 'str', **kwargs: Any) ->Tensor:
        """Converts data to token ids.

        Although this is not part of the forward pass, it is used to generate labels for training
        as well as inputs for autoregressive decoding.

        Args:
            x (Any): Data to be encoded, e.g., ``List[str]`` for text, ``Tensor`` of shape
                ``(b, c, d1, ..., dn)`` for audio/image/video.
            modality (str): Input or output modality string used to select the encoder.
            kwargs (Any): Other keyword arguments suitable for the encoder.

        Returns:
            A tensor of token ids of shape ``(b, seq_len)``.

        Raises:
            ValueError: If ``modality`` is neither ``in`` nor ``out``.
        """
        if modality == 'in':
            encoder = self.in_tokenizer.encode
        elif modality == 'out':
            encoder = self.out_tokenizer.encode
        else:
            raise ValueError(f'Invalid modality parameter: {modality}')
        token_ids = encoder(x, **kwargs)
        return token_ids.flatten(start_dim=1, end_dim=-1)

    def decode(self, token_ids: 'Tensor', **kwargs: Any) ->Any:
        """Converts out-modality tokens ids back to data during generation.

        Args:
            token_ids (Tensor): Token ID sequence ``(b, seq_len)`` to be decoded.
            kwargs (Any): Other keywords arguments suitable for the decoder.

        Returns:
            The decoded data, e.g., ``List[str]`` for text, a tensor of shape ``(b, c, d1. ,,, dn)`` for
                audio/image/video.

        Raises:
            ValueError: If the shape of ``token_ids`` is not of dimension two.
            ValueError: If the sequence dim of ``token_ids`` does not match that inferred from ``latent_shape``.
        """
        if len(token_ids.shape) != 2:
            raise ValueError(f"Shape of token ids should be '(batch_size, sequence_length)' but got {token_ids.shape}")
        latent_seq_len = torch.prod(torch.tensor(self.latent_shape)).item()
        if token_ids.shape[1] != latent_seq_len:
            raise ValueError(f'Sequence to decode does not match that inferred from the tokenizer: {latent_seq_len}')
        token_ids = token_ids.view(token_ids.shape[0], *self.latent_shape)
        return self.out_tokenizer.decode(token_ids, **kwargs)

    def lookup(self, token_ids: 'Tensor', modality: 'str') ->Tensor:
        """Looks up the latent embeddings corresponding to the token ids during generation.

        We ask each tokenizer to implement this method. An example is :class:`torchmultimodal.models.vqvae.VQVAE`.

        Args:
            token_ids (Tensor): Token ID sequence ``(b, seq_len)``.
            modality (str): The modality at which this method is performed.

        Returns:
            A tensor of embeddings corresponding to the token ids.

        Raises:
            ValueError: If ``modality`` is neither ``in`` nor ``out``.
        """
        if modality == 'in':
            tokenizer = self.in_tokenizer
        elif modality == 'out':
            tokenizer = self.out_tokenizer
        else:
            raise ValueError(f'Invalid modality parameter: {modality}')
        return tokenizer.lookup(token_ids)


class MultimodalTransformerDecoder(nn.Module):
    """A transformer decoder for two modalities

    The token- and position- embedding layers are per modality:
        * During training both modalities are fed into the module and concatenated as a single sequence of
            tokenized embedding vectors
        * During generation the future data points are predicted step-wise from the past. The input modality
            is processed before the output modality (see ``torchmultimodal.utils.common.generate``). Therefore,
            at any point in time the input data contains only one modality.

    Args:
        in_pos_emb (nn.Module): Input modality position embedding layer.
        out_pos_emb (nn.Module): Output modality position embedding layer.
        decoder (nn.Module): The transformer decoder. An instance of :py:class:`TransformerDecoder`.
        right_shift (nn.Module): Layer that shifts the embedding vectors to the right and prepends it with
            start of sentence token (SOS). An instance of :py:class:`RightShift`.

    Note:
        * During training mode, the SOS token is prepended to the left of the concatenated input and
            output modality sequence;
        * During generation mode, the SOS token is only required for the input modality sequence as
            the initial token to be learnt from. Right shift should be turned off
            (``right_shift = False``, see args) when we start to generate the output modality samples.
    """

    def __init__(self, in_pos_emb: 'nn.Module', out_pos_emb: 'nn.Module', decoder: 'nn.Module', right_shift: 'nn.Module') ->None:
        super().__init__()
        self.in_pos_emb = in_pos_emb
        self.out_pos_emb = out_pos_emb
        self.decoder = decoder
        self.right_shift = right_shift

    def forward(self, in_modality: 'Optional[Tensor]'=None, out_modality: 'Optional[Tensor]'=None, in_pos_ids: 'Optional[Tensor]'=None, out_pos_ids: 'Optional[Tensor]'=None, attn_mask: 'Optional[Tensor]'=None, head_mask: 'Optional[Tensor]'=None, use_cache: 'bool'=False, causal: 'bool'=False, right_shift: 'bool'=False, return_attn_weights: 'bool'=False, return_hidden_states: 'bool'=False) ->TransformerDecoderOutput:
        """
        Args:
            in_modality (Tensor, optional): Tensor of dimension ``(b, in_seq_len, d_model)`` containing tokenized
                embeddings for the input modality. Defaults to ``None``.
            out_modality (Tensor, optional): Tensor of dimension ``(b, out_seq_len, d_model')`` containing tokenized
                embeddings for the output modality. Defaults to ``None``.
            in_pos_ids (Tensor, optional): Tensor of dimension ``(b, in_seq_len)`` containing indices for the
                input modality position embeddings. Defaults to ``None``.
            out_pos_ids (Tensor, optional): Tensor of dimension ``(b, out_seq_len)`` containing indices for the
                output modality position embeddings. Defaults to ``None``.
            attn_mask (Tensor, optional): Tensor of dimension ``(q_seq_len, k_seq_len)`` or
                ``(b, q_seq_len, k_seq_len)`` where prefixes ``q`` and ``k`` stand for query and key.
                Contains 1s for positions to attend to and 0s for masked positions. Defaults to ``None``.
            head_mask (Tensor, optional): Tensor of dimension ``(h, q_seq_len, k_seq_len)`` or
                ``(b, h, q_seq_len, k_seq_len)``. Masks need to be specified for each attention head.
                Defaults to ``None``.
            use_cache (bool, optional): If ``True``, caches past key/value tensors for faster decoding.
                If ``False``, recomputes key and value for each decoding step. Defaults to ``False``.
            causal (bool, optional): If ``True``, use causal attention. Defaults to ``False``.
            right_shift (bool): If ``True``, shifts the embedding vectors to the right and prepends it with start of
                sentence token. Defaults to ``False``. This option is disregarded during training mode
            return_attn_weights (bool, optional): If ``True``, returns attention probabilities of each transformer
                layer. Defaults to ``False``.
            return_hidden_states (bool, optional): If ``True``, returns the embeddings of each transformer layer.
                Defaults to ``False``.

        Returns:
            An instace of :class:`~torchmultimodal.models.video_gpt.gpt.TransformerDecoderOutput`.
        """
        if in_modality is None and out_modality is None:
            raise ValueError('in_modality and out_modality sequences cannot be both empty')
        if in_modality is None:
            out_pos_ids = self._norm_pos_ids(out_modality, out_pos_ids)
            x = out_modality + self.out_pos_emb(out_pos_ids)
        elif out_modality is None:
            in_pos_ids = self._norm_pos_ids(in_modality, in_pos_ids)
            x = in_modality + self.in_pos_emb(in_pos_ids)
        else:
            in_pos_ids = self._norm_pos_ids(in_modality, in_pos_ids)
            out_pos_ids = self._norm_pos_ids(out_modality, out_pos_ids)
            x_in = in_modality + self.in_pos_emb(in_pos_ids)
            x_out = out_modality + self.out_pos_emb(out_pos_ids)
            x = torch.cat((x_in, x_out), dim=1)
        if self.training or right_shift:
            x = self.right_shift(x)
        return self.decoder(x, attn_mask, head_mask, use_cache, causal, return_attn_weights, return_hidden_states)

    def _norm_pos_ids(self, x: 'Tensor', pos_ids: 'Optional[Tensor]'=None) ->Tensor:
        _, seq_len, _ = x.shape
        if pos_ids is None:
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)[None, :]
        if pos_ids.shape[1] != seq_len:
            raise ValueError(f'Input sequence and position ids must be equal in length: {pos_ids.shape[1]} != {seq_len}')
        return pos_ids


class RightShift(nn.Module):
    """Shifts the embedding vectors along the sequence dimension to the right.

    Since the decoder progresses by taking the token it generates in the previous step, before it
    has generated anything it needs a token to start with. Hence, the start-of-sentence (SOS) token.
    The SOS token is a learnable parameter of the decoder and the choice of its initialization is taken
    from VideoGPT: https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/attention.py#L517

    Args:
        embedding_dim (int): Dimension of the embedding vector for each token along the sequence.

    Attributes:
        sos (nn.Parameter): The starting token to be prepended to the sequence.
    """

    def __init__(self, embedding_dim: 'int') ->None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sos = nn.Parameter(torch.FloatTensor(embedding_dim).normal_(std=0.02))

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): An input tensor of shape ``(b, seq_len, emb_dim)``.

        Returns;
            A tensor of the same shape as that of the input with the ``sos`` token prepended.
        """
        x_shape = list(x.shape)
        x = x.flatten(start_dim=1, end_dim=-2)
        sos = self.sos.unsqueeze(0).unsqueeze(1).repeat(x_shape[0], 1, 1)
        x = torch.cat((sos.data, x[:, :-1, :]), dim=1)
        x = x.view(*x_shape)
        return x


def calculate_transpose_padding(kernel_size: 'Union[int, Tuple[int, ...]]', stride: 'Union[int, Tuple[int, ...]]', input_shape: 'Union[Size, Tuple[int, ...]]', input_pad: 'Union[int, Tuple[int, ...]]'=0) ->Tuple[Tuple, Tuple]:
    """Calculates padding for transposed convolution based on input dims, kernel size, and stride.

    Pads to match the 'SAME' padding in Keras, i.e., with a stride of 1 output is guaranteed
    to have the same shape as input, with stride 2 the dimensions of output are doubled.

    The 'padding' argument in ConvTranspose effectively trims the output, and the 'output_padding'
    argument effectively expands the output. These two knobs are adjusted to meet desired output dim.

    Args:
        kernel_size (int or Tuple[int, ...]): Size of convolutional kernel.
        stride (int or Tuple[int, ...]): Stride amount of kernel.
        input_shape (Size or Tuple[int, ...]): Shape of input, without batch or channel dimension.
        input_pad (int or Tuple[int, ...]): Amount of padding added to input, must be twice length of
            kernel/stride/input_shape.

    Returns:
        A tuple of padding and output_padding to be used in ConvTranspose layers
    """
    n_dims = len(input_shape)
    if isinstance(kernel_size, int):
        kernel_size = tuple(repeat(kernel_size, n_dims))
    if isinstance(stride, int):
        stride = tuple(repeat(stride, n_dims))
    if isinstance(input_pad, int):
        input_pad = tuple(repeat(input_pad, n_dims * 2))
    if not len(kernel_size) == len(stride) == len(input_shape):
        raise ValueError('dims for kernel, stride, and input must match')
    if len(input_pad) % 2 != 0 or len(input_pad) // 2 != len(input_shape):
        raise ValueError('input_pad length must be twice the number of dims')
    transpose_pad = []
    output_pad = []
    for i, (d, k, s) in enumerate(zip(input_shape, kernel_size, stride)):
        output_shape_actual = k + (d + input_pad[2 * i] + input_pad[2 * i + 1] - 1) * s
        output_shape_expected = d * s
        transpose_pad.append(max((output_shape_actual - output_shape_expected + 1) // 2, 0))
        output_pad.append(output_shape_expected - (output_shape_actual - transpose_pad[-1] * 2))
    transpose_pad = tuple(transpose_pad)
    output_pad = tuple(output_pad)
    return transpose_pad, output_pad


class SamePadConvTranspose3d(nn.Module):
    """Performs a same padded transposed convolution on a 3D input.

    This ensures output shape in input shape multiplied by stride.

    Code reference:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        in_channels (int): Number of channels in input, same as Conv3d
        out_channels (int): Number of channels for output, same as Conv3d
        kernel_size (int or Tuple[int, int, int]): Size of convolutional filter, same as Conv3d
        stride (int or Tuple[int, int, int], optional): Stride for convolution, same as Conv3d
        bias (bool, optional): If ``True`` use a bias for convolutional layer or not,
            same as ``nn.Conv3d``. Defaults to ``True``.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'Union[int, Tuple[int, int, int]]', stride: 'Union[int, Tuple[int, int, int]]'=1, bias: 'bool'=True, **kwargs: Any) ->None:
        super().__init__()
        self.pad_input: 'Tuple' = None
        self.kernel_size = kernel_size
        self.stride = stride
        if 'padding' in kwargs:
            warnings.warn('Padding was specified but will not be used in favor of same padding,                 use ConvTranspose3d directly for custom padding')
        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, **kwargs)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): Input of shape ``(b, c, d1, d2, d3)``.
        """
        if self.pad_input is None:
            self.pad_input = calculate_same_padding(self.kernel_size, self.stride, x.shape[2:])
            self.convt.padding, self.convt.output_padding = calculate_transpose_padding(self.kernel_size, self.stride, x.shape[2:], self.pad_input[::-1])
        return self.convt(F.pad(x, self.pad_input))


class VideoDecoder(nn.Module):
    """Decoder for Video VQVAE.

    Takes quantized output from codebook and applies a ``SamePadConv3d`` layer, a stack of
    ``AttentionResidualBlocks``, followed by a specified number of ``SamePadConvTranspose3d``
    layers. The residual blocks use Axial Attention to enhance representations of video data
    without significantly increasing computational cost.

    Follows VideoGPT's implementation:
        https://github.com/wilson1yan/VideoGPT/blob/master/videogpt/vqvae.py

    Args:
        out_channel_dims (Tuple[int, ...]): Output channel dimension for each layer in conv stack.
        kernel_sizes (Tuple[Tuple[int, int, int], ...]): Kernel sizes for each layer in conv stack.
        strides (Tuple[Tuple[int, int, int], ...]): Strides for each layer in conv stack
        input_dim (int): Input channel dimension for first conv layer before attention stack
        n_res_layers (int): Number of ``AttentionResidualBlocks`` to include. Default is ``4``.
        attn_hidden_dim (int): Size of hidden dimension in attention block. Default is ``240``.
        kwargs (Any): Keyword arguments to be passed into ``SamePadConvTranspose3d`` and used by
            ``nn.ConvTranspose3d``.

    Raises:
        ValueError: If the lengths of ``out_channel_dims``, ``kernel_sizes``, and ``strides`` are not
            all equivalent.
    """

    def __init__(self, out_channel_dims: 'Tuple[int, ...]', kernel_sizes: 'Tuple[Tuple[int, int, int], ...]', strides: 'Tuple[Tuple[int, int, int], ...]', input_dim: 'int', n_res_layers: 'int'=4, attn_hidden_dim: 'int'=240, **kwargs: Any):
        super().__init__()
        assert_equal_lengths(out_channel_dims, kernel_sizes, strides, msg='out_channel_dims, kernel_sizes, and strides must be same length.')
        self.conv_in = SamePadConv3d(input_dim, attn_hidden_dim, kernel_size=1, stride=1)
        self.res_stack = nn.Sequential(*[AttentionResidualBlock(attn_hidden_dim) for _ in range(n_res_layers)], nn.BatchNorm3d(attn_hidden_dim), nn.ReLU())
        transpose_convolutions: 'List[nn.Module]' = []
        n_conv_layers = len(out_channel_dims)
        for i in range(n_conv_layers):
            in_channel = out_channel_dims[i - 1] if i > 0 else attn_hidden_dim
            out_channel = out_channel_dims[i]
            kernel = kernel_sizes[i]
            stride = strides[i]
            transpose_convolutions.append(SamePadConvTranspose3d(in_channel, out_channel, kernel, stride, bias=True, **kwargs))
            if i < n_conv_layers - 1:
                transpose_convolutions.append(nn.ReLU())
        self.convts = nn.Sequential(*transpose_convolutions)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): Input quantized embeddings with shape ``(b, emb_dim, d1, d2, d3)``.
        """
        in_channel = x.shape[1]
        if in_channel != self.conv_in.conv.in_channels:
            raise ValueError(f'expected input channel dim to be {self.conv_in.conv.in_channels}, but got {in_channel}')
        h = self.conv_in(x)
        h = self.res_stack(h)
        h = self.convts(h)
        return h


class CodebookOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.modules.layers.codebook.Codebook`.

    Attributes:
        encoded_flat (Tensor): The flattened encoder output of shape ``(b x d1 x ... x dn, c)``.
        quantized_flat (Tensor): The nearest embeddings for the encoded of shape ``(b x d1 x ... x dn, emb_dim)``.
        codebook_indices (Tensor): Indices of the nearest embeddings of shape ``(b, d1, d2, ..., dn)``.
        quantized (Tensor): The nearest embeddings reshaped back to ``(b, emb_dim, d1, ..., dn)``.
    """
    encoded_flat: 'Tensor'
    quantized_flat: 'Tensor'
    codebook_indices: 'Tensor'
    quantized: 'Tensor'


class Codebook(nn.Module):
    """Bottleneck layer of VQVAE model

    Codebook provides an embedding layer that takes in the output of an encoder
    and performs a nearest-neighbor lookup in the embedding space.
    Vector quantization was introduced in Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)
    to generate high-fidelity images, videos, and audio data.
    The embedding weights are trained with exponential moving average updates as described
    in original paper.

    Code was largely inspired by a PyTorch implementation of the author's original code, found here:
    https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    and by the implementation in MUGEN (Hayes et al. 2022), found here:
    https://github.com/mugen-org/MUGEN_baseline/blob/main/lib/models/video_vqvae/vqvae.py

    Args:
        num_embeddings (int): Number of vectors in the embedding space.
        embedding_dim (int): Dimensionality of the embedding vectors.
        decay (float, optional): Factor used in exponential moving average update of the embeddings.
            Defaults to ``0.99``.
        codebook_usage_threshold (float, optional): Threshold for the average number of times an embedding vector
            is chosen below which it will be re-initialized. Defaults to ``1.0``.
        epsilon (float, optional): Noise used in Laplace smoothing of codebook usage. Defaults to ``1e-7``.
    """

    def __init__(self, num_embeddings: 'int', embedding_dim: 'int', decay: 'float'=0.99, codebook_usage_threshold: 'float'=1.0, epsilon: 'float'=1e-07) ->None:
        super().__init__()
        randn_init_embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', randn_init_embedding.clone())
        self.register_buffer('code_usage', torch.zeros(num_embeddings))
        self.register_buffer('code_avg', randn_init_embedding.clone())
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self._decay = decay
        self._epsilon = epsilon
        self.codebook_usage_threshold = codebook_usage_threshold
        self._is_embedding_init = False

    def _load_from_state_dict(self, state_dict: 'Mapping[str, Any]', prefix: 'str', local_metadata: 'Mapping', strict: 'bool', missing_keys: 'List[str]', unexpected_keys: 'List[str]', error_msgs: 'List[str]') ->None:
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._is_embedding_init = True

    def _tile(self, x: 'Tensor', n: 'int') ->Tensor:
        num_vectors, num_channels = x.shape
        if num_vectors < n:
            num_repeats = (n + num_vectors - 1) // num_vectors
            std = 0.01 / torch.sqrt(torch.tensor(num_channels))
            x = x.repeat(num_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _get_random_vectors(self, x: 'Tensor', n: 'int') ->Tensor:
        x_tiled = self._tile(x, n)
        idx = torch.randperm(x_tiled.shape[0])
        x_rand = x_tiled[idx][:n]
        return x_rand

    def _preprocess(self, encoded: 'Tensor') ->Tuple[Tensor, Size]:
        encoded_permuted = shift_dim(encoded, 1, -1)
        permuted_shape = encoded_permuted.shape
        encoded_flat = encoded_permuted.view(-1, permuted_shape[-1])
        if encoded_flat.shape[-1] != self.embedding_dim:
            raise ValueError(f'Expected {encoded_flat.shape[-1]} to be embedding size of {self.embedding_dim}')
        return encoded_flat, permuted_shape

    def _postprocess(self, quantized_flat: 'Tensor', permuted_shape: 'Union[Size, Tuple]') ->Tensor:
        quantized_permuted = quantized_flat.view(permuted_shape)
        quantized = shift_dim(quantized_permuted, -1, 1)
        return quantized

    def _init_embedding(self, encoded_flat: 'Tensor') ->None:
        self._is_embedding_init = True
        encoded_flat_rand = self._get_random_vectors(encoded_flat, self.num_embeddings)
        self.embedding = encoded_flat_rand
        self.code_avg = encoded_flat_rand
        self.code_usage = torch.ones(self.num_embeddings)

    def _ema_update_embedding(self, encoded_flat: 'Tensor', codebook_indices: 'Tensor') ->None:
        codebook_onehot = nn.functional.one_hot(codebook_indices, num_classes=self.num_embeddings).type(torch.float)
        codebook_selection_count = torch.sum(codebook_onehot, 0)
        self.code_usage.mul_(self._decay).add_(codebook_selection_count, alpha=1 - self._decay)
        n = torch.sum(self.code_usage)
        self.code_usage.add_(self._epsilon).divide_(n + self.num_embeddings * self._epsilon).mul_(n)
        encoded_per_codebook = torch.matmul(codebook_onehot.t(), encoded_flat)
        self.code_avg.mul_(self._decay).add_(encoded_per_codebook, alpha=1 - self._decay)
        self.embedding = self.code_avg / self.code_usage.unsqueeze(1)
        encoded_flat_rand = self._get_random_vectors(encoded_flat, self.num_embeddings)
        self.embedding = torch.where(self.code_usage.unsqueeze(1) >= self.codebook_usage_threshold, self.embedding, encoded_flat_rand)

    def _quantize(self, encoded_flat: 'Tensor') ->Tuple[Tensor, Tensor]:
        distances = torch.cdist(encoded_flat, self.embedding, p=2.0) ** 2
        codebook_indices_flat = torch.argmin(distances, dim=1)
        quantized_flat = F.embedding(codebook_indices_flat, self.embedding)
        if self.training:
            self._ema_update_embedding(encoded_flat, codebook_indices_flat)
        quantized_flat = encoded_flat + (quantized_flat - encoded_flat).detach()
        return quantized_flat, codebook_indices_flat

    def forward(self, z: 'Tensor') ->CodebookOutput:
        """
        Args:
            z (Tensor): Tensor containing a batch of encoder outputs of shape ``(b, c, d1, ..., dn)``.

        Returns:
            An instance of :class:`~torchmultimodal.modules.layers.codebook.CodebookOutput`.
        """
        encoded_flat, permuted_shape = self._preprocess(z)
        if not self._is_embedding_init and self.training:
            self._init_embedding(encoded_flat)
        quantized_flat, codebook_indices_flat = self._quantize(encoded_flat)
        quantized = self._postprocess(quantized_flat, permuted_shape)
        codebook_indices = codebook_indices_flat.view(z.shape[0], *z.shape[2:])
        return CodebookOutput(encoded_flat, quantized_flat, codebook_indices, quantized)

    def extra_repr(self) ->str:
        return 'num_embeddings={}, embedding_dim={}'.format(self.num_embeddings, self.embedding_dim)

    def lookup(self, indices: 'Tensor') ->Tensor:
        return F.embedding(indices, self.embedding)


class VQVAEOutput(NamedTuple):
    """Outputs from :class:`~torchmultimodal.models.vqvae.VQVAE`.

    Attributes:
        decoded (Tensor): Output of the decoder.
        codebook_output (CodebookOutput): Output of codebook layer to be used in loss calculations.
    """
    decoded: 'Tensor'
    codebook_output: 'CodebookOutput'


class VQVAE(nn.Module):
    """General model for VQVAE that provides codebook layer to link user specified
    encoder and decoder.

    Vector Quantized Variational Autoencoder is a type of autoencoder that defines
    an embedding of discrete vectors as the latent variables in the bottleneck layer
    instead of normally distributed latent variables as in a standard VAE. This enables
    high-fidelity reconstruction of input data. It was first introduced in "Neural
    Discrete Representation Learning" (Oord et al. 2017) and has since seen success in
    tokenizing and generating high-resolution image, audio, and video data.

    Args:
        encoder (nn.Module): Model that accepts single Tensor as input in forward, ``encoder(x)``.
            Will be used to project input into codebook layer. Expects channel
            dim of encoder output to match ``embedding_dim`` of codebook.
            See :class:`~torchmultimodal.modules.layers.codebook.Codebook`.
        decoder (nn.Module): Model that accepts single Tensor as input in forward, ``decoder(x)``.
            Should be able to accept output shape of codebook layer, which matches output shape of
            the encoder.
        num_embeddings (int): Number of embedding vectors in codebook.
        embedding_dim (int): Dimensionality of embedding vectors in codebook.
    """

    def __init__(self, encoder: 'nn.Module', decoder: 'nn.Module', num_embeddings: 'int', embedding_dim: 'int') ->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.codebook = Codebook(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def latent_shape(self, input_shape: 'Union[Size, Tuple]') ->Tuple[int, ...]:
        """Returns the downsampled shape of the encoder output: (d1, ..., dn)"""
        if not hasattr(self.encoder, 'get_latent_shape'):
            raise AttributeError(f"Missing attribute 'get_latent_shape' of the encoder {self.encoder}")
        return self.encoder.get_latent_shape(input_shape)

    def encode(self, x: 'Tensor', return_embeddings: 'bool'=False) ->Union[Tuple[Tensor, Tensor], Tensor]:
        """Converts input data to token ids

        Args:
            x (Tensor): Input data of shape ``(b, c, d1, ..., dn)``.
            return_embeddings (bool): Flag to return also the quantized embeddings. Defaults to ``False``.

        Returns:
            * A tensor of token ids: ``(b, d1, ...., dn)``
            * A tuple of token ids and quantized embeddings ``(b, emb_dim, d1, ..., dn)``.
        """
        encoded = self.encoder(x)
        out = self.codebook(encoded)
        indices = out.codebook_indices
        quantized = out.quantized
        if return_embeddings:
            return indices, quantized
        return indices

    def decode(self, indices: 'Tensor') ->Tensor:
        """Converts token ids back to data"""
        quantized = self.lookup(indices)
        quantized = shift_dim(quantized, -1, 1)
        return self.decoder(quantized)

    def lookup(self, indices: 'Tensor') ->Tensor:
        if not hasattr(self.codebook, 'lookup'):
            raise AttributeError(f"Missing attribute 'lookup' of the codebook {self.codebook}")
        return self.codebook.lookup(indices)

    def forward(self, x: 'Tensor') ->VQVAEOutput:
        """
        Args:
            x (Tensor): Input data of shape ``(b, c, d1, ..., dn)``.

        Returns:
            An instance of :class:`~torchmultimodal.models.vqvae.VQVAEOutput`.
        """
        encoded = self.encoder(x)
        codebook_output = self.codebook(encoded)
        decoded = self.decoder(codebook_output.quantized)
        return VQVAEOutput(decoded, codebook_output)


POOLING_TYPES = ['sum', 'mean', 'max']


class EmbeddingEncoder(nn.Module):
    """Combine embeddings for tensor representing list of indices based on pooling type

    Args:
        embedding (nn.Embedding): embedding module
        pooling_type (str): pooling function to combine the embeddings like sum. Choose
        from pooling_types
        pooling_dim (int) : dimension along which the pooling function is applied
        use_hash (bool): if hashing based on embedding vocab size if applied to input
        before embedding layer

    Inputs:
        x (Tensor): Tensor bsz x max seq length representing (padded) list of indices
        for embedding

    """

    def __init__(self, embedding: 'nn.Embedding', pooling_type: 'str', pooling_dim: 'int'=1, use_hash: 'bool'=False):
        super().__init__()
        self.embedding = embedding
        if pooling_type not in POOLING_TYPES:
            raise ValueError(f'pooling type should be in {POOLING_TYPES}, found {pooling_type}')
        self.pooling_type = pooling_type
        self.pooling_dim = pooling_dim
        self.use_hash = use_hash

    def forward(self, x: 'Tensor') ->Tensor:
        if self.use_hash:
            x = x % (self.embedding.num_embeddings - 1) + 1
        out = self.embedding(x)
        if self.pooling_type == 'sum':
            out = torch.sum(out, dim=self.pooling_dim)
        elif self.pooling_type == 'mean':
            out = torch.mean(out, dim=self.pooling_dim)
        else:
            out = torch.max(out, dim=self.pooling_dim).values
        return out


class DeepsetFusionModule(nn.Module):
    """
    Fuse embeddings through stacking followed by pooling strategy and MLP
    See https://arxiv.org/pdf/2003.01607.pdf

    Args:
        channel_to_encoder_dim (Dict[str, int]): mapping of channel name to the        encoding dimension
        mlp (nn.Module): MLP with in dim as projection dim (min of embed dim).        Use MLP for mlp_classifier for default mlp.
        pooling_function (Callable): Pooling function to combine the tensors,        like torch.median        apply_attention (bool): If self attention (2 layer net) is applied before        stacking embeddings, defaults to False.
        attention_dim (int): intermediate dim for attention layer.        defaults to projection dim / 2
        modality_normalize (bool): If normalization is applied along the modality axis,        defaults to False
        norm_factor(float): norm factor for normalization, defaults to 2.0
        use_auto_mapping(bool): If true, projection layer to min embedding dim         is applied to the embeddings. defaults to False

    """

    def __init__(self, channel_to_encoder_dim: 'Dict[str, int]', mlp: 'nn.Module', pooling_function: 'Callable', apply_attention: 'bool'=False, attention_dim: 'Optional[int]'=None, modality_normalize: 'bool'=False, norm_factor: 'float'=2.0, use_auto_mapping: 'bool'=False):
        super().__init__()
        self.apply_attention = apply_attention
        self.modality_normalize = modality_normalize
        self.norm_factor = norm_factor
        self.use_auto_mapping = use_auto_mapping
        projection_dim = DeepsetFusionModule.get_projection_dim(channel_to_encoder_dim, use_auto_mapping)
        if self.use_auto_mapping:
            self.projections = nn.ModuleDict({channel: nn.Linear(dim, projection_dim) for channel, dim in channel_to_encoder_dim.items()})
        else:
            self.projections = nn.ModuleDict({channel: nn.Identity() for channel in channel_to_encoder_dim})
        if self.apply_attention:
            self.attention: 'nn.Module'
            if attention_dim is None:
                attention_dim = projection_dim // 2
            self.attention = nn.Sequential(nn.Linear(projection_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1), nn.Softmax(dim=-2))
        else:
            self.attention = nn.Identity()
        self.pooling_function = pooling_function
        self.mlp = mlp

    def forward(self, embeddings: 'Dict[str, Tensor]') ->Tensor:
        projections = {}
        for channel, projection in self.projections.items():
            projections[channel] = projection(embeddings[channel])
        embedding_list = [projections[k] for k in sorted(projections.keys())]
        stacked_embeddings = torch.stack(embedding_list, dim=1)
        if self.apply_attention:
            attn_weights = self.attention(stacked_embeddings)
            stacked_embeddings = stacked_embeddings * attn_weights
        if self.modality_normalize:
            normalized_embeddings = F.normalize(stacked_embeddings, p=self.norm_factor, dim=1)
        else:
            normalized_embeddings = F.normalize(stacked_embeddings, p=self.norm_factor, dim=2)
        pooled_features = self._pool_features(normalized_embeddings)
        fused = self.mlp(pooled_features)
        return fused

    @classmethod
    def get_projection_dim(cls, channel_to_encoder_dim: 'Dict[str, int]', use_auto_mapping: 'bool') ->int:
        if use_auto_mapping:
            projection_dim = min(channel_to_encoder_dim.values())
        else:
            encoder_dim = set(channel_to_encoder_dim.values())
            if len(encoder_dim) != 1:
                raise ValueError('Encoder dimension should be same for all channels                     if use_auto_mapping is set to false')
            projection_dim = encoder_dim.pop()
        return projection_dim

    def _pool_features(self, embeddings: 'Tensor') ->Tensor:
        pooled_embeddings = self.pooling_function(embeddings, dim=1)
        if torch.jit.isinstance(pooled_embeddings, Tuple[Tensor, Tensor]):
            return pooled_embeddings.values
        if not isinstance(pooled_embeddings, Tensor):
            raise ValueError(f'Result from pooling function should be a tensor.             {self.pooling_function} does not satisfy that')
        return pooled_embeddings


class DeepsetFusionWithTransformer(DeepsetFusionModule):

    def __init__(self, channel_to_encoder_dim: 'Dict[str, int]', mlp: 'nn.Module', pooling_function: 'nn.TransformerEncoder', apply_attention: 'bool'=False, attention_dim: 'Optional[int]'=None, modality_normalize: 'bool'=False, norm_factor: 'float'=2.0, use_auto_mapping: 'bool'=False):
        super().__init__(channel_to_encoder_dim, mlp, pooling_function, apply_attention, attention_dim, modality_normalize, norm_factor, use_auto_mapping)

    def _pool_features(self, embeddings: 'Tensor') ->Tensor:
        pooled = self.pooling_function(embeddings)
        return pooled[:, 0, :]


class MILEncoder(nn.Module):
    """
    Multi instance learning encoder that partitions the input into a set of inputs
    and uses a shared encoder followed by deepset
    fusion to get a pooled representation of the entire input. Example use is to build a
    single representation from embeddings of all images in a post.

    Args:
        partition_sizes (List[int]): list of size for each partition of the input
        shared_encoder (nn.Module): Shared encoder for each partition of the input.
        shared_encoder_dim (int) : Output dimension of the encoders
        Following fields are same as the params for deepset fusion
        mlp (nn.Module): MLP with in dim as projection dim (min of embed dim).        Use MLP from mlp_classifier for default mlp implementation.
        pooling_function (Callable): Pooling function to combine the tensors,        like torch.median
        apply_attention (bool): If self attention is applied before        stacking embeddings, defaults to False
        modality_normalize (bool): If normalization is applied along the modality axis,        defaults to False
        norm_factor(float): norm factor for normalization, defaults to 2.0
        use_auto_mapping(bool): If true, projection layer to min embedding dim         is applied to the embeddings. defaults to False

    """

    def __init__(self, partition_sizes: 'List[int]', shared_encoder: 'nn.Module', shared_encoder_dim: 'int', mlp: 'nn.Module', pooling_function: 'Callable', apply_attention: 'bool'=False, attention_dim: 'Optional[int]'=None, modality_normalize: 'bool'=False, norm_factor: 'float'=2.0, use_auto_mapping: 'bool'=False):
        super().__init__()
        self.partition_sizes = partition_sizes
        self.shared_encoder = shared_encoder
        channel_to_encoder_dim = {}
        for i in range(len(partition_sizes)):
            channel_to_encoder_dim[self.get_channel_name(i)] = shared_encoder_dim
        deepset_fusion_cls = DeepsetFusionWithTransformer if isinstance(pooling_function, nn.TransformerEncoder) else DeepsetFusionModule
        self.deepset_fusion: 'Union[DeepsetFusionWithTransformer, DeepsetFusionModule]' = deepset_fusion_cls(channel_to_encoder_dim=channel_to_encoder_dim, mlp=mlp, pooling_function=pooling_function, apply_attention=apply_attention, attention_dim=attention_dim, modality_normalize=modality_normalize, norm_factor=norm_factor, use_auto_mapping=use_auto_mapping)

    def get_channel_name(self, id: 'int') ->str:
        return f'mil_{id}'

    def forward(self, x: 'Tensor') ->Tensor:
        idx = 0
        input_size = x.size(dim=1)
        if input_size != sum(self.partition_sizes):
            raise ValueError(f'partition sizes should sum to the input size {input_size}')
        partitioned_input = torch.split(x, self.partition_sizes, dim=1)
        encoded_input = {}
        for idx, input in enumerate(partitioned_input):
            key = self.get_channel_name(idx)
            encoded_input[key] = self.shared_encoder(input)
        return self.deepset_fusion(encoded_input)


class VisionTransformer(nn.Module):
    """
    General image transformer encoder with embeddings. Similar to ``VisionTransformer`` in torchvision,
    but more composable. Can be constructed with any user-provided embeddings, encoder, and task head.

    Attributes:
        embeddings (nn.Module): Module that projects image pixels into embeddings.
            See :py:class: PatchEmbeddings for interface.
        encoder (nn.Module): Module for transformer encoder. See :py:class: TransformerEncoder for interface.
        pooler (nn.Module, optional): Module for pooler to be applied after layernorm. Defaults to ``None``.
        weight_init_fn (Callable, optional): function for custom weight initialization of both the transformer
            encoder and embeddings. See :py:func: init_transformer_weights as an example. Defaults to ``None``.

    Args:
        images (Tensor): Tensor of input images of shape ``(b, c, h, w)``.
        image_patches_mask (Tensor, optional): Tensor indicating which patches to replace with mask tokens,
            shape ``(b, seq_len)``, where seq_len = (image_size // patch_size) ** 2
        attention_mask (Tensor, optional): Tensor indicating which tokens to attend to, shape ``(b, seq_len + 1)``.
            Concatenating class_token adds 1 to seq_len.
    """

    def __init__(self, embeddings: 'nn.Module', encoder: 'nn.Module', pooler: 'Optional[nn.Module]'=None, weight_init_fn: 'Optional[Callable]'=None) ->None:
        super().__init__()
        torch._C._log_api_usage_once(f'torchmultimodal.{self.__class__.__name__}')
        self.embeddings = embeddings
        self.encoder = encoder
        self.pooler = pooler
        if weight_init_fn:
            self.apply(weight_init_fn)

    def forward(self, images: 'Tensor', image_patches_mask: 'Optional[Tensor]'=None, attention_mask: 'Optional[Tensor]'=None) ->TransformerOutput:
        embedding_output = self.embeddings(images, image_patches_mask=image_patches_mask).embeddings
        encoder_output = self.encoder(embedding_output, attention_mask=attention_mask, return_hidden_states=True)
        last_hidden_state = encoder_output.last_hidden_state
        if self.pooler is not None:
            assert last_hidden_state is not None, 'For pooler, last hidden state cannot be None.'
            pooled_output = self.pooler(last_hidden_state)
        else:
            pooled_output = None
        return TransformerOutput(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_output.hidden_states, attentions=encoder_output.attentions)


class GlobalAveragePooler(nn.Module):
    """
    Global average pooler that averages the embeddings over all the patches in a sample
    and applies layer norm and an optional linear head on top.
    Args:
        input_dim (int): hidden dim of the transformer last hidden state.
        output_dim (Optional[int]): output dim of the linear head. if None, no linear head is added. Defaults to None.
        ln_eps (float): layer norm epsilon. Defaults to 1e-6.
        init_weights (Optional[Callable]): function to initialize weights of the module. Defaults to None.

    """

    def __init__(self, input_dim: 'int', output_dim: 'Optional[int]'=None, ln_eps: 'float'=1e-06, init_weights: 'Optional[Callable]'=None) ->None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=ln_eps)
        if output_dim:
            self.head: 'nn.Module' = nn.Linear(input_dim, output_dim)
        else:
            self.head = nn.Identity()
        if init_weights is not None:
            self.apply(init_weights)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Args:
            x (Tensor): Input tensor with shape bsz x len x input_dim. The first entry in assumed to be CLS
                and ignored during averaging
        Returns:
            Tensor: Output tensor with shape bsz x output_dim
        """
        out = x[:, 1:, :].mean(dim=1)
        out = self.norm(out)
        out = self.head(out)
        return out


class WeightedEmbeddingEncoder(nn.Module):
    """Combine weighted embeddings for tensor representing list of indices based on
    pooling type.

    Args:
        embedding (nn.Embedding): embedding module
        pooling_function (Callable[[Tensor, int], Union[Tensor, Tuple]]): pooling function to combine the weighted embeddings,        example: torch.sum function should return a tensor or namedtuple containing the tensor in the values field like torch.max
        pooling_dim (int) : dimension along which the pooling function is applied

    Inputs:
        weights (Tensor): A float tensor of shape [batch_size x num_categories] containing the weights of a categorical feature.            The weights represent multiplier factors for the corresponding category embedding vectors.

    """

    def __init__(self, embedding: 'nn.Embedding', pooling_function: 'Callable[[Tensor, int], Union[Tensor, Tuple]]', pooling_dim: 'int'=1) ->None:
        super().__init__()
        self.embedding = embedding
        self.pooling_function = pooling_function
        self.pooling_dim = pooling_dim

    def forward(self, weights: 'Tensor') ->Tensor:
        index = torch.arange(0, weights.size(1), dtype=torch.int)
        index = index
        weighted_embeddings = self.embedding(index) * weights.unsqueeze(-1)
        pooled_embeddings = self.pooling_function(weighted_embeddings, self.pooling_dim)
        if isinstance(pooled_embeddings, Tensor):
            output: 'Tensor' = pooled_embeddings
        else:
            assert hasattr(pooled_embeddings, 'values'), 'pooled embeddings should be Tensor or tuple with values field as Tensor'
            output = pooled_embeddings.values
        return output


class AttentionFusionModule(nn.Module):
    """
    Fuse embeddings through weighted sum of the corresponding linear projections.
    Linear layer for learning the weights.

    Args:
        channel_to_encoder_dim: mapping of channel name to the encoding dimension
        encoding_projection_dim: common dimension to project the encodings to.
        defaults to min of the encoder dim if not set

    """

    def __init__(self, channel_to_encoder_dim: 'Dict[str, int]', encoding_projection_dim: 'Optional[int]'=None):
        super().__init__()
        attn_in_dim = sum(channel_to_encoder_dim.values())
        self.attention = nn.Sequential(nn.Linear(attn_in_dim, len(channel_to_encoder_dim)), nn.Softmax(-1))
        if encoding_projection_dim is None:
            encoding_projection_dim = min(channel_to_encoder_dim.values())
        encoding_projection = {}
        for channel in sorted(channel_to_encoder_dim.keys()):
            encoding_projection[channel] = nn.Linear(channel_to_encoder_dim[channel], encoding_projection_dim)
        self.encoding_projection = nn.ModuleDict(encoding_projection)

    def forward(self, embeddings: 'Dict[str, Tensor]') ->Tensor:
        concatenated_in = torch.cat([embeddings[k] for k in sorted(embeddings.keys())], dim=-1)
        attention_weights = self.attention(concatenated_in)
        projected_embeddings: 'List[Tensor]' = []
        for channel, projection in self.encoding_projection.items():
            projected_embedding = projection(embeddings[channel])
            projected_embeddings.append(projected_embedding)
        for i in range(len(projected_embeddings)):
            projected_embeddings[i] = attention_weights[:, i].unsqueeze(-1) * projected_embeddings[i]
        fused = torch.sum(torch.stack(projected_embeddings), dim=0)
        return fused


class ConcatFusionModule(nn.Module):
    """Module to fuse modalities via concatenation. Sorted by keys for consistency.

    Inputs:
        embeddings (Dict[str, Tensor]): A dictionary mapping modalities to their
            tensor representations.

    """

    def __init__(self, projection: 'nn.Module'=None):
        super().__init__()
        if projection:
            self.projection = projection
        else:
            self.projection = nn.Identity()

    def forward(self, embeddings: 'Dict[str, torch.Tensor]') ->torch.Tensor:
        concatenated_in = torch.cat([embeddings[k] for k in sorted(embeddings.keys())], dim=-1)
        return self.projection(concatenated_in)


class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation function

    .. math:: 	ext{GEGLU}(a,b) = a * 	ext{GELU}(b)

    where :math:`a` is the first half of the input matrices and :math:`b` is
    the second half, as descibed in the paper:
    `"GLU Variants Improve Transformer"<https://arxiv.org/pdf/2002.05202.pdf>`.
    """

    def __init__(self, dim: 'int'=-1):
        super().__init__()
        self.split_dim = dim

    def forward(self, x: 'Tensor') ->Tensor:
        x, gate = x.chunk(2, dim=self.split_dim)
        return x * F.gelu(gate)


class AttentionPooler(nn.Module):
    """
    Attention pooling layer: pools inputs to sequence length n_queries by performing
        cross-attention with learned query embeddings. Originally proposed in
        https://arxiv.org/abs/1810.00825. This implementation is based on the one
        in open_clip repo: https://tinyurl.com/4yj492sc.
    Args:
        input_embed_dim (int): Embedding dimension of inputs.
        output_embed_dim (int): Embedding dimension of outputs.
        n_head (int): Number of attention heads.
        n_queries (int): Number of queries. Defaults to 256
        layer_norm_eps (Optional[float]): Epsilon for layer norms. Defaults to 1e-5
    """

    def __init__(self, input_embed_dim: 'int', output_embed_dim: 'int', n_head: 'int', n_queries: 'int'=256, layer_norm_eps: 'float'=1e-05):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, output_embed_dim))
        self.attn = MultiHeadAttentionWithCache(dim_q=output_embed_dim, dim_kv=input_embed_dim, num_heads=n_head)
        self.ln_q = nn.LayerNorm(output_embed_dim, layer_norm_eps)
        self.ln_k = nn.LayerNorm(input_embed_dim, layer_norm_eps)
        self.ln_post = nn.LayerNorm(output_embed_dim, layer_norm_eps)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_embed_dim).
        Returns:
            Attention pooled tensor with shape
                (batch_size, n_queries, output_embed_dim).
        """
        x = self.ln_k(x)
        query = self.ln_q(self.query)
        batch_size = x.shape[0]
        query = self._repeat(query, batch_size)
        out = self.attn(query, x, x)
        assert isinstance(out, Tensor)
        out = self.ln_post(out)
        return out

    def _repeat(self, query: 'Tensor', n: 'int') ->Tensor:
        return query.unsqueeze(0).repeat(n, 1, 1)


class CascadedAttentionPooler(nn.Module):
    """
    Wrapper class to perform cascaded attention pooling given multiple attention
    poolers. E.g. in CoCa the contrastive pooler is applied on top of the outputs of
    the captioning pooler.

    Args:
        poolers (List[AttentionPooler]): List of individual attention poolers
    """

    def __init__(self, poolers: 'List[AttentionPooler]'):
        super().__init__()
        self.poolers = nn.ModuleList(poolers)

    def forward(self, x: 'Tensor') ->List[Tensor]:
        """
        Inputs:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_embed_dim).
        Returns:
            List[Tensor] containing attention pooled tensors with shapes
                (batch_size, n_queries, output_embed_dim), where n_queries and
                output_embed_dim are determined by each individual pooler.
        """
        pooler_outs = []
        for pooler in self.poolers:
            x = pooler(x)
            pooler_outs.append(x)
        return pooler_outs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    as proposed in: https://arxiv.org/abs/1910.07467

    Calcs are done in fp32.

    original impl: https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim(int) = model size
        eps(float) = epsilon
    """

    def __init__(self, dim: 'int', eps: 'float'=1e-06):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x: 'Tensor') ->Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: 'Tensor') ->Tensor:
        x_normed = self._norm(x.float()).type_as(x)
        return x_normed * self.scale


class SimpleRMSNorm(nn.Module):
    """Simple RMSNorm

    SRMSNorm(x) = (x / ∥x∥2) /√d

    as proposed in:
    Scaling TransNormer to 175 Billion Parameters
    https://arxiv.org/abs/2307.14995

    Usage: use as drop in replacement for RMSNorm.
    """

    def __init__(self, dim: 'int', eps: 'float'=1e-12):
        super().__init__()
        self.scaling = dim ** 0.5
        self.eps = eps

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        denom = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps).expand_as(x)
        return x / denom * self.scaling


class BroadcastedPositionEmbedding(nn.Module):
    """Spatiotemporal broadcasted positional embeddings.

    Based on broadcasted position embedding algorithm in codebase:
        https://github.com/wilson1yan/VideoGPT/blob/c21cc7e2579f820cb2b90097406d72cf69a46474/videogpt/attention.py#L458

        Each embedding vector of the ``i``-th dim is repeated by ``N`` times, where
    :math:`N = \\prod_{j>i}\\text{dim}[j]`.

    Args:
        latent_shape (Tuple[int, ...]): Shape of encoded data before batching and embedding.
        embedding_dim (int): The size of each embedding vector.

    Raises:
        ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``.
    """

    def __init__(self, latent_shape: 'Tuple[int, ...]', embedding_dim: 'int') ->None:
        """
        Args:
            latent_shape (Tuple[int, ...]): Shape of encoded data before batching and embedding.
            embedding_dim (int): The size of each embedding vector.

        Raises:
            ValueError: if ``embedding_dim`` is not an integer multiple of ``len(shape)``
        """
        super().__init__()
        if embedding_dim % len(latent_shape) != 0:
            raise ValueError(f'Embedding dim {embedding_dim} modulo len(latent_shape) {len(latent_shape)} is not zero')
        self.latent_shape = latent_shape
        self.n_dim = n_dim = len(self.latent_shape)
        self.embedding_dim = embedding_dim
        self.embedding = nn.ParameterDict({f'd_{i}': nn.Parameter(torch.randn(self.latent_shape[i], embedding_dim // n_dim) * 0.01) for i in range(n_dim)})

    @property
    def indices(self) ->Tensor:
        """Returns broadcasted indices of the data

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 3), embedding_dim=6)
            >>> pos_emb.indices
            tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        """
        return torch.cartesian_prod(*[torch.arange(s) for s in self.latent_shape])

    def _broadcast(self, i: 'int') ->Tensor:
        """Broadcasts the ``i``-th embedding matrix ``(self.latent_shape[i], self.embedding_dim // n_dim)`` along the other
        dims of ``self.latent_shape``. The embedding dim is not touched.

        For example::

            >>> pos_emb = BroadcastedPositionEmbedding(shape=(2, 4), embedding_dim=6)
            >>> print(pos_emb.embedding["d_0"].shape)
            torch.Size([2, 3])
            >>> pos_emb.embedding["d_0"] = nn.Parameter(torch.tensor([[0., 0., 0.], [0., 0., 1.]]))
            >>> out = pos_emb._broadcast(i=0)
            >>> print(out)
            tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]])
            >>> print(out.shape)
            (2, 4, 3)

        The input is broadcasted along the second dim ``4`` since it's the ``0``-th embedding constructed w.r.t the
        first dim ``2``.
        """
        emb = self.embedding[f'd_{i}']
        emb = emb.view(*itertools.repeat(1, i), self.latent_shape[i], *itertools.repeat(1, self.n_dim - i - 1), -1)
        emb = emb.expand(*self.latent_shape, -1)
        return emb

    def forward(self, position_ids: 'Tensor') ->Tensor:
        """
        Args:
            position_ids (Tensor): batches of of 1D integer tensors indicating locations of the broadcasted
                position embeddings to be returned.

        Returns:
            A tensor with the position embeddings selected by position ids.

        Raises:
            IndexError: If any position id(s) provided is outside of the indices range.
        """
        invalid_ids = position_ids[torch.logical_or(position_ids >= len(self.indices), position_ids < -1)]
        if len(invalid_ids):
            raise IndexError(f'Invalid position ids: {invalid_ids}')
        embeddings = []
        for i in range(self.n_dim):
            emb = self._broadcast(i)
            embeddings.append(emb)
        embeddings = torch.cat(embeddings, dim=-1)
        indices = [*self.indices[position_ids].permute(2, 1, 0)]
        embeddings = embeddings[indices].transpose(0, 1)
        return embeddings


class ImageTextContrastiveLoss(nn.Module):
    """
    Compute the image-text contrastive loss from image-text similarity, as used in ALBEF.
    Support loss distillation with pseudo-targets for non-zero alpha. Compute standard contrastive loss for zero alpha.

    Inputs:
        image_to_text_sim (Tensor): Image to text similarity.
        text_to_image_sim (Tensor): Text to image similarity.
        image_to_text_sim_m (Optional[Tensor]): Image to text similarity from momentum models.
            Required if alpha is non-zero.
        text_to_image_sim_m (Optional[Tensor]): Text to image similarity from momentum models.
            Required if alpha is non-zero.
        sim_targets (Optional[Tensor]): Similarity pseudo-targets from momentum models. Default is the diagonal matrix.
            Requires all Tensor inputs to have the same size.
        alpha (Optional[float]): The interpolation value of momentum similarity and sim_targets. Default is 0.
    """

    def __init__(self) ->None:
        super().__init__()

    def forward(self, image_to_text_sim: 'Tensor', text_to_image_sim: 'Tensor', image_to_text_sim_m: 'Optional[Tensor]'=None, text_to_image_sim_m: 'Optional[Tensor]'=None, sim_targets: 'Optional[Tensor]'=None, alpha: 'Optional[float]'=0.0) ->Tensor:
        if sim_targets is None:
            sim_targets = torch.zeros(image_to_text_sim.size())
            sim_targets.fill_diagonal_(1)
        if alpha != 0:
            assert image_to_text_sim_m is not None and text_to_image_sim_m is not None, 'sim_i2t_m and sim_t2i_m cannot be none for non-zero alpha'
            with torch.no_grad():
                image_to_text_sim_targets = alpha * F.softmax(image_to_text_sim_m, dim=1) + (1 - alpha) * sim_targets
                text_to_image_sim_targets = alpha * F.softmax(text_to_image_sim_m, dim=1) + (1 - alpha) * sim_targets
        else:
            image_to_text_sim_targets = sim_targets
            text_to_image_sim_targets = sim_targets
        loss_i2t = -torch.sum(F.log_softmax(image_to_text_sim, dim=1) * image_to_text_sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(text_to_image_sim, dim=1) * text_to_image_sim_targets, dim=1).mean()
        loss_itc = (loss_i2t + loss_t2i) / 2
        return loss_itc


class CausalLanguageModelingLoss(nn.Module):
    """
    Compute the autoregressive masked language modeling loss by predicting the next token, as used in VQA.
    Support loss distillation for non-zero alpha. Compute standard mlm loss for zero alpha.

    Args:
        mask_token_id (int): The token id indicating a masked token. Default is -100.

    Inputs:
        labels (Tensor of shape (batch_size, seq_length)): The masked output tokens.
        prediction_scores (Tensor of shape (batch_size, seq_length, vocab_size)):
            The prediction scores from a prediction head.
        prediction_scores_m (Optional[Tensor] of shape (batch_size, seq_length, vocab_size)):
            The prediction scores from a momentum prediction head.
            Required if alpha is non-zero.
        alpha (float): The interpolation value between mlm_loss and loss_distill. Default is 0.
    """

    def __init__(self, mask_token_id: 'int'=-100) ->None:
        super().__init__()
        self.mask_token_id = mask_token_id

    def forward(self, labels: 'Tensor', prediction_scores: 'Tensor', prediction_scores_m: 'Optional[Tensor]'=None, alpha: 'Optional[float]'=0.0) ->Tensor:
        batch_size = labels.size(0)
        prediction_scores = prediction_scores[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        mlm_loss = F.cross_entropy(prediction_scores.view(-1, prediction_scores.shape[-1]), labels.view(-1), reduction='none')
        mlm_loss = mlm_loss.view(batch_size, -1).sum(1)
        if alpha != 0:
            assert prediction_scores_m is not None, 'prediction_scores_m cannot be None for non-zero alpha'
            with torch.no_grad():
                prediction_scores_m = prediction_scores_m[:, :-1, :].contiguous()
            loss_distill = -torch.sum(F.log_softmax(prediction_scores, dim=-1) * F.softmax(prediction_scores_m, dim=-1), dim=-1)
            loss_distill = (loss_distill * (labels != self.mask_token_id)).sum(1)
            mlm_loss = (1 - alpha) * mlm_loss + alpha * loss_distill
        return mlm_loss


@dataclass
class Blip2Stage1Losses(OrderedDict):
    """Blip-2 stage 1 losses"""
    image_text_contrastive_loss: 'torch.Tensor'
    image_text_matching_loss: 'torch.Tensor'
    image_captioning_loss: 'torch.Tensor'
    total_loss: 'torch.Tensor'


def concat_gather_all_gpu(tensor: 'Tensor', backprop_type: 'BackpropType'=BackpropType.GLOBAL, dim: 'int'=0) ->Tensor:
    """Gathers a tensor across all GPUs.

    Inputs:
        tensor (Tensor): Tensors that need to be gathered.
        backprop_type (BackpropType): whether to backpropagate gradients to all
            workers (GLOBAL), just the local worker (LOCAL), or not at all (NONE).
            Default: BackpropType.GLOBAL
        dim: the dimension over which the tensors are concatenated, default to 0.

    Returns:
        Tensor: concatenated gathered tensors across all GPUs.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    tensors_all_gpus = gather_tensor(tensor, backprop_type)
    return torch.cat(tensors_all_gpus, dim=dim)


def compute_image_text_similarity(image_features: 'torch.Tensor', text_features: 'torch.Tensor', temp: 'nn.Parameter') ->Tuple[torch.Tensor, torch.Tensor]:
    """Compute image-text similarity across all the devices for itc and itm usage.

    Inputs:
        image_features (torch.Tensor): Blip2 image output of shape [bsz, num_query_tokens, embed_dim]
        text_features (torch.Tensor): Blip2 text output of shape [bsz, embed_dim]
        temp (nn.Parameter): Temperature parameter

    Returns:
        a tuple of tensor contains image-to-text similarity and text-to-image similarity.
    """
    image_features_all = concat_gather_all_gpu(image_features, backprop_type=BackpropType.NONE)
    text_features_all = concat_gather_all_gpu(text_features, backprop_type=BackpropType.NONE)
    sim_q2t = torch.matmul(image_features.unsqueeze(1), text_features_all.unsqueeze(-1)).squeeze()
    sim_i2t, _ = sim_q2t.max(-1)
    sim_i2t = sim_i2t / temp
    sim_t2q = torch.matmul(text_features.unsqueeze(1).unsqueeze(1), image_features_all.permute(0, 2, 1)).squeeze()
    sim_t2i, _ = sim_t2q.max(-1)
    sim_t2i = sim_t2i / temp
    return sim_i2t, sim_t2i


def itc_loss(sim_i2t: 'torch.Tensor', sim_t2i: 'torch.Tensor', label_smoothing: 'float'=0.1) ->torch.Tensor:
    """Compute image-text contrastive loss by given similarity between image and text.

    Inputs:
        sim_i2t(torch.Tensor): image-to-text similarity, shape [bsz, bsz x num_gpu]
        sim_t2i (torch.Tensor): text-to-image similarity, shape [bsz, bsz x num_gpu]
        label_smoothing (Optional[float]): Label smoothing for cross-entropy. Default: 0.1.

    Returns:
        itc_loss (torch.Tensor)
    """
    rank = get_rank()
    local_batch_size = sim_i2t.size(0)
    targets = local_batch_size * rank + torch.arange(local_batch_size, device=sim_i2t.device)
    loss = (F.cross_entropy(sim_i2t, targets, label_smoothing=label_smoothing) + F.cross_entropy(sim_t2i, targets, label_smoothing=label_smoothing)) / 2
    return loss


def itg_loss(input_ids: 'torch.Tensor', prediction_scores: 'torch.Tensor', decoder_bos_token_id: 'int', pad_token_id: 'int', vocab_size: 'int', reduction: 'str'='mean', label_smoothing: 'float'=0.1) ->torch.Tensor:
    """Compute image caption loss from BLIP2 predictions.

    Inputs:
        input_ids (torch.Tensor): text input ids of shape (bsz, seq_len).
        prediction_scores (torch.Tensor): BLIP2 prediction scores, shape of (bsz, seq_len, vocab_size)
        decoder_bos_token_id (int): bos_token_id for decoder, which is used to replace CLS token.
        pad_token_id (int): pad_token_id for decoder
        vocab_size (int): vocab size of BLIP2 model
        reduction (str): reduction for loss computation, default is "mean".
        label_smoothing (float): label smoothing value for cross-entropy loss, default is 0.1.

    Returns:
        itg_loss (torch.Tensor): image caption loss.
    """
    decoder_input_ids = input_ids.clone()
    decoder_input_ids[:, 0] = decoder_bos_token_id
    labels = decoder_input_ids.masked_fill(decoder_input_ids == pad_token_id, -100)
    shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    itg_loss = F.cross_entropy(shifted_prediction_scores.view(-1, vocab_size), labels.view(-1), reduction=reduction, label_smoothing=label_smoothing)
    return itg_loss


def itm_loss(input_ids: 'torch.Tensor', image_embeds: 'torch.Tensor', sim_i2t: 'torch.Tensor', sim_t2i: 'torch.Tensor', model_query_tokens: 'nn.Parameter', qformer: 'nn.Module', itm_head: 'nn.Module', attention_mask: 'torch.Tensor') ->torch.Tensor:
    """Compute image-text matching loss
    ITM loss computation uses hard negative mining strategy. Negative text and image examples
    are selected based on their corresponding similarities.

    The concatenated image-text pairs are constructed as a size of 3 x bsz batch (pos, neg, neg)
    with text concatenated inputs (pos, pos, neg) and image inputs (pos, neg, pos).

    Query embedding output are fed into a two-class linear classifier to obtain a logit,
    and average the logits across all queries as the output matching score.

    Inputs:
        input_ids (torch.Tensor): text input ids of shape [bsz, seq_len].
        image_embeds (torch.Tensor): image embeddings returned by vision encoder
            with shape [bsz, image_embedding_dim]
        sim_i2t (torch.Tensor): image-to-text similarity, shape [bsz, bsz x num_gpu]
        sim_t2i (torch.Tensor): text-to-image similarity, shape [bsz, bsz x num_gpu]
        model_query_tokens(nn.Parameter): Blip2 query tokens
        qformer (nn.Module): Q-Former module
        itm_head (nn.Module): ITM head defined in blip2 stage1 loss
        attention_mask (torch.Tensor): attention mask for text input, shape [bsz, seq_len].

    Returns:
        itm_loss (torch.Tensor): image-text matching loss
    """
    local_batch_size = image_embeds.size(0)
    device = image_embeds.device
    text_input_ids_all_gpus = concat_gather_all_gpu(input_ids, backprop_type=BackpropType.NONE)
    text_attention_mask_all_gpus = concat_gather_all_gpu(attention_mask, backprop_type=BackpropType.NONE)
    image_embeds_all_gpus = concat_gather_all_gpu(image_embeds, backprop_type=BackpropType.GLOBAL)
    rank = get_rank()
    with torch.no_grad():
        weights_t2i_for_neg_sampling = F.softmax(sim_t2i, dim=1) + 0.0001
        weights_t2i_for_neg_sampling[:, rank * local_batch_size:rank * local_batch_size + local_batch_size].fill_diagonal_(0)
        weights_i2t_for_neg_sampling = F.softmax(sim_i2t, dim=1) + 0.0001
        weights_i2t_for_neg_sampling[:, rank * local_batch_size:rank * local_batch_size + local_batch_size].fill_diagonal_(0)
    image_embeds_neg = []
    for b in range(local_batch_size):
        neg_idx = int(torch.multinomial(weights_t2i_for_neg_sampling[b], 1).item())
        image_embeds_neg.append(image_embeds_all_gpus[neg_idx])
    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
    text_ids_neg = []
    text_atts_neg = []
    for b in range(local_batch_size):
        neg_idx = int(torch.multinomial(weights_i2t_for_neg_sampling[b], 1).item())
        text_ids_neg.append(text_input_ids_all_gpus[neg_idx])
        text_atts_neg.append(text_attention_mask_all_gpus[neg_idx])
    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_atts_neg = torch.stack(text_atts_neg, dim=0)
    text_ids_all = torch.cat([input_ids, input_ids, text_ids_neg], dim=0)
    text_atts_all = torch.cat([attention_mask, attention_mask, text_atts_neg], dim=0)
    query_tokens_itm = model_query_tokens.expand(text_ids_all.shape[0], -1, -1)
    query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long)
    attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)
    image_embeds_all = torch.cat([image_embeds, image_embeds_neg, image_embeds], dim=0)
    output_itm = qformer(input_ids=text_ids_all, query_embeds=query_tokens_itm, attention_mask=attention_mask_all, encoder_hidden_states=image_embeds_all)
    vl_embeddings = output_itm[0][:, :query_tokens_itm.size(1), :]
    vl_output = itm_head(vl_embeddings)
    itm_logits = vl_output.mean(dim=1)
    itm_labels = torch.cat([torch.ones(local_batch_size, dtype=torch.long), torch.zeros(2 * local_batch_size, dtype=torch.long)], dim=0)
    return F.cross_entropy(itm_logits, itm_labels, reduction='mean')


class Blip2Phase1Loss(nn.Module):
    """
    Blip2 phase 1 loss module

    Args:
        dim_q (int): Dimension of query tensor, this value should be the same as dim_q in qformer.
            default value is 768 as in the paper.
        enable_itc (bool): enable image-text contrastive loss, default is True
        enable_itm (bool): enable image-text matching, default is True
        enable_itg (bool): enable image caption loss, default is True
        temp (float): temperature for image-text similarity computation, default is 0.07
        label_smoothing (float): label smoothing value, default is 0.1
    """

    def __init__(self, dim_q: 'int'=768, enable_itc: 'bool'=True, enable_itm: 'bool'=True, enable_itg: 'bool'=True, temp: 'float'=0.07, label_smoothing: 'float'=0.1) ->None:
        super().__init__()
        if not enable_itc and not enable_itm and not enable_itg:
            raise ValueError('All the loss tasks are disabled, please set at least one of them.')
        self.label_smoothing = label_smoothing
        self.enable_itc = enable_itc
        self.enable_itm = enable_itm
        self.enable_itg = enable_itg
        self.itm_head = nn.Linear(dim_q, 2)
        self.temp = nn.Parameter(temp * torch.ones([]))

    def forward(self, model_output: 'Blip2Output', blip2: 'nn.Module', input_ids: 'Optional[torch.Tensor]', attention_mask: 'Optional[torch.Tensor]') ->Blip2Stage1Losses:
        """
        Inputs:
            model_output (Blip2Output): model output from BLIP2 (see blip2.py)
            blip2 (nn.Module): BLIP2 model with updated params
            input_ids (Optional[torch.Tensor]): text input ids of shape [bsz, seq_len].
            attention_mask (Optional[torch.Tensor]): text input attention mask of shape [bsz, seq_len].

        Returns:
            loss (Blip2Stage1Losses): computed loss for phase 1 tasks.
        """
        assert model_output.text_features is not None
        sim_i2t, sim_t2i = compute_image_text_similarity(model_output.image_features, model_output.text_features, temp=self.temp)
        loss_itm = torch.tensor(0.0)
        if self.enable_itm:
            assert input_ids is not None and attention_mask is not None
            loss_itm = itm_loss(input_ids=input_ids, attention_mask=attention_mask, image_embeds=model_output.image_embeddings, sim_i2t=sim_i2t, sim_t2i=sim_t2i, model_query_tokens=blip2.query_tokens, qformer=blip2.qformer.model, itm_head=self.itm_head)
        loss_itg = torch.tensor(0.0)
        if self.enable_itg:
            assert input_ids is not None and model_output.prediction_scores is not None
            loss_itg = itg_loss(input_ids=input_ids, prediction_scores=model_output.prediction_scores, decoder_bos_token_id=blip2.decoder_bos_token_id, pad_token_id=blip2.qformer.pad_token_id, vocab_size=blip2.qformer.vocab_size, label_smoothing=self.label_smoothing)
        loss_itc = torch.tensor(0.0)
        if self.enable_itc:
            loss_itc = itc_loss(sim_i2t=sim_i2t, sim_t2i=sim_t2i, label_smoothing=self.label_smoothing)
        return Blip2Stage1Losses(image_text_contrastive_loss=loss_itc, image_captioning_loss=loss_itg, image_text_matching_loss=loss_itm, total_loss=loss_itc + loss_itm + loss_itg)


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss that computed MSE between predicted and target values as described in MAE paper
    https://arxiv.org/abs/2111.06377. Loss is averaged only over masked patches.

    Args:
        normalize_target (bool) : Whether target should be normalized. Defaults to True

    """

    def __init__(self, normalize_target: 'bool'=True):
        super().__init__()
        self.normalize_target = normalize_target

    def forward(self, pred: 'Tensor', target: 'Tensor', mask: 'Tensor') ->Tensor:
        """
        Args:
            pred (Tensor): predicted tensor with shape bsz x num_patches x (patch_size ** 2 * channels)
            target (Tensor): patchified input with the same shape as pred
            mask (Tensor):  Tensor of shape bsz x num_patches indicating which patches are masked.
            1 indicates masked patch amd 0 indicated unmasked patch.
        Returns: computed loss

        """
        if mask.sum() == 0:
            raise ValueError('At least one patch must be masked')
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-06) ** 0.5
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


class CommitmentLoss(nn.Module):
    """Commitment loss calculates the mean Euclidean distance between pairs of encoder output vectors
    and their corresponding quantized vectors. It encourages an encoder to generate outputs closer to an embedding.
    This is the beta in Eq. 3 of Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)

    Args:
        commitment_cost (float): multiplicative weight for the commitment loss value
    """

    def __init__(self, commitment_cost: 'float'=1.0, **kwargs: Any) ->None:
        super().__init__()
        self.commitment_cost = commitment_cost

    def forward(self, quantized: 'Tensor', encoded: 'Tensor') ->Tensor:
        loss = F.mse_loss(quantized.detach(), encoded) * self.commitment_cost
        return loss


CLIP_DEFAULT_VOCAB_BPE_PATH = 'http://download.pytorch.org/models/text/clip_merges.bpe'


@lru_cache()
def bytes_to_unicode() ->Dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: 'Tuple[str, ...]') ->Set[Tuple[str, str]]:
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class CLIPBPETokenizer:
    """
    Construct a CLIP tokenizer. Based on byte-level Byte-Pair-Encoding.

    This implementation is adapted from https://git.io/JDTuJ.
    Example usage:
    tokenizer = CLIPBPETokenizer()
    sentence = "Hello I am using CLIP tokenizer."
    tokens = tokenizer.encode(sentence)
    tokens -> [3306, 328, 687, 1996, 9289, 32634, 23895, 269]
    decoded_sentence = tokenizer.decode(tokens)
    decoded_sentence -> "hello i am using clip tokenizer ."

    Args:
        bpe_path (str): path to the BPE file
        bos_token (str): beginning of sentence token. Defaults to "<|startoftext|>".
        eos_token (str): end of sentence token. Defaults to "<|endoftext|>".
        num_merges (Optional[int]): number of merges.
            If None, it will load all merges from the BPE file.
    """

    def __init__(self, bpe_path: 'str'=CLIP_DEFAULT_VOCAB_BPE_PATH, bos_token: 'str'='<|startoftext|>', eos_token: 'str'='<|endoftext|>', num_merges: 'Optional[int]'=None):
        self._separator = '\x01'
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with _PATH_MANAGER.open(bpe_path, 'r', encoding='utf-8') as f:
            bpe_merges = f.read().split('\n')[1:]
        num_merges = num_merges or len(bpe_merges)
        bpe_merges = bpe_merges[:num_merges]
        self.bpe_merges = bpe_merges[:num_merges]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.num_merges = num_merges
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        bpe_vocab = list(bytes_to_unicode().values())
        bpe_vocab = bpe_vocab + [(v + '</w>') for v in bpe_vocab]
        bpe_vocab.extend([''.join(merge_pair) for merge_pair in bpe_merges[:num_merges]])
        special_tokens = [bos_token, eos_token]
        bpe_vocab.extend(special_tokens)
        self.bpe_vocab = bpe_vocab
        self.encoder = {v: i for i, v in enumerate(bpe_vocab)}
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.cache = {tok: tok for tok in special_tokens}
        self.pat = re.compile("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+", re.IGNORECASE)

    @property
    def vocab_size(self) ->int:
        return len(self.encoder)

    def bpe(self, token: 'str') ->str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)
        if not pairs:
            return token + '</w>'
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: 'List[str]' = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text: 'str') ->List[int]:
        bpe_tokens: 'List[int]' = []
        text = text.lower().strip()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens: 'List[int]') ->str:
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace').replace('</w>', ' ')
        return text


class CLIPBPETransform(nn.Module):
    """
    nn.Module wrapper around CLIPBPETokenizer. Supports either a single string
    or list of strings for tokenization.

    Args:
        bpe_path (Optional[str]): path to the BPE file.
            Defaults to CLIP_DEFAULT_VOCAB_BPE_PATH
        bos_token (Optional[str]): beginning of sentence token.
            Defaults to "<|startoftext|>".
        eos_token (Optional[str]): end of sentence token.
            Defaults to "<|endoftext|>".
        num_merges (Optional[int]): number of merges.
            If None, it will load all merges from the BPE file.
    """

    def __init__(self, bpe_path: 'Optional[str]'=CLIP_DEFAULT_VOCAB_BPE_PATH, bos_token: 'Optional[str]'='<|startoftext|>', eos_token: 'Optional[str]'='<|endoftext|>', num_merges: 'Optional[int]'=None):
        super().__init__()
        self.bpe = CLIPBPETokenizer(bpe_path=bpe_path, bos_token=bos_token, eos_token=eos_token, num_merges=num_merges)

    def forward(self, text: 'Union[str, List[str]]') ->Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self.bpe.encode(text)
        else:
            return [self.bpe.encode(t) for t in text]


class CLIPTextTransform(nn.Module):
    """CLIP text transform
    CLIP BPE tokenizer transform, adds start and end tokens, then pads/truncates to text_max_length as necessary.
    This transform is torch scriptable.
    Args:
        text_max_length (int): Maximum length of text token sequences.
        text_start_token (str): Special start token passed to BPE tokenizer.
        text_end_token (str): Special end token passed to BPE tokenizer.
        text_pad_token (str): Special pad token to insert pad the sequence to text_max_length.
        text_bpe_merges_path (str): Location of BPE merges file for text transform.
        num_merges (int, optional): Number of merges to use from BPE merges file.
            Default: 48894 = 49152 (vocab size) - 256 (# bytes) - 2 (bos/eos tokens)

    Inputs:
        text (Union[List[str],str]): Text or batch of texts upon which to apply
            the transform.
    """

    def __init__(self, text_max_length: 'int'=77, text_start_token: 'str'='<|startoftext|>', text_end_token: 'str'='<|endoftext|>', text_pad_token: 'str'=None, text_bpe_merges_path: 'str'=CLIP_DEFAULT_VOCAB_BPE_PATH, num_merges: 'Optional[int]'=48894) ->None:
        super().__init__()
        local_merges_path = _PATH_MANAGER.get_local_path(text_bpe_merges_path)
        tokenizer = CLIPBPETransform(local_merges_path, text_start_token, text_end_token, num_merges)
        text_start_token = tokenizer([text_start_token])[0][0]
        text_end_token = tokenizer([text_end_token])[0][0]
        text_pad_token_id = 0
        if text_pad_token is not None:
            text_pad_token_id = tokenizer([text_pad_token])[0][0]
        text_max_length = text_max_length
        self.text_transform = nn.Sequential(*[tokenizer, text_transforms.Truncate(text_max_length - 2), text_transforms.AddToken(text_start_token, begin=True), text_transforms.AddToken(text_end_token, begin=False), text_transforms.ToTensor(padding_value=0), text_transforms.PadTransform(max_length=text_max_length, pad_value=text_pad_token_id)])

    def forward(self, text: 'Union[List[str], str]') ->Tensor:
        text_result = self.text_transform(text)
        assert torch.jit.isinstance(text_result, Tensor)
        return text_result


CLIP_DEFAULT_MEAN = 0.48145466, 0.4578275, 0.40821073


CLIP_DEFAULT_STD = 0.26862954, 0.26130258, 0.27577711


class CLIPImageTransform(nn.Module):
    """CLIP image transform
    random resized crop (train mode) or resize and center crop, followed by RGB conversion, tensor conversion, and normalization.

    Args:
        image_size (Union[int, Tuple[int]): desired output image size.
        image_interpolation (torchvision.transforms.InterpolationMode):
            Torchvision interpolation mode used during resizing. Defaults to bicubic.
        image_mean (Tuple[float]): mean of images, used for normalization.
        image_std (Tuple[float]): std of images, used for normalization.
        is_train (bool): Whether transform is run in train mode.

    Inputs:
        image (Union[List[Image], Image]): Image or batch of images upon which
            to apply the transform.
    """

    def __init__(self, image_size: 'Union[int, Tuple[int, int]]'=224, image_interpolation: 'InterpolationMode'=InterpolationMode.BICUBIC, image_mean: 'Tuple[float, float, float]'=CLIP_DEFAULT_MEAN, image_std: 'Tuple[float, float, float]'=CLIP_DEFAULT_STD, is_train: 'bool'=True) ->None:
        super().__init__()
        joint_transforms: 'List[Callable]' = [convert_to_rgb, image_transforms.ToTensor(), image_transforms.Normalize(image_mean, image_std)]
        base_transform: 'List[Callable]'
        if is_train:
            base_transform = [image_transforms.RandomResizedCrop(image_size, interpolation=image_interpolation)]
        else:
            base_transform = [image_transforms.Resize(image_size, interpolation=image_interpolation), image_transforms.CenterCrop(image_size)]
        self.image_transform = image_transforms.Compose(base_transform + joint_transforms)

    def forward(self, image: 'Union[List[Image], Image]') ->Tensor:
        if isinstance(image, Image):
            return self.image_transform(image)
        image_result = torch.stack([self.image_transform(x) for x in image])
        return image_result


class CLIPTransform(nn.Module):
    """Image and text transform for CLIP model.

    Image transform: either random resized crop (train mode) or resize and center
        crop, followed by RGB conversion, tensor conversion, and normalization.
    Text transform: applies CLIP's BPE tokenizer transform, adds start and end
        tokens, then pads/truncates to text_max_length as necessary.


    Args:
        image_size (Union[int, Tuple[int]): desired output image size.
        image_interpolation (torchvision.transforms.InterpolationMode):
            Torchvision interpolation mode used during resizing. Defaults to bicubic.
        image_mean (Tuple[float]): mean of images, used for normalization.
        image_std (Tuple[float]): std of images, used for normalization.
        text_max_length (int): Maximum length of text token sequences.
        is_train (bool): Whether transform is run in train mode.
        text_start_token (str): Special start token passed to BPE tokenizer.
        text_end_token (str): Special end token passed to BPE tokenizer.
        text_pad_token (str): Special pad token to insert pad the sequence to text_max_length.
        text_bpe_merges_path (str): Location of BPE merges file for text transform.
        num_merges (int, optional): Number of merges to use from BPE merges file.
            Default: 48894 = 49152 (vocab size) - 256 (# bytes) - 2 (bos/eos tokens)

    Inputs:
        image (Union[List[Image], Image]): Image or batch of images upon which
            to apply the transform.
        text (Union[List[str],str]): Text or batch of texts upon which to apply
            the transform.
    """

    def __init__(self, image_size: 'Union[int, Tuple[int, int]]'=224, image_interpolation: 'InterpolationMode'=InterpolationMode.BICUBIC, image_mean: 'Tuple[float, float, float]'=CLIP_DEFAULT_MEAN, image_std: 'Tuple[float, float, float]'=CLIP_DEFAULT_STD, text_max_length: 'int'=77, is_train: 'bool'=True, text_start_token: 'str'='<|startoftext|>', text_end_token: 'str'='<|endoftext|>', text_pad_token: 'str'=None, text_bpe_merges_path: 'str'=CLIP_DEFAULT_VOCAB_BPE_PATH, num_merges: 'Optional[int]'=48894) ->None:
        super().__init__()
        self.image_transform = CLIPImageTransform(image_size, image_interpolation, image_mean, image_std, is_train)
        self.text_transform = CLIPTextTransform(text_max_length, text_start_token, text_end_token, text_pad_token, text_bpe_merges_path, num_merges)

    def forward(self, image: 'Union[List[Image], Image]', text: 'Union[List[str], str]') ->Tuple[torch.Tensor, torch.Tensor]:
        return self.image_transform(image), self.text_transform(text)


def truncate(input: 'Any', max_seq_len: 'int') ->Any:
    """Truncate input sequence or batch

    :param input: Input sequence or batch to be truncated
    :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    :param max_seq_len: Maximum length beyond which input is discarded
    :type max_seq_len: int
    :return: Truncated sequence
    :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    """
    if torch.jit.isinstance(input, List[int]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[str]):
        return input[:max_seq_len]
    elif torch.jit.isinstance(input, List[List[int]]):
        output_int: 'List[List[int]]' = []
        for ids in input:
            output_int.append(ids[:max_seq_len])
        return output_int
    elif torch.jit.isinstance(input, List[List[str]]):
        output_str: 'List[List[str]]' = []
        for ids in input:
            output_str.append(ids[:max_seq_len])
        return output_str
    else:
        raise TypeError('Input type not supported')


class Truncate(nn.Module):
    """Truncate input sequence

    :param max_seq_len: The maximum allowable length for input sequence
    :type max_seq_len: int
    """

    def __init__(self, max_seq_len: 'int') ->None:
        super().__init__()
        self.max_seq_len = max_seq_len

    def forward(self, x: 'Any') ->Any:
        """
        :param input: Input sequence or batch of sequence to be truncated
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        :return: Truncated sequence
        :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return truncate(x, self.max_seq_len)


def add_token(input: 'Any', token_id: 'Any', begin: 'bool'=True) ->Any:
    """Add token to start or end of sequence

    :param input: Input sequence or batch
    :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    :param token_id: token to be added
    :type token_id: Union[str, int]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    :return: sequence or batch with token_id added to begin or end or input
    :rtype: Union[List[Union[str, int]], List[List[Union[str, int]]]]
    """
    if torch.jit.isinstance(input, List[int]) and torch.jit.isinstance(token_id, int):
        if begin:
            return [token_id] + input
        else:
            return input + [token_id]
    elif torch.jit.isinstance(input, List[str]) and torch.jit.isinstance(token_id, str):
        if begin:
            return [token_id] + input
        else:
            return input + [token_id]
    elif torch.jit.isinstance(input, List[List[int]]) and torch.jit.isinstance(token_id, int):
        output_int: 'List[List[int]]' = []
        if begin:
            for ids in input:
                output_int.append([token_id] + ids)
        else:
            for ids in input:
                output_int.append(ids + [token_id])
        return output_int
    elif torch.jit.isinstance(input, List[List[str]]) and torch.jit.isinstance(token_id, str):
        output_str: 'List[List[str]]' = []
        if begin:
            for ids in input:
                output_str.append([token_id] + ids)
        else:
            for ids in input:
                output_str.append(ids + [token_id])
        return output_str
    else:
        raise TypeError('Input type not supported')


class AddToken(nn.Module):
    """Add token to beginning or end of sequence

    :param token: The token to be added
    :type token: Union[int, str]
    :param begin: Whether to insert token at start or end or sequence, defaults to True
    :type begin: bool, optional
    """

    def __init__(self, token: 'Union[int, str]', begin: 'bool'=True) ->None:
        super().__init__()
        self.token = token
        self.begin = begin

    def forward(self, input: 'Any') ->Any:
        """
        :param input: Input sequence or batch
        :type input: Union[List[Union[str, int]], List[List[Union[str, int]]]]
        """
        return add_token(input, self.token, self.begin)


class PadTransform(nn.Module):
    """Pad tensor to a fixed length with given padding value.

    :param max_length: Maximum length to pad to
    :type max_length: int
    :param pad_value: Value to pad the tensor with
    :type pad_value: bool
    """

    def __init__(self, max_length: 'int', pad_value: 'int') ->None:
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: 'Tensor') ->Tensor:
        """
        :param x: The tensor to pad
        :type x: Tensor
        :return: Tensor padded up to max_length with pad_value
        :rtype: Tensor
        """
        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x


def to_tensor(input: 'Any', padding_value: 'Optional[int]'=None, dtype: 'torch.dtype'=torch.long) ->Tensor:
    """Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    :param input: Sequence or batch of token ids
    :type input: Union[List[int], List[List[int]]]
    :rtype: Tensor
    """
    if torch.jit.isinstance(input, List[int]):
        return torch.tensor(input, dtype=torch.long)
    elif torch.jit.isinstance(input, List[List[int]]):
        if padding_value is None:
            output = torch.tensor(input, dtype=dtype)
            return output
        else:
            output = pad_sequence([torch.tensor(ids, dtype=dtype) for ids in input], batch_first=True, padding_value=float(padding_value))
            return output
    else:
        raise TypeError('Input type not supported')


class ToTensor(nn.Module):
    """Convert input to torch tensor

    :param padding_value: Pad value to make each input in the batch of length equal to the longest sequence in the batch.
    :type padding_value: Optional[int]
    :param dtype: :class:`torch.dtype` of output tensor
    :type dtype: :class:`torch.dtype`
    """

    def __init__(self, padding_value: 'Optional[int]'=None, dtype: 'torch.dtype'=torch.long) ->None:
        super().__init__()
        self.padding_value = padding_value
        self.dtype = dtype

    def forward(self, input: 'Any') ->Tensor:
        """
        :param input: Sequence or batch of token ids
        :type input: Union[List[int], List[List[int]]]
        :rtype: Tensor
        """
        return to_tensor(input, padding_value=self.padding_value, dtype=self.dtype)

