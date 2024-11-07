import sys
_module = sys.modules[__name__]
del sys
retrieval_chatbot = _module
sft_summarizer = _module
reward_modeling = _module
diffuser_args = _module
diffuser_finetuner = _module
finetune_t2i = _module
t2i_dataset = _module
function_call_finetune = _module
conf = _module
benchmarking = _module
chatbot = _module
chatbot_gradio = _module
detail_memory = _module
dpo_train = _module
dpov2_train = _module
evaluation = _module
finetune = _module
finetune_multi_modal = _module
inference = _module
iterative_dpo_train = _module
merge_lora = _module
multistage_finetune = _module
raft_align = _module
reward_modeling = _module
rm_inference = _module
speculative_inference = _module
tool_inference = _module
vis_chatbot = _module
vis_chatbot_gradio = _module
vllm_inference = _module
train_diffusion_dpo = _module
train_diffusion_dpo_lisa = _module
test_instruct_pix2pix = _module
train_instruct_pix2pix_lisa = _module
train_lcm_distill_sd_wds_lisa = _module
train_lcm_distill_sd_wds_lora = _module
single_lisa = _module
train_text_to_image_lora = _module
convert_llama_weights_to_hf = _module
add_end_mark = _module
add_prompt = _module
concat = _module
concat_shuffle_split = _module
count = _module
merge = _module
raw2textonly = _module
sample = _module
shuffle = _module
export_llama_state_dict_checkpoint = _module
print_model_architecture = _module
app = _module
setup = _module
lmflow = _module
args = _module
datasets = _module
dataset = _module
multi_modal_dataset = _module
models = _module
auto_model = _module
base_model = _module
decoder_model = _module
encoder_decoder_model = _module
hf_decoder_model = _module
hf_encoder_decoder_model = _module
hf_model_mixin = _module
hf_text_regression_model = _module
interfaces = _module
tunable = _module
regression_model = _module
text_regression_model = _module
vision2seq_model = _module
vision_encoder = _module
clip_encoder = _module
optim = _module
adabelief = _module
adabound = _module
adadelta = _module
adagrad = _module
adam = _module
adamax = _module
adamp = _module
adamw_schedule_free = _module
adan = _module
dummy = _module
lamb = _module
lars = _module
nadam = _module
novograd = _module
optimizers = _module
radam = _module
sgd_schedule_free = _module
sgdp = _module
sophia = _module
yogi = _module
pipeline = _module
auto_pipeline = _module
base_aligner = _module
base_pipeline = _module
base_tuner = _module
dpo_aligner = _module
dpov2_aligner = _module
evaluator = _module
finetuner = _module
inferencer = _module
iterative_dpo_aligner = _module
raft_aligner = _module
rm_inferencer = _module
rm_tuner = _module
utils = _module
dpov2_dataprocessor = _module
dpov2_trainer = _module
memory_safe_dpov2_align = _module
memory_safe_vllm_inference = _module
peft_trainer = _module
raft_trainer = _module
rm_dataprocessor = _module
rm_trainer = _module
vllm_inferencer = _module
tokenization = _module
common = _module
constants = _module
conversation_template = _module
base = _module
chatglm = _module
chatml = _module
deepseek = _module
gemma = _module
internlm = _module
llama = _module
phi = _module
qwen = _module
yi = _module
zephyr = _module
data_utils = _module
flash_attention = _module
bloom_flash_attention = _module
gpt2_flash_attention = _module
gpt_neo_flash_attention = _module
llama_flash_attention = _module
triton_flash_attention = _module
llava_conversation_lib = _module
model = _module
multimodal = _module
position_interpolation = _module
llama_rope_scaled_monkey_patch = _module
versioning = _module
version = _module
tests = _module
test_dataset = _module
test_auto_model = _module
test_hf_decoder_model = _module
test_tool_inferencer = _module
test_auto_pipeline = _module
test_memory_safe_vllm_inferencer = _module
test_conversation_formatter = _module
test_conversation_template = _module
test_data_utils = _module
apply_delta = _module
convert_json_to_txt = _module
convert_minigpt4_checkpoints = _module
download_hf_file = _module
lm_evaluator = _module
make_delta = _module
merge_tokenizer = _module
preprocess_multimodal_data = _module
train_tokenizer = _module

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


import logging


from typing import Optional


from typing import List


import torch


from typing import Any


from typing import Dict


from typing import Union


import numpy as np


import torch.nn as nn


import copy


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torchvision import transforms


import warnings


import time


import math


import torch.utils.checkpoint


import functools


import itertools


import random


import torchvision.transforms.functional as TF


from torch.utils.data import default_collate


from typing import Tuple


from torch.nn import CrossEntropyLoss


from torch.optim.optimizer import Optimizer


import torch.optim


from torch import Tensor


from typing import Callable


from typing import Iterable


from torch import nn


from torch.optim import Optimizer


import torch.optim as optim


import torch.distributed as dist


from itertools import chain


from copy import deepcopy


from torch.nn.utils.rnn import pad_sequence


from typing import Literal


import inspect


import re


from collections.abc import Mapping


from typing import TYPE_CHECKING


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


from typing import TypedDict


from functools import partial


IGNORE_INDEX = -100


IMAGE_TOKEN_INDEX = -200


class CLIPVisionTower(nn.Module):

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.vision_select_layer
        self.select_feature = getattr(args, 'vision_select_feature', 'patch')
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def encode_images(self, images, language_projection):
        image_features = self(images)
        if language_projection is not None:
            image_features = language_projection(image_features)
        return image_features

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, language_projection=None, language_model=None, **kwargs):
        """
        Copy from the LLAVA code base.
        Should be polished.
        """
        vision_tower = self.vision_tower
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images, language_projection)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images, language_projection)
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_embeds = language_model.embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (0.0 * language_projection(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(language_model.embed_tokens(cur_input_ids[:image_token_start - 1]).detach())
                    cur_new_input_embeds.append(language_model.embed_tokens(cur_input_ids[image_token_start - 1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(language_model.embed_tokens(cur_input_ids[image_token_start + 1:image_token_start + 2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start + 1])
                        cur_labels = cur_labels[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids
                    cur_new_input_embeds.append(language_model.embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start + 1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(language_model.embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(language_model.embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, attention_mask, past_key_values, new_input_embeds, new_labels


class CondenseRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, pi_ratio, ntk_ratio, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.ntk_ratio = ntk_ratio
        max_position_embeddings *= ntk_ratio
        base = base * ntk_ratio ** (dim / (dim - 2))
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.pi_ratio = pi_ratio
        max_position_embeddings *= pi_ratio
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype) / pi_ratio
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype) / self.pi_ratio
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

