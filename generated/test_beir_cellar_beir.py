
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


import logging


import math


import queue


import torch.multiprocessing as mp


from typing import List


from typing import Dict


import numpy as np


import re


from typing import Iterable


from torch import nn


from torch import Tensor


from typing import Union


from typing import Tuple


from torch.nn import functional as F


from typing import Mapping


from typing import Optional


from scipy.sparse import csr_matrix


from numpy import ndarray


from torch.utils.data import DataLoader


import time


from torch.optim import Optimizer


from typing import Callable


import random


from torch.utils.data import Dataset


class MarginMSELoss(nn.Module):
    """
    Computes the Margin MSE loss between the query, positive passage and negative passage. This loss
    is used to train dense-models using cross-architecture knowledge distillation setup. 

    Margin MSE Loss is defined as from (Eq.11) in Sebastian Hofstätter et al. in https://arxiv.org/abs/2010.02666:
    Loss(𝑄, 𝑃+, 𝑃−) = MSE(𝑀𝑠(𝑄, 𝑃+) − 𝑀𝑠(𝑄, 𝑃−), 𝑀𝑡(𝑄, 𝑃+) − 𝑀𝑡(𝑄, 𝑃−))
    where 𝑄: Query, 𝑃+: Relevant passage, 𝑃−: Non-relevant passage, 𝑀𝑠: Student model, 𝑀𝑡: Teacher model

    Remember: Pass the difference in scores of the passages as labels.
    """

    def __init__(self, model, scale: 'float'=1.0, similarity_fct='dot'):
        super(MarginMSELoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: 'Iterable[Dict[str, Tensor]]', labels: 'Tensor'):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]
        scores_pos = (embeddings_query * embeddings_pos).sum(dim=-1) * self.scale
        scores_neg = (embeddings_query * embeddings_neg).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg
        return self.loss_fct(margin_pred, labels)


class SpladeNaver(torch.nn.Module):

    def __init__(self, model_path):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_path)

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)['logits']
        return torch.max(torch.log(1 + torch.relu(out)) * kwargs['attention_mask'].unsqueeze(-1), dim=1).values

    def _text_length(self, text: 'Union[List[int], List[List[int]]]'):
        """helper function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """
        if isinstance(text, dict):
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):
            return 1
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            return sum([len(t) for t in text])

    def encode_sentence_bert(self, tokenizer, sentences: 'Union[str, List[str], List[int]]', batch_size: 'int'=32, show_progress_bar: 'bool'=None, output_value: 'str'='sentence_embedding', convert_to_numpy: 'bool'=True, convert_to_tensor: 'bool'=False, device: 'str'=None, normalize_embeddings: 'bool'=False, maxlen: 'int'=512, is_q: 'bool'=False) ->Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = True
        if convert_to_tensor:
            convert_to_numpy = False
        if output_value == 'token_embeddings':
            convert_to_tensor = False
            convert_to_numpy = False
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self
        all_embeddings = []
        length_sorted_idx = np.argsort([(-self._text_length(sen)) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc='Batches', disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = tokenizer(sentences_batch, add_special_tokens=True, padding='longest', truncation='only_first', max_length=maxlen, return_attention_mask=True, return_tensors='pt')
            features = batch_to_device(features, device)
            with torch.no_grad():
                out_features = self.forward(**features)
                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(token_emb[0:last_mask_id + 1])
                else:
                    embeddings = out_features
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

