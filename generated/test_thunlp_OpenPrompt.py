
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


from collections import defaultdict


from collections import namedtuple


from typing import *


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data.dataset import Subset


from typing import Union


import random


from abc import ABC


from abc import abstractmethod


from collections import Counter


from typing import List


from typing import Dict


from typing import Callable


from typing import Sequence


from torch.utils.data import dataset


import copy


import torch


from torch.utils.data._utils.collate import default_collate


from torch.utils.data.sampler import RandomSampler


import torch.nn as nn


from torch.utils.data import DataLoader


import itertools


import warnings


from typing import Tuple


from typing import Optional


import torch.nn.functional as F


from functools import partial


import re


import string


from torch import nn


from torch.nn.parallel import DataParallel


from inspect import Parameter


from torch.nn.parameter import Parameter


from torch.nn.parallel.data_parallel import DataParallel


from torch.utils.data.dataset import Dataset


from math import ceil


import inspect


import time


from sklearn.model_selection import train_test_split


def signature(f):
    """Get the function f 's input arguments. A useful gadget
    when some function slot might be instantiated into multiple functions.
    
    Args:
        f (:obj:`function`) : the function to get the input arguments.
    
    Returns:
        namedtuple : of args, default, varargs, keywords, respectively.s

    """
    sig = inspect.signature(f)
    args = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    varargs = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_POSITIONAL]
    varargs = varargs[0] if varargs else None
    keywords = [p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD]
    keywords = keywords[0] if keywords else None
    defaults = [p.default for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is not p.empty] or None
    argspec = namedtuple('Signature', ['args', 'defaults', 'varargs', 'keywords'])
    return argspec(args, defaults, varargs, keywords)


class PromptModel(nn.Module):
    """``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``,
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``

    Args:
        plm (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    """

    def __init__(self, plm: 'PreTrainedModel', template: 'Template', freeze_plm: 'bool'=False, plm_eval_mode: 'bool'=False):
        super().__init__()
        self.plm = plm
        self.template = template
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False
        self.forward_keys = signature(self.plm.forward).args
        self._prepare_main_input_name()

    def _prepare_main_input_name(self):
        model = self.plm
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'main_input_name'):
            if model.encoder.main_input_name != model.main_input_name:
                main_input_name = model.encoder.main_input_name
            else:
                main_input_name = model.main_input_name
        else:
            main_input_name = getattr(model, 'main_input_name', 'input_ids')
        self.main_input_name = main_input_name

    def train(self, mode: 'bool'=True):
        if not isinstance(mode, bool):
            raise ValueError('training mode is expected to be boolean')
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch: 'Union[Dict, InputFeatures]') ->torch.Tensor:
        """
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.template.post_processing_outputs(outputs)
        return outputs

    def prepare_model_inputs(self, batch: 'Union[Dict, InputFeatures]') ->Dict:
        """Will be used in generation
        """
        batch = self.template.process_batch(batch)
        input_batch = {key: batch[key] for key in batch if key in self.forward_keys}
        return input_batch


class PromptForClassification(nn.Module):
    """``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a ``PromptModel``) into the
    logits of the labels, using a verbalizer.

    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the labels to label words for classification, e.g. ``ManualVerbalizer``.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    """

    def __init__(self, plm: 'PreTrainedModel', template: 'Template', verbalizer: 'Verbalizer', freeze_plm: 'bool'=False, plm_eval_mode: 'bool'=False):
        super().__init__()
        self.prompt_model = PromptModel(plm, template, freeze_plm, plm_eval_mode)
        self.verbalizer = verbalizer

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self):
        """Register the device parameter."""
        return self.plm.device

    def extract_at_mask(self, outputs: 'torch.Tensor', batch: 'Union[Dict, InputFeatures]'):
        """Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        outputs = outputs[torch.where(batch['loss_ids'] > 0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, batch: 'Union[Dict, InputFeatures]') ->torch.Tensor:
        """
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the label words (obtained by the current verbalizer).
        """
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, batch) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, batch)
        label_words_logits = self.verbalizer.process_outputs(outputs_at_mask, batch=batch)
        return label_words_logits

    def predict(self):
        pass

    def forward_without_verbalize(self, batch: 'Union[Dict, InputFeatures]') ->torch.Tensor:
        outputs = self.prompt_model(batch)
        outputs = self.verbalizer.gather_outputs(outputs)
        outputs_at_mask = self.extract_at_mask(outputs, batch)
        return outputs_at_mask

    @property
    def tokenizer(self):
        """Utility property, to get the tokenizer more easily.
        """
        return self.verbalizer.tokenizer

    def parallelize(self, device_map=None):
        """Parallelize the model across device
        """
        if hasattr(self.plm, 'parallelize'):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template
            self.verbalizer
        else:
            raise NotImplementedError('parallelize method was not implemented for this plm.')

    def deparallelize(self):
        """Deparallelize the model across device
        """
        if hasattr(self.plm, 'deparallelize'):
            self.plm.deparallelize()
            self.device_map = None
            self.template.cpu()
            self.verbalizer.cpu()
        else:
            raise NotImplementedError('parallelize method was not implemented for this plm.')


logger = logging.getLogger()


class InputFeatures(dict):
    """
    The class for input to the PLM and Prompts. To make users explicitly know the available keys,
    we define a dict with a set of predefined possible keys. The default value to any key is None.
    When use it as a dict, all the keys whose values are None are invisible.

    This class support most of the dict's operation (See Examples). It can also be consumed by
    pytorch's default_collate in DataLoader.
    Also a :py:meth:`to_tensor()` method is build to convert the values into torch.Tensor for torch's input.

    Examples:

    ..  code-block:: python

        in_feat = InputFeatures(**{'input_ids':[1,4,5], 'soft_token_ids': [3,4,5]})  # init from dict
        print(in_feat.keys())       # ['input_ids, 'soft_token_ids']
        in_feat['label'] = 3        # can assign value like normal dict
        print(in_feat.keys())       # ['input_ids','label', 'soft_token_ids'] (Note that it's also ordered)
        print(in_feat['label'])     # 3
        in_feat['alice'] = 0        # KeyError: Key alice not in predefined set of keys
        in_feat.values()            # [[1,4,5], 3, [3,4,5]]  (Note that it's also ordered)
        [in_feat[key] for key in in_feat]   # [[1,4,5], 3, [3,4,5]]
        new_dict= {**in_feat, 'new_key':2}  # new_dict is {'input_ids': [1, 4, 5], 'label': 3, 'soft_token_ids': [3, 4, 5], 'new_key': 2}

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """
    tensorable_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label', 'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids', 'past_key_values', 'loss_ids']
    all_keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'token_type_ids', 'label', 'decoder_input_ids', 'decoder_inputs_embeds', 'soft_token_ids', 'past_key_values', 'loss_ids', 'guid', 'tgt_text', 'encoded_tgt_text', 'input_ids_len']
    non_tensorable_keys = []

    def __init__(self, input_ids: 'Optional[Union[List, torch.Tensor]]'=None, inputs_embeds: 'Optional[torch.Tensor]'=None, attention_mask: 'Optional[Union[List[int], torch.Tensor]]'=None, token_type_ids: 'Optional[Union[List[int], torch.Tensor]]'=None, label: 'Optional[Union[int, torch.Tensor]]'=None, decoder_input_ids: 'Optional[Union[List, torch.Tensor]]'=None, decoder_inputs_embeds: 'Optional[torch.Tensor]'=None, soft_token_ids: 'Optional[Union[List, torch.Tensor]]'=None, past_key_values: 'Optional[torch.Tensor]'=None, loss_ids: 'Optional[Union[List, torch.Tensor]]'=None, guid: 'Optional[str]'=None, tgt_text: 'Optional[str]'=None, use_cache: 'Optional[bool]'=None, encoded_tgt_text: 'Optional[str]'=None, input_ids_len: 'Optional[int]'=None, **kwargs):
        self.input_ids = input_ids
        self.inputs_embeds = inputs_embeds
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids
        self.decoder_inputs_embeds = decoder_inputs_embeds
        self.soft_token_ids = soft_token_ids
        self.past_key_values = past_key_values
        self.loss_ids = loss_ids
        self.guid = guid
        self.tgt_text = tgt_text
        self.encoded_tgt_text = encoded_tgt_text
        self.use_cache = use_cache
        self.input_ids_len = input_ids_len
        for k in kwargs.keys():
            logger.warning('Your are passing an unexpected key words: {} to InputFeatures, might yield unexpected behaviours!'.format(k))
            setattr(self, k, kwargs[k])

    @classmethod
    def add_tensorable_keys(cls, *args):
        cls.tensorable_keys.extend(args)

    @classmethod
    def add_not_tensorable_keys(cls, *args):
        cls.not_tensorable_keys.extend(args)

    @classmethod
    def add_keys(cls, *args):
        cls.all_keys.extend(args)

    def __repr__(self):
        return str(self.to_json_string())

    def __len__(self):
        return len(self.keys())

    def to_tensor(self, device: 'str'='cuda'):
        """inplace operation, convert all tensorable features into :obj:`torch.tensor`"""
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, torch.tensor(value))
        return self

    def to(self, device: 'str'='cuda:0'):
        """move the tensor keys to runtime device, such as gpu:0
        """
        for key in self.tensorable_keys:
            value = getattr(self, key)
            if value is not None:
                setattr(self, key, value)
        return self

    def cuda(self, device: 'str'='cuda:0'):
        """mimic the tensor behavior
        """
        return self

    def to_json_string(self, keep_none=False):
        """Serializes this instance to a JSON string."""
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                data[key] = value.detach().cpu().tolist()
            elif value is None and keep_none:
                data[key] = None
            else:
                data[key] = value
        return json.dumps(data) + '\n'

    def keys(self, keep_none=False) ->List[str]:
        """get all keys of the InputFeatures

        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.

        Returns:
            :obj:`List[str]`: keys of the InputFeatures
        """
        if keep_none:
            return self.all_keys
        else:
            return [key for key in self.all_keys if getattr(self, key) is not None]

    def to_dict(self, keep_none=False) ->Dict[str, Any]:
        """get the dict of mapping from keys to values of the InputFeatures

        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.

        Returns:
            :obj:`Dict[str, Any]`: dict of mapping from keys to values of the InputFeatures
        """
        data = {}
        for key in self.all_keys:
            value = getattr(self, key)
            if value is not None:
                data[key] = value
            elif value is None and keep_none:
                data[key] = None
        return data

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.keys())

    def __setitem__(self, key, item):
        if key not in self.all_keys:
            raise KeyError('Key {} not in predefined set of keys'.format(key))
        setattr(self, key, item)

    def values(self, keep_none=False) ->List[Any]:
        """get the values with respect to the keys  of the InputFeatures

        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.

        Returns:
            :obj:`List[Any]`: the values with respect to the keys of the InputFeatures
        """
        return [getattr(self, key) for key in self.keys(keep_none=keep_none)]

    def __contains__(self, key, keep_none=False):
        return key in self.keys(keep_none)

    def items(self):
        """get the (key, value) pairs  of the InputFeatures

        Args:
            keep_none (:obj:`bool`, optional): whether to keep the predefined keys whose value is none. Defaults to False.

        Returns:
            :obj:`List[Any]`: the (key, value) pairs of the InputFeatures
        """
        return [(key, self.__getitem__(key)) for key in self.keys()]

    @staticmethod
    def collate_fct(batch: 'List'):
        """
        This function is used to collate the input_features.

        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        """
        elem = batch[0]
        return_dict = {}
        for key in elem:
            if key == 'encoded_tgt_text':
                return_dict[key] = [d[key] for d in batch]
            else:
                try:
                    return_dict[key] = default_collate([d[key] for d in batch])
                except:
                    None
        return InputFeatures(**return_dict)


class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always necessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """

    def __init__(self, guid=None, text_a='', text_b='', label=None, meta: 'Optional[Dict]'=None, tgt_text: 'Optional[Union[str, List[str]]]'=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + '\n'

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path: 'str') ->List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: "List['InputExample']", path: 'str') ->None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}


def convert_cfg_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            None
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict


class Template(nn.Module):
    """
    Base class for all the templates.
    Most of methods are abstract, with some exceptions to hold the common methods for all template, such as ``loss_ids``, ``save``, ``load``.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text.
    """
    registered_inputflag_names = ['loss_ids', 'shortenable_ids']

    def __init__(self, tokenizer: 'PreTrainedTokenizer', placeholder_mapping: 'dict'={'<text_a>': 'text_a', '<text_b>': 'text_b'}):
        super().__init__()
        self.tokenizer = tokenizer
        self.placeholder_mapping = placeholder_mapping
        self._in_on_text_set = False
        self.mixed_token_start = '{'
        self.mixed_token_end = '}'

    def get_default_loss_ids(self) ->List[int]:
        """Get the loss indices for the template using mask.
        e.g. when self.text is ``'{"placeholder": "text_a"}. {"meta": "word"} is {"mask"}.'``,
        output is ``[0, 0, 0, 0, 1, 0]``.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]:

            - 1 for a masked tokens.
            - 0 for a sequence tokens.
        """
        return [(1 if 'mask' in d else 0) for d in self.text]

    def get_default_shortenable_ids(self) ->List[int]:
        """Every template needs shortenable_ids, denoting which part of the template can be truncate to fit
        the language model's ``max_seq_length``. Default: the input text is shortenable, while the template text and other
        special tokens are not shortenable.

        e.g. when self.text is ``'{"placeholder": "text_a"} {"placeholder": "text_b", "shortenable": False} {"meta": "word"} is {"mask"}.'``,
        output is ``[1, 0, 0, 0, 0, 0, 0]``.

        Returns:
            :obj:`List[int]`: A list of integers in the range ``[0, 1]``:

            - 1 for the input tokens.
            - 0 for the template sequence tokens.
        """
        idx = []
        for d in self.text:
            if 'shortenable' in d:
                idx.append(1 if d['shortenable'] else 0)
            else:
                idx.append(1 if 'placeholder' in d else 0)
        return idx

    def get_default_soft_token_ids(self) ->List[int]:
        """
        This function identifies which tokens are soft tokens.

        Sometimes tokens in the template are not from the vocabulary,
        but a sequence of soft tokens.
        In this case, you need to implement this function

        Raises:
            NotImplementedError: if needed, add ``soft_token_ids`` into ``registered_inputflag_names`` attribute of Template class and implement this method.
        """
        raise NotImplementedError

    def incorporate_text_example(self, example: 'InputExample', text=None):
        if text is None:
            text = self.text.copy()
        else:
            text = text.copy()
        for i, d in enumerate(text):
            if not callable(d.get('post_processing')):
                d['post_processing'] = eval(d.get('post_processing', 'lambda x:x'))
            if 'placeholder' in d:
                text[i] = d['add_prefix_space'] + d.get('post_processing')(getattr(example, d['placeholder']))
            elif 'meta' in d:
                text[i] = d['add_prefix_space'] + d.get('post_processing')(example.meta[d['meta']])
            elif 'soft' in d:
                text[i] = d['soft']
            elif 'mask' in d:
                text[i] = '<mask>'
            elif 'special' in d:
                text[i] = d['special']
            elif 'text' in d:
                text[i] = d['add_prefix_space'] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        return text

    def _check_template_format(self):
        """check whether the template format is correct.
        TODO: add more
        """
        mask_num = 0
        for i, d in enumerate(self.text):
            if 'mask' in d:
                mask_num += 1
        if mask_num == 0:
            raise RuntimeError(f"'mask' position not found in the template: {self.text}. Please Check!")

    def parse_text(self, text: 'str') ->List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {'add_prefix_space': ' ' if i > 0 and text[i - 1] == ' ' else ''}
            while i < len(text) and text[i] == ' ':
                d['add_prefix_space'] = ' '
                i = i + 1
            if i == len(text):
                break
            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d['text'] = text[i:j].rstrip(' ')
                i = j
            else:
                j = i + 1
                mixed_token_cnt = 1
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        mixed_token_cnt -= 1
                        if mixed_token_cnt == 0:
                            break
                    elif text[j] == self.mixed_token_start:
                        mixed_token_cnt += 1
                    j = j + 1
                if j == len(text):
                    raise ValueError(f'mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}')
                dict_str = '{' + text[i + 1:j] + '}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    None
                    None
                    exit()
                i = j + 1
            parsed.append(d)
        return parsed

    def wrap_one_example(self, example: 'InputExample') ->List[Dict]:
        """Given an input example which contains input text, which can be referenced
        by self.template.placeholder_mapping 's value.
        This function process the example into a list of dict,
        Each dict functions as a group, which has the sample properties, such as
        whether it's shortenable, whether it's the masked position, whether it's soft token, etc.
        Since a text will be tokenized in the subsequent processing procedure,
        these attributes are broadcasted along the tokenized sentence.

        Args:
            example (:obj:`InputExample`): An :py:class:`~openprompt.data_utils.data_utils.InputExample` object, which should have attributes that are able to be filled in the template.

        Returns:
            :obj:`List[Dict]`: A list of dict of the same length as self.text. e.g. ``[{"loss_ids": 0, "text": "It was"}, {"loss_ids": 1, "text": "<mask>"}, ]``
        """
        if self.text is None:
            raise ValueError('template text has not been initialized')
        if isinstance(example, InputExample):
            text = self.incorporate_text_example(example)
            not_empty_keys = example.keys()
            for placeholder_token in self.placeholder_mapping:
                not_empty_keys.remove(self.placeholder_mapping[placeholder_token])
            not_empty_keys.remove('meta')
            keys, values = ['text'], [text]
            for inputflag_name in self.registered_inputflag_names:
                keys.append(inputflag_name)
                v = None
                if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:
                    v = getattr(self, inputflag_name)
                elif hasattr(self, 'get_default_' + inputflag_name):
                    v = getattr(self, 'get_default_' + inputflag_name)()
                    setattr(self, inputflag_name, v)
                else:
                    raise ValueError("""
                    Template's inputflag '{}' is registered but not initialize.
                    Try using template.{} = [...] to initialize
                    or create an method get_default_{}(self) in your template.
                    """.format(inputflag_name, inputflag_name, inputflag_name))
                if len(v) != len(text):
                    raise ValueError("Template: len({})={} doesn't match len(text)={}.".format(inputflag_name, len(v), len(text)))
                values.append(v)
            wrapped_parts_to_tokenize = []
            for piece in list(zip(*values)):
                wrapped_parts_to_tokenize.append(dict(zip(keys, piece)))
            wrapped_parts_not_tokenize = {key: getattr(example, key) for key in not_empty_keys}
            return [wrapped_parts_to_tokenize, wrapped_parts_not_tokenize]
        else:
            raise TypeError('InputExample')

    @abstractmethod
    def process_batch(self, batch):
        """Template should rewrite this method if you need to process the batch input such as substituting embeddings.
        """
        return batch

    def post_processing_outputs(self, outputs):
        """Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        return outputs

    def save(self, path: 'str', **kwargs) ->None:
        """
        A save method API.

        Args:
            path (str): A path to save your template.
        """
        raise NotImplementedError

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text
        if text is None:
            return
        if not self._in_on_text_set:
            self.safe_on_text_set()
        self._check_template_format()

    def safe_on_text_set(self) ->None:
        """With this wrapper function, setting text inside ``on_text_set()``
            will not trigger ``on_text_set()`` again to prevent endless recursion.
        """
        self._in_on_text_set = True
        self.on_text_set()
        self._in_on_text_set = False

    @abstractmethod
    def on_text_set(self):
        """
        A hook to do something when template text was set.
        The designer of the template should explicitly know what should be done when the template text is set.
        """
        raise NotImplementedError

    def from_file(self, path: 'str', choice: 'int'=0):
        """
        Read the template from a local file.

        Args:
            path (:obj:`str`): The path of the local template file.
            choice (:obj:`int`): The id-th line of the file.
        """
        with open(path, 'r') as fin:
            text = fin.readlines()[choice].rstrip()
            logger.info(f'using template: {text}')
        self.text = text
        return self

    @classmethod
    def from_config(cls, config: 'CfgNode', **kwargs):
        """load a template from template's configuration node.

        Args:
            config (:obj:`CfgNode`): the sub-configuration of template, i.e. config[config.template]
                        if config is a global config node.
            kwargs: Other kwargs that might be used in initialize the verbalizer.
                    The actual value should match the arguments of __init__ functions.
        """
        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs}
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        template = cls(**init_dict)
        if hasattr(template, 'from_file'):
            if not hasattr(config, 'file_path'):
                pass
            elif (not hasattr(config, 'text') or config.text is None) and config.file_path is not None:
                if config.choice is None:
                    config.choice = 0
                template.from_file(config.file_path, config.choice)
            elif (hasattr(config, 'text') and config.text is not None) and config.file_path is not None:
                raise RuntimeError("The text can't be both set from `text` and `file_path`.")
        return template


class Verbalizer(nn.Module):
    """
    Base class for all the verbalizers.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        classes (:obj:`Sequence[str]`): A sequence of classes that need to be projected.
    """

    def __init__(self, tokenizer: 'Optional[PreTrainedTokenizer]'=None, classes: 'Optional[Sequence[str]]'=None, num_classes: 'Optional[int]'=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.classes = classes
        if classes is not None and num_classes is not None:
            assert len(classes) == num_classes, 'len(classes) != num_classes, Check you config.'
            self.num_classes = num_classes
        elif num_classes is not None:
            self.num_classes = num_classes
        elif classes is not None:
            self.num_classes = len(classes)
        else:
            self.num_classes = None
        self._in_on_label_words_set = False

    @property
    def label_words(self):
        """
        Label words means the words in the vocabulary projected by the labels.
        E.g. if we want to establish a projection in sentiment classification: positive :math:`\\rightarrow` {`wonderful`, `good`},
        in this case, `wonderful` and `good` are label words.
        """
        if not hasattr(self, '_label_words'):
            raise RuntimeError("label words haven't been set.")
        return self._label_words

    @label_words.setter
    def label_words(self, label_words):
        if label_words is None:
            return
        self._label_words = self._match_label_words_to_label_ids(label_words)
        if not self._in_on_label_words_set:
            self.safe_on_label_words_set()

    def _match_label_words_to_label_ids(self, label_words):
        """
        sort label words dict of verbalizer to match the label order of the classes
        """
        if isinstance(label_words, dict):
            if self.classes is None:
                raise ValueError("""
                classes attribute of the Verbalizer should be set since your given label words is a dict.
                Since we will match the label word with respect to class A, to A's index in classes
                """)
            if set(label_words.keys()) != set(self.classes):
                raise ValueError('name of classes in verbalizer are different from those of dataset')
            label_words = [label_words[c] for c in self.classes]
        elif isinstance(label_words, list) or isinstance(label_words, tuple):
            pass
        else:
            raise ValueError('Verbalizer label words must be list, tuple or dict')
        return label_words

    def safe_on_label_words_set(self):
        self._in_on_label_words_set = True
        self.on_label_words_set()
        self._in_on_label_words_set = False

    def on_label_words_set(self):
        """A hook to do something when textual label words were set.
        """
        pass

    @property
    def vocab(self) ->Dict:
        if not hasattr(self, '_vocab'):
            self._vocab = self.tokenizer.convert_ids_to_tokens(np.arange(self.vocab_size).tolist())
        return self._vocab

    @property
    def vocab_size(self) ->int:
        return self.tokenizer.vocab_size

    @abstractmethod
    def generate_parameters(self, **kwargs) ->List:
        """
        The verbalizer can be seen as an extra layer on top of the original
        pre-trained models. In manual verbalizer, it is a fixed one-hot vector of dimension
        ``vocab_size``, with the position of the label word being 1 and 0 everywhere else.
        In other situation, the parameters may be a continuous vector over the
        vocab, with each dimension representing a weight of that token.
        Moreover, the parameters may be set to trainable to allow label words selection.

        Therefore, this function serves as an abstract methods for generating the parameters
        of the verbalizer, and must be instantiated in any derived class.

        Note that the parameters need to be registered as a part of pytorch's module to
        It can be achieved by wrapping a tensor using ``nn.Parameter()``.
        """
        raise NotImplementedError

    def register_calibrate_logits(self, logits: 'torch.Tensor'):
        """
        This function aims to register logits that need to be calibrated, and detach the original logits from the current graph.
        """
        if logits.requires_grad:
            logits = logits.detach()
        self._calibrate_logits = logits

    def process_outputs(self, outputs: 'torch.Tensor', batch: 'Union[Dict, InputFeatures]', **kwargs):
        """By default, the verbalizer will process the logits of the PLM's
        output.

        Args:
            logits (:obj:`torch.Tensor`): The current logits generated by pre-trained language models.
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of the data.
        """
        return self.process_logits(outputs, batch=batch, **kwargs)

    def gather_outputs(self, outputs: 'ModelOutput'):
        """ retrieve useful output for the verbalizer from the whole model output
        By default, it will only retrieve the logits

        Args:
            outputs (:obj:`ModelOutput`) The output from the pretrained language model.

        Return:
            :obj:`torch.Tensor` The gathered output, should be of shape (``batch_size``,
            ``seq_len``, ``any``)
        """
        return outputs.logits

    @staticmethod
    def aggregate(label_words_logits: 'torch.Tensor') ->torch.Tensor:
        """ To aggregate logits on multiple label words into the label's logits
        Basic aggregator: mean of each label words' logits to a label's logits
        Can be re-implemented in advanced verbaliezer.

        Args:
            label_words_logits (:obj:`torch.Tensor`): The logits of the label words only.

        Return:
            :obj:`torch.Tensor`: The final logits calculated by the label words.
        """
        if label_words_logits.dim() > 2:
            return label_words_logits.mean(dim=-1)
        else:
            return label_words_logits

    def normalize(self, logits: 'torch.Tensor') ->torch.Tensor:
        """
        Given logits regarding the entire vocab, calculate the probs over the label words set by softmax.

        Args:
            logits(:obj:`Tensor`): The logits of the entire vocab.

        Returns:
            :obj:`Tensor`: The probability distribution over the label words set.
        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    @abstractmethod
    def project(self, logits: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """This method receives input logits of shape ``[batch_size, vocab_size]``, and use the
        parameters of this verbalizer to project the logits over entire vocab into the
        logits of labels words.

        Args:
            logits (:obj:`Tensor`): The logits over entire vocab generated by the pre-trained language model with shape [``batch_size``, ``max_seq_length``, ``vocab_size``]

        Returns:
            :obj:`Tensor`: The normalized probs (sum to 1) of each label .
        """
        raise NotImplementedError

    def handle_multi_token(self, label_words_logits, mask):
        """
        Support multiple methods to handle the multi tokens produced by the tokenizer.
        We suggest using 'first' or 'max' if the some parts of the tokenization is not meaningful.
        Can broadcast to 3-d tensor.

        Args:
            label_words_logits (:obj:`torch.Tensor`):

        Returns:
            :obj:`torch.Tensor`
        """
        if self.multi_token_handler == 'first':
            label_words_logits = label_words_logits.select(dim=-1, index=0)
        elif self.multi_token_handler == 'max':
            label_words_logits = label_words_logits - 1000 * (1 - mask.unsqueeze(0))
            label_words_logits = label_words_logits.max(dim=-1).values
        elif self.multi_token_handler == 'mean':
            label_words_logits = (label_words_logits * mask.unsqueeze(0)).sum(dim=-1) / (mask.unsqueeze(0).sum(dim=-1) + 1e-15)
        else:
            raise ValueError('multi_token_handler {} not configured'.format(self.multi_token_handler))
        return label_words_logits

    @classmethod
    def from_config(cls, config: 'CfgNode', **kwargs):
        """load a verbalizer from verbalizer's configuration node.

        Args:
            config (:obj:`CfgNode`): the sub-configuration of verbalizer, i.e. ``config[config.verbalizer]``
                        if config is a global config node.
            kwargs: Other kwargs that might be used in initialize the verbalizer.
                    The actual value should match the arguments of ``__init__`` functions.
        """
        init_args = signature(cls.__init__).args
        _init_dict = {**convert_cfg_to_dict(config), **kwargs} if config is not None else kwargs
        init_dict = {key: _init_dict[key] for key in _init_dict if key in init_args}
        verbalizer = cls(**init_dict)
        if hasattr(verbalizer, 'from_file'):
            if not hasattr(config, 'file_path'):
                pass
            elif (not hasattr(config, 'label_words') or config.label_words is None) and config.file_path is not None:
                if config.choice is None:
                    config.choice = 0
                verbalizer.from_file(config.file_path, config.choice)
            elif (hasattr(config, 'label_words') and config.label_words is not None) and config.file_path is not None:
                raise RuntimeError("The text can't be both set from `text` and `file_path`.")
        return verbalizer

    def from_file(self, path: 'str', choice: 'Optional[int]'=0):
        """Load the predefined label words from verbalizer file.
        Currently support three types of file format:
        1. a .jsonl or .json file, in which is a single verbalizer
        in dict format.
        2. a .jsonal or .json file, in which is a list of verbalizers in dict format
        3.  a .txt or a .csv file, in which is the label words of a class are listed in line,
        separated by commas. Begin a new verbalizer by an empty line.
        This format is recommended when you don't know the name of each class.

        The details of verbalizer format can be seen in :ref:`How_to_write_a_verbalizer`.

        Args:
            path (:obj:`str`): The path of the local template file.
            choice (:obj:`int`): The choice of verbalizer in a file containing
                             multiple verbalizers.

        Returns:
            Template : `self` object
        """
        if path.endswith('.txt') or path.endswith('.csv'):
            with open(path, 'r') as f:
                lines = f.readlines()
                label_words_all = []
                label_words_single_group = []
                for line in lines:
                    line = line.strip().strip(' ')
                    if line == '':
                        if len(label_words_single_group) > 0:
                            label_words_all.append(label_words_single_group)
                        label_words_single_group = []
                    else:
                        label_words_single_group.append(line)
                if len(label_words_single_group) > 0:
                    label_words_all.append(label_words_single_group)
                if choice >= len(label_words_all):
                    raise RuntimeError('choice {} exceed the number of verbalizers {}'.format(choice, len(label_words_all)))
                label_words = label_words_all[choice]
                label_words = [label_words_per_label.strip().split(',') for label_words_per_label in label_words]
        elif path.endswith('.jsonl') or path.endswith('.json'):
            with open(path, 'r') as f:
                label_words_all = json.load(f)
                if isinstance(label_words_all, list):
                    if choice >= len(label_words_all):
                        raise RuntimeError('choice {} exceed the number of verbalizers {}'.format(choice, len(label_words_all)))
                    label_words = label_words_all[choice]
                elif isinstance(label_words_all, dict):
                    label_words = label_words_all
                    if choice > 0:
                        logger.warning('Choice of verbalizer is 1, but the file                          only contains one verbalizer.')
        self.label_words = label_words
        if self.num_classes is not None:
            num_classes = len(self.label_words)
            assert num_classes == self.num_classes, 'number of classes in the verbalizer file                                            does not match the predefined num_classes.'
        return self


class AutomaticVerbalizer(Verbalizer):
    """
    This implementation is slightly different from the original code in that
    1). we allow re-selecting the verbalizer after a fixed training steps.
    The original implementation only performs one step selection after getting
    the initial logits on the training data. To adopt their implementation,
    please only do ``optimize()`` after the first pass of training data.

    2). We strictly follows the probility calculation in Equation (3) in the
    paper, which take softmax over the logits.

    3). We do not implements the ``combine_patterns'' if-branch. Since it's
    not a pure verbalizer type, and doesn't yield much improvement. However,
    it can be achieve by using EnsembleTrainer to pass text wrapped by
    multiple templates together with this verbalizer.

    We use a probs_buffer to store the probability :math:`q_{P,t}(1|\\mathbf{x})` that to be used in later verbalizer selection,
    and a label_buffer to store the label :math:`y` that to be used in later verbalizer selection.

    Args:
        num_candidates (:obj:`int`, optional): the number of candidates for further selection based on Section 4.1
        label_word_num_per_class (:obj:`int`, optional): set to be greater than 1 to support Multi-Verbalizers in Section 4.2
        num_searches (:obj:`int`, optional): Maximnum number of label_words search. After reaching this number, the verbalizer will use the same label_words as the previous iterations.
        search_id (:obj:`int`, optional): the id of current search, used to determine when to stop label words searching.
        score_fct (:obj:`str`, optional): the scoring function of label words selection. ``llr`` means log likelihood ratio, corresponding to Equation (7); ``ce`` means cross entropy, corresponding to Equation (6). As the paper points out, ``llr'' is significantly better than 'ce', we only keep it to match the original code.
        balance (:obj:`book`, optional): whether to perform normalization of unbalanced training dataset, as Equation (5).
    """

    def __init__(self, tokenizer: 'PreTrainedTokenizer'=None, num_candidates: 'Optional[int]'=1000, label_word_num_per_class: 'Optional[int]'=1, num_searches: 'Optional[int]'=1, score_fct: 'Optional[str]'='llr', balance: 'Optional[bool]'=True, num_classes: 'Optional[bool]'=None, classes: 'Optional[List[str]]'=None, init_using_split: 'Optional[str]'='train', **kwargs):
        super().__init__(num_classes=num_classes, tokenizer=tokenizer, classes=classes)
        self.num_candidates = num_candidates
        self.label_word_num_per_class = label_word_num_per_class
        self.probs_buffer, self.labels_buffer = None, None
        assert num_searches > 0, 'You requires the verbalizer to perform {} searches. Invalid.'.format(num_searches)
        self.num_searches = num_searches
        self.search_id = 0
        self.accumulate_step = 0
        self.accumulate = True
        self.score_fct = score_fct
        self.balance = balance
        self.init_using_split = init_using_split

    def register_buffer(self, logits, labels):
        """

        Args:
            logits (:obj:`torch.Tensor`):
            labels (:obj:`List`):
        """
        logits = F.softmax(logits.detach(), dim=-1)
        labels = labels.detach()
        if self.probs_buffer is None:
            self.probs_buffer = logits
            self.labels_buffer = labels
        else:
            self.probs_buffer = torch.vstack([self.probs_buffer, logits])
            self.labels_buffer = torch.hstack([self.labels_buffer, labels])

    def process_logits(self, logits: 'torch.Tensor', **kwargs):
        if self.accumulate:
            self.accumulate_step += 1
            self.register_buffer(logits, kwargs['batch']['label'])
        if hasattr(self, 'label_words_ids'):
            label_words_logits = self.project(logits, **kwargs)
            label_words_probs = self.normalize(label_words_logits)
            if hasattr(self, '_calibrate_logits') and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)
            label_words_logits = torch.log(label_words_probs + 1e-15)
            if label_words_logits.dim() > 2:
                label_logits = self.aggregate(label_words_logits)
            else:
                label_logits = label_words_logits
            return label_logits
        else:
            return torch.randn((logits.size(0), self.num_classes), requires_grad=True)

    def project(self, logits: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """When this verbalizer hasn't perform optimize(), it has no
        ``label_words_ids``, thus will give random predictions, and should
        have no connection to the model to give (miss-leading) grads.

        Args:
            logits (:obj:`torch.Tensor`): The original logits over the vocabulary.

        Returns:
            :obj:`torch.Tensor`: The projected logits of label words.
        """
        label_words_logits = logits[:, self.label_words_ids]
        return label_words_logits

    def optimize(self):
        pass

    def optimize_to_initialize(self):
        """This is an epoch-level optimize. If used in batch-level like an ordinary
        gradient descend optimizer, the result may not be very satisfying since the accumated
        examples (i.e., the probs_buffer and the labels_buffer) are not enough if the batchsize
        is small.
        """
        if self.search_id < self.num_searches:
            self.label_words_ids = self._find_verbalizer(words_per_label=self.label_word_num_per_class, num_candidates=self.num_candidates, score_fct=self.score_fct, balance=self.balance)
            self.probs_buffer, self.labels_buffer = None, None
            self.search_id += 1
            if self.search_id == self.num_searches:
                self.accumulate = False
        else:
            logger.info("Verbalizer's max num_searches reached, use the previous label words.")
        self._show_verbalizer()

    def _show_verbalizer(self):
        tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in self.label_words_ids]
        logger.info('Verbalizer is {}'.format(tokens))

    def _find_verbalizer(self, words_per_label: 'int'=1, num_candidates: 'int'=1000, balance: 'bool'=True, score_fct: 'str'='llr'):
        logger.info('Finding verbalizer ...')
        probs = self.probs_buffer
        labels = self.labels_buffer
        candidates = self._get_candidates(num_candidates=num_candidates, probs=probs, labels=labels)
        label_words = self._get_top_words(probs=probs, candidates=candidates, balance=balance, words_per_label=words_per_label, score_fct=score_fct)
        return label_words

    def _get_candidates(self, num_candidates: 'int', probs: 'torch.Tensor', labels: 'torch.Tensor') ->Dict[str, List[str]]:
        if num_candidates <= 0:
            return [torch.arange(self.vocab_size) for label_id in range(self.num_classes)]
        log_probs = torch.log(probs + 1e-15)
        candidate_ids = []
        for label_id in range(self.num_classes):
            label_mask = (labels == label_id).unsqueeze(-1)
            score = torch.sum(log_probs * label_mask, dim=0)
            candidate_id = torch.argsort(score, descending=True)[:num_candidates]
            candidate_ids.append(candidate_id)
        return candidate_ids

    def _get_top_words(self, probs: 'torch.Tensor', candidates: 'List[torch.Tensor]', balance: 'bool'=True, words_per_label: 'int'=10, score_fct: 'Optional[str]'='llr'):
        label_words_ids = []
        for label_id in range(self.num_classes):
            label_mask = self.labels_buffer == label_id
            probs_per_label = probs[:, candidates[label_id]]
            if score_fct == 'llr':
                s = self._log_likelihood_ratio(probs_per_label, label_mask, balance)
            elif score_fct == 'ce':
                s = self._cross_entropy(probs_per_label, label_mask, balance)
            else:
                raise ValueError(f"Score function '{score_fct}' not implemented")
            sorted_ids = torch.argsort(s, descending=True)[:words_per_label]
            selected_ids = candidates[label_id][sorted_ids]
            label_words_ids.append(selected_ids)
        label_words_ids = torch.vstack(label_words_ids)
        return label_words_ids

    def _log_likelihood_ratio(self, probs, label_mask, balance):
        if balance:
            scale_factor = torch.sum(label_mask) / torch.sum(1 - label_mask) * (1 - label_mask).unsqueeze(-1)
        else:
            scale_factor = (1 - label_mask).unsqueeze(-1)
        label_mask = label_mask.unsqueeze(-1)
        pos_score = torch.sum(torch.log(probs + 1e-15) * label_mask, dim=0) - torch.sum(torch.log(1 - probs + 1e-15) * label_mask, dim=0)
        neg_score = torch.sum(torch.log(1 - probs + 1e-15) * scale_factor, dim=0) - torch.sum(torch.log(probs + 1e-15) * scale_factor, dim=0)
        return pos_score + neg_score

    def _cross_entropy(self, probs, label_mask, balance):
        if balance:
            scale_factor = torch.sum(label_mask) / torch.sum(1 - label_mask) * (1 - label_mask).unsqueeze(-1)
        else:
            scale_factor = (1 - label_mask).unsqueeze(-1)
        label_mask = label_mask.unsqueeze(-1)
        pos_score = torch.sum(torch.log(probs + 1e-15) * label_mask, dim=0)
        neg_score = torch.sum(torch.log(1 - probs + 1e-15) * scale_factor, dim=0)
        return pos_score + neg_score

    def from_file(self, path: 'str', choice: 'Optional[int]'=0):
        raise NotImplementedError("This verbalizer is learned and can't be set from file.")


class GenerationVerbalizer(Verbalizer):
    """
    This verbalizer is useful when the label prediction is better defined by a piece of input.
    For example, in correference resolution, the tgt_text is a proper noun mentioned in the text.
    There is no fixed mapping between a class label and its label words. This verbalizer
    can be used as verbalizer of ``COPA`` and ``WSC`` datasets in SuperGlue.

    This verbalizer is especially powerful when combined with
    `All NLP Tasks Are Generation Tasks <https://arxiv.org/abs/2103.10360>`_ Paradigm (Also see
    `Crossfit <https://arxiv.org/abs/2104.08835>`_). It can make any piece of text the tgt_text. The tgt_text will then be filled in the `{"mask"}`.

    For example, when label word is ``"good"``, the tgt_text is ``"good"``;

    when label word is ``{"text":"good"}``, the tgt_text is also ``"good"``;

    when label word is ``{"meta":"choice1"}``, the tgt_text is the ``"meta['choice1']"`` field of the ``InputExample``;

    when label word is ``{"meta":"choice1"} {"placeholder", "text_a"} .``, the tgt_text is the ``"meta['choice1']"`` field of the ``InputExample``,
    followed by ``text_a`` field of the ``InputExample``, and then a ``'.'``;

    A use case can be seen in `Tutorial 4.1 <https://github.com/thunlp/OpenPrompt/blob/main/tutorial/4.1_all_tasks_are_generation.py>`_

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        is_rule (:obj:`bool`, optional): When the verbalizer use the rule syntax of MixTemplate.
        label_words (:obj:`dict`, optional): The label words of the generation verbalizer

    Example:
    To use this verbalizer to train the T5 model to predict answer and explanation using two masks.

    When the template (Defined by :obj:`MixedTemplate`) is:
    >>> input_example = InputExample(text_a = "Can fish run?", meta={"answer":"no", "explanation": "The fish have no legs"}, label=0)
    >>> template = "{'placeholder':'text_a'} answer: {'mask'} explanation: {'mask'}"

    The verbalizer can be:
    >>> label_words = {0:["no", "{'meta':'explanation'}"], 1:["yes", "{'meta':'explanation'}"]}
    >>> verbalizer = GenerationVerbalizer(tokenizer, classes=None, is_rule=True, label_words=label_words)




    """

    def __init__(self, tokenizer: 'PreTrainedTokenizer', classes: 'Optional[List[str]]'=None, num_classes: 'Optional[int]'=None, is_rule: 'Optional[bool]'=False, label_words: 'Optional[dict]'=None):
        if classes is None and label_words is not None:
            classes = list(label_words.keys())
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = ''
        self.is_rule = is_rule
        self.mixed_token_start = '{'
        self.mixed_token_end = '}'
        if label_words is not None:
            self.label_words = label_words

    def wrap_one_example(self, example: 'InputExample') ->List[Dict]:
        """Take an InputExample, and fill the tgt_text with label words
        """
        if not isinstance(self.label_words[example.label], list):
            label_word = [self.label_words[example.label]]
        else:
            label_word = self.label_words[example.label]
        if example.tgt_text is not None:
            logger.warning(f'The example already has tgt_text {example.tgt_text}, and will be filled with new label words, is this intended?')
        if not self.is_rule:
            instance_label_word = label_word
        else:
            instance_label_word = [i(example) for i in label_word]
        if len(instance_label_word) == 1:
            example.tgt_text = instance_label_word[0]
        else:
            example.tgt_text = instance_label_word
        return example

    def on_label_words_set(self):
        """
        Process the text into the label words (sometimes a function) according to the syntax of MixedTemplate
        """
        if isinstance(self.label_words[0], list):
            self.label_words = [x[0] for x in self.label_words]
        if self.is_rule:
            for id, label_word in enumerate(self.label_words):
                try:
                    d = self.parse_text(label_word)
                except:
                    raise RuntimeError(f"is_rule={self.is_rule} but label_word: {label_word} can't be converted to object.")
                self.label_words[id] = partial(lambda x, text: self.incorporate_text_example(text, x), text=d)

    def parse_text(self, text: 'str') ->List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {'add_prefix_space': ' ' if i > 0 and text[i - 1] == ' ' else ''}
            while i < len(text) and text[i] == ' ':
                d['add_prefix_space'] = ''
                i = i + 1
            if i == len(text):
                break
            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d['text'] = text[i:j].rstrip(' ')
                i = j
            else:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        break
                    j = j + 1
                if j == len(text):
                    raise ValueError(f'mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}')
                dict_str = '{' + text[i + 1:j] + '}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    None
                    None
                    exit()
                i = j + 1
            parsed.append(d)
        return parsed

    def incorporate_text_example(self, text, example: 'InputExample'):
        text = text.copy()
        for i, d in enumerate(text):
            if 'placeholder' in d:
                text[i] = d['add_prefix_space'] + d.get('post_processing', lambda x: x)(getattr(example, d['placeholder']))
            elif 'meta' in d:
                text[i] = d['add_prefix_space'] + d.get('post_processing', lambda x: x)(example.meta[d['meta']])
            elif 'soft' in d:
                raise RuntimeError('soft token not supported in verbalizer')
            elif 'mask' in d:
                raise RuntimeError('mask token not supported in verbalizer')
            elif 'special' in d:
                raise RuntimeError('special token not supported in verbalizer')
            elif 'text' in d:
                text[i] = d['add_prefix_space'] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        text = ' '.join(text)
        return text


class ManualVerbalizer(Verbalizer):
    """
    The basic manually defined verbalizer class, this class is inherited from the :obj:`Verbalizer` class.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """

    def __init__(self, tokenizer: 'PreTrainedTokenizer', classes: 'Optional[List]'=None, num_classes: 'Optional[Sequence[str]]'=None, label_words: 'Optional[Union[Sequence[str], Mapping[str, str]]]'=None, prefix: 'Optional[str]'=' ', multi_token_handler: 'Optional[str]'='first', post_log_softmax: 'Optional[bool]'=True):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        """Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]
        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith('<!>'):
                    new_label_words_per_label.append(word.split('<!>')[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) ->List:
        """In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)
        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [([([1] * len(ids) + [0] * (max_len - len(ids))) for ids in ids_per_label] + [[0] * max_len] * (max_num_label_words - len(ids_per_label))) for ids_per_label in all_ids]
        words_ids = [([(ids + [0] * (max_len - len(ids))) for ids in ids_per_label] + [[0] * max_len] * (max_num_label_words - len(ids_per_label))) for ids_per_label in all_ids]
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def project(self, logits: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The original logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """
        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000 * (1 - self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: 'torch.Tensor', **kwargs):
        """A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        label_words_logits = self.project(logits, **kwargs)
        if self.post_log_softmax:
            label_words_probs = self.normalize(label_words_logits)
            if hasattr(self, '_calibrate_logits') and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)
            label_words_logits = torch.log(label_words_probs + 1e-15)
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits: 'torch.Tensor') ->torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def aggregate(self, label_words_logits: 'torch.Tensor') ->torch.Tensor:
        """Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1) / self.label_words_mask.sum(-1)
        return label_words_logits

    def calibrate(self, label_words_probs: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() == 1, 'self._calibrate_logits are not 1-d tensor'
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] and calibrate_label_words_probs.shape[0] == 1, 'shape not match'
        label_words_probs /= calibrate_label_words_probs + 1e-15
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1, keepdim=True)
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs


class MixedTemplate(Template):
    """The Mixed Template class defined by a string of `text`. See more examples in the `tutorial <https://github.com/thunlp/OpenPrompt/blob/ca27491101df0108a8dd753e5b1e79bf591f65d3/tutorial/1.1_mixed_template.py>`_.

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
    """
    registered_inputflag_names = ['soft_token_ids', 'loss_ids', 'shortenable_ids']

    def __init__(self, model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizer', text: 'Optional[str]'=None, placeholder_mapping: 'dict'={'<text_a>': 'text_a', '<text_b>': 'text_b'}):
        super().__init__(tokenizer=tokenizer, placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.text = text

    def get_default_soft_token_ids(self) ->List[int]:
        return self.soft_token_ids

    def prepare(self):
        """get the soft token indices ( soft_token_ids ) for the template

        ``"soft_id"`` can be used to reference the previous soft token, which means these tokens use the same embeddings.
        **Note that ``"soft_id"`` should have index start from 1 but not 0**

        e.g. when self.text is ``'{"soft": None} {"soft": "the", "soft_id": 1} {"soft": None} {"soft": "it", "soft_id": 3} {"soft_id": 1} {"soft": "was"} {"mask"}'``,
        output is [1, 2, 3, 4, 2, 5, 0]
        """
        num_soft_token = 0
        text = []
        soft_token_ids = []
        idx_mp = {}
        emb_mp = {}
        for d in self.text:
            if 'soft' not in d and 'soft_id' not in d:
                text.append(d)
                soft_token_ids.append(0)
                continue
            old_num = num_soft_token
            if 'soft_id' in d:
                if not isinstance(d['soft_id'], int) or d['soft_id'] <= 0:
                    raise ValueError(f"soft_id should be integer greater than zero, but get {d['soft_id']}")
                if d['soft_id'] in idx_mp:
                    id_list = idx_mp[d['soft_id']]
                    text.extend([{'soft': None} for _ in range(len(id_list))])
                    soft_token_ids.extend(id_list)
                    continue
                elif 'soft' not in d:
                    d['soft'] = None
            if d['soft'] is None:
                if 'duplicate' in d:
                    if 'same' in d and d['same']:
                        num_soft_token += 1
                        id_list = [num_soft_token for _ in range(d['duplicate'])]
                    else:
                        num_soft_token += d['duplicate']
                        id_list = list(range(old_num + 1, num_soft_token + 1))
                else:
                    num_soft_token += 1
                    id_list = [num_soft_token]
                text.extend([{'soft': ''} for _ in range(len(id_list))])
            else:
                token_ids = self.tokenizer(d['add_prefix_space'] + d['soft'], add_special_tokens=False)['input_ids']
                surface_forms = self.tokenizer.convert_ids_to_tokens(token_ids)
                assert len(token_ids) == len(surface_forms)
                num_soft_token += len(token_ids)
                id_list = list(range(old_num + 1, num_soft_token + 1))
                for idx, soft_id in enumerate(id_list):
                    emb_mp[soft_id] = token_ids[idx]
                text.extend([{'soft': surface_form} for surface_form in surface_forms])
            soft_token_ids.extend(id_list)
            if 'soft_id' in d:
                idx_mp[d['soft_id']] = id_list
        self.num_soft_token = num_soft_token
        self.text = text
        self.soft_token_ids = soft_token_ids
        self.soft_embedding = nn.Embedding(1 + self.num_soft_token, self.embedding_size)
        for soft_id, token_id in emb_mp.items():
            self.soft_embedding.weight.data[soft_id, :] = self.raw_embedding.weight.data[token_id, :].clone().detach().requires_grad_(True)

    def parse_text(self, text: 'str') ->List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {'add_prefix_space': ' ' if i > 0 and text[i - 1] == ' ' else ''}
            while i < len(text) and text[i] == ' ':
                d['add_prefix_space'] = ' '
                i = i + 1
            if i == len(text):
                break
            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d['text'] = text[i:j].rstrip(' ')
                i = j
            else:
                j = i + 1
                mixed_token_cnt = 1
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        mixed_token_cnt -= 1
                        if mixed_token_cnt == 0:
                            break
                    elif text[j] == self.mixed_token_start:
                        mixed_token_cnt += 1
                    j = j + 1
                if j == len(text):
                    raise ValueError(f'mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}')
                dict_str = '{' + text[i + 1:j] + '}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    None
                    None
                    exit()
                i = j + 1
            parsed.append(d)
        return parsed

    def on_text_set(self):
        """
        when template text was set

        1. parse text

        2. generate parameter needed
        """
        self.text = self.parse_text(self.text)
        self.prepare()

    def process_batch(self, batch: 'Union[Dict, InputFeatures]') ->Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        raw_embeds = self.raw_embedding(batch['input_ids'])
        soft_embeds = self.soft_embedding(batch['soft_token_ids'])
        inputs_embeds = torch.where((batch['soft_token_ids'] > 0).unsqueeze(-1), soft_embeds, raw_embeds)
        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch


class One2oneVerbalizer(Verbalizer):
    """
    The basic manually defined verbalizer class, this class is inherited from the :obj:`Verbalizer` class.
    This class restrict the use of label words to one words per label. For a verbalzer with less constraints,
    please use Basic ManualVerbalizer.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`classes`): The classes (or labels) of the current task.
        num_classes (:obj:`int`): Optional. The number of classes of the verbalizer. Only one of `classes` and `num_classes` should be used.
        label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer. (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """

    def __init__(self, tokenizer: 'PreTrainedTokenizer', num_classes: 'Optional[int]'=None, classes: 'Optional[List]'=None, label_words: 'Optional[Union[Sequence[str], Mapping[str, str]]]'=None, prefix: 'Optional[str]'=' ', multi_token_handler: 'Optional[str]'='first', post_log_softmax: 'Optional[bool]'=True):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        """Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], list):
            assert max([len(w) for w in label_words]) == 1, 'Providing multiple label words, you should use other verbalizers instead.'
            label_words = [w[0] for w in label_words]
        for word in label_words:
            if word.startswith('<!>'):
                new_label_words.append(word.split('<!>')[1])
            else:
                new_label_words.append(prefix + word)
        return new_label_words

    def generate_parameters(self) ->List:
        """In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        words_ids = []
        for word in self.label_words:
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) > 1:
                logger.warning('Word {} is split into multiple tokens: {}.                     If this is not what you expect, try using another word for this verbalizer'.format(word, self.tokenizer.convert_ids_to_tokens(word_ids)))
            words_ids.append(word_ids)
        max_len = max([len(ids) for ids in words_ids])
        words_ids_mask = [([1] * len(ids) + [0] * (max_len - len(ids))) for ids in words_ids]
        words_ids = [(ids + [0] * (max_len - len(ids))) for ids in words_ids]
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.label_words_mask = nn.Parameter(words_ids_mask, requires_grad=False)

    def project(self, logits: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The original logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """
        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: 'torch.Tensor', **kwargs):
        """A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the label words set.
        """
        label_words_logits = self.project(logits, **kwargs)
        if self.post_log_softmax:
            label_words_probs = self.normalize(label_words_logits)
            if hasattr(self, '_calibrate_logits') and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)
            label_words_logits = torch.log(label_words_probs + 1e-15)
        return label_words_logits

    def normalize(self, logits: 'torch.Tensor') ->torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def calibrate(self, label_words_probs: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() == 1, 'self._calibrate_logits are not 1-d tensor'
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] and calibrate_label_words_probs.shape[0] == 1, 'shape not match'
        label_words_probs /= calibrate_label_words_probs + 1e-15
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1, keepdim=True)
        label_words_probs /= norm
        return label_words_probs


class PrefixTuningTemplate(Template):
    """This is the implementation which support T5 and other Encoder-Decoder model,
    as soon as their blocks allows the ``past_key_values`` to be injected to the model.
    This implementation modifies the huggingface's T5 forward without touching the code-base.
    However, it may fail to work when used in DataParallel model. Please use it using
    single gpu or model-parallel training.

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained model.
        plm_config (:obj:`PretrainedConfig`): The configuration of the current pre-trained model.
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model.
        mapping_hook (:obj:`nn.Module`, optional):
        text (:obj:`str`, optional):
        num_token (:obj:`int`, optional):
        placeholder_mapping (:obj:`dict`):
        prefix_dropout (:obj:`float`, optional): The dropout rate for the prefix sequence.
    """
    registered_inputflag_names = ['loss_ids', 'shortenable_ids']

    def __init__(self, model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizer', mapping_hook: 'Optional[nn.Module]'=None, text: 'Optional[str]'=None, num_token: 'Optional[int]'=5, placeholder_mapping: 'dict'={'<text_a>': 'text_a', '<text_b>': 'text_b'}, prefix_dropout: 'Optional[float]'=0.0, mid_dim: 'Optional[int]'=512, using_encoder_past_key_values: 'Optional[bool]'=True, using_decoder_past_key_values: 'Optional[bool]'=True):
        super().__init__(tokenizer=tokenizer, placeholder_mapping=placeholder_mapping)
        raw_embedding = model.get_input_embeddings()
        self.config = model.config
        self.mapping_hook = mapping_hook
        self.embedding_size = raw_embedding.weight.shape[-1]
        self.num_token = num_token
        self.using_encoder_past_key_values = using_encoder_past_key_values
        self.using_decoder_past_key_values = using_decoder_past_key_values
        assert self.using_encoder_past_key_values or self.using_decoder_past_key_values, "Can't be both False."
        if not self.config.is_encoder_decoder and not self.using_decoder_past_key_values:
            logger.warning('Ignore using_decoder_past_key_values=False in a decoder-only LM.')
        if isinstance(self.config, T5Config):
            self.n_layer = self.config.num_layers
            self.n_embd = self.config.d_model
            self.n_head = self.config.num_heads
            self.n_decoder_layer = self.config.num_decoder_layers
            self.match_n_decoder_layer = self.n_decoder_layer
            self.match_n_layer = self.n_layer
        elif isinstance(self.config, GPT2Config):
            self.n_decoder_layer = self.config.n_layer
            self.n_embd = self.config.n_embd
            self.n_head = self.config.n_head
            self.match_n_decoder_layer = self.n_decoder_layer
        self.mid_dim = mid_dim
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head
        self.prefix_dropout = prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)
        self.default_text1 = '{"placeholder": "text_a"} {"mask"}'
        self.default_text2 = '{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"}'
        self.text = text
        self.generate_parameters()
        self.plm_modified = False
        self.modify_plm(model)

    def on_text_set(self):
        self.text = self.parse_text(self.text)
        self.generate_parameters()

    def get_past_key_values(self, batch_size=1):
        pvs = []
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:
            input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            temp_control = self.wte(input_tokens)
            past_key_values = self.control_trans(temp_control)
            _, seqlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            pvs.append(past_key_values)
        else:
            pvs.append(None)
        if not self.config.is_encoder_decoder or self.using_decoder_past_key_values:
            decoder_input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            decoder_temp_control = self.decoder_wte(decoder_input_tokens)
            decoder_past_key_values = self.decoder_control_trans(decoder_temp_control)
            _, decoder_seqlen, _ = decoder_past_key_values.shape
            decoder_past_key_values = decoder_past_key_values.view(batch_size, decoder_seqlen, self.match_n_decoder_layer * 2, self.match_n_head, self.match_n_embd)
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            pvs.append(decoder_past_key_values)
        else:
            pvs.append(None)
        return pvs

    def generate_parameters(self) ->None:
        """
        Generate parameters needed for new tokens' embedding in P-tuning
        """
        self.input_tokens = nn.Parameter(torch.arange(self.num_token).long(), requires_grad=False)
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:
            self.wte = nn.Embedding(self.num_token, self.n_embd)
            self.control_trans = nn.Sequential(nn.Linear(self.n_embd, self.mid_dim), nn.Tanh(), nn.Linear(self.mid_dim, self.n_layer * 2 * self.n_embd))
        if not self.config.is_encoder_decoder or self.using_decoder_past_key_values:
            self.decoder_wte = nn.Embedding(self.num_token, self.n_embd)
            self.decoder_control_trans = nn.Sequential(nn.Linear(self.n_embd, self.mid_dim), nn.Tanh(), nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd))

    def wrap_one_example(self, example) ->List[Dict]:
        if self.text is None:
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)

    def expand_to_batchsize(self, tup, batch_size):
        return tuple(t.expand(-1, batch_size, -1, -1, -1) for t in tup)

    def expand_to_batchsize_for_layer(self, tup, batch_size, layer_id):
        return tup[layer_id].expand(-1, batch_size, -1, -1, -1)

    def process_batch(self, batch: 'Union[Dict, InputFeatures]') ->Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal token, use the embedding inside PLM
        for new token, use MLP or LSTM
        """
        batch_size = batch['input_ids'].size(0)
        self.past_key_values = self.get_past_key_values()
        if self.config.is_encoder_decoder:
            pass
        else:
            past_key_values = self.expand_to_batchsize(self.past_key_values[1], batch_size)
            if 'attention_mask' in batch:
                am = batch['attention_mask']
                batch['attention_mask'] = torch.cat([torch.ones((batch_size, self.num_token), dtype=am.dtype, device=am.device), am], dim=-1)
            batch['past_key_values'] = past_key_values
        return batch

    def modify_plm(self, model):
        if self.plm_modified:
            return None
        if isinstance(model, T5ForConditionalGeneration):
            if self.using_encoder_past_key_values:
                backup_encoder_forward_functions = []
                for i, layer_module in enumerate(model.encoder.block):
                    backup_encoder_forward_functions.append(layer_module.layer[0].forward)

                    def modified_encoder_forward(*args, **kwargs):
                        layer_id = kwargs.pop('layer_id')
                        batch_size = args[0].shape[0]
                        device = args[0].device
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] = self.expand_to_batchsize_for_layer(self.past_key_values[0], batch_size, layer_id)
                        if kwargs['attention_mask'] is not None:
                            am = kwargs['attention_mask']
                            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1], self.num_token), dtype=am.dtype, device=am.device), am], dim=-1)
                        return backup_encoder_forward_functions[layer_id](*args, **kwargs)
                    layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)
            if self.using_decoder_past_key_values:
                backup_decoder_self_attn_forward_functions = []
                for i, layer_module in enumerate(model.decoder.block):
                    backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)

                    def modified_decoder_self_attn_forward(*args, **kwargs):
                        batch_size = args[0].shape[0]
                        layer_id = kwargs.pop('layer_id')
                        device = args[0].device
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] = self.expand_to_batchsize_for_layer(self.past_key_values[1], batch_size, layer_id)
                        if kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1):
                            pass
                        elif kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1) + self.num_token:
                            am = kwargs['attention_mask']
                            kwargs['attention_mask'] = torch.cat([torch.zeros((*am.shape[:-1], self.num_token), dtype=am.dtype, device=am.device), am], dim=-1)
                        else:
                            raise RuntimeError('Size not match: past length: {}, inputlength:{},                                attention mask length {}'.format(kwargs['past_key_value'][0].size(-2), args[0].size(-2), kwargs['attention_mask'].size(-1)))
                        return backup_decoder_self_attn_forward_functions[layer_id](*args, **kwargs)
                    layer_module.layer[0].forward = partial(modified_decoder_self_attn_forward, layer_id=i)
        elif isinstance(model, GPT2LMHeadModel):
            pass
        else:
            raise NotImplementedError
        self.plm_modified = True


class ProtoVerbalizer(Verbalizer):
    """
    The implementation of the prototypical verbalizer in `Prototypical Verbalizer for Prompt-based Few-shot Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ This class is inherited from the :obj:`Verbalizer` class.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
        lr: (:obj:`float`, optional): The learning rate for prototypes.
        mid_dim: (:obj:`int`, optional): The dimension of prototype embeddings.
        epochs: (:obj:`int`, optional): The training epochs of prototypes.
        multi_verb (:obj:`str`, optional): `multi` to ensemble with manual verbalizers, `proto` to use only ProtoVerb.
    """

    def __init__(self, tokenizer: 'Optional[PreTrainedTokenizer]', model: 'Optional[PreTrainedModel]', classes: 'Optional[List]'=None, num_classes: 'Optional[Sequence[str]]'=None, label_words: 'Optional[Union[Sequence[str], Mapping[str, str]]]'=None, prefix: 'Optional[str]'=' ', multi_token_handler: 'Optional[str]'='first', post_log_softmax: 'Optional[bool]'=True, lr: 'Optional[float]'=0.001, mid_dim: 'Optional[int]'=64, epochs: 'Optional[int]'=5, multi_verb: 'Optional[str]'='multi'):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.post_log_softmax = post_log_softmax
        self.multi_verb = multi_verb
        self.lr = lr
        self.mid_dim = mid_dim
        self.epochs = epochs
        self.trained = False
        self.hidden_dims = model.config.hidden_size
        self.head = torch.nn.Linear(self.hidden_dims, self.mid_dim, bias=False)
        if label_words is not None:
            self.label_words = label_words
        w = torch.empty((self.num_classes, self.mid_dim))
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)
        self.optimizer = torch.optim.Adam(self.group_parameters_proto, lr=self.lr)

    @property
    def group_parameters_proto(self):
        """Include the last layer's parameters
        """
        if isinstance(self.head, torch.nn.Linear):
            return [p for n, p in self.head.named_parameters()] + [self.proto]
        else:
            return [p for n, p in self.head.named_parameters()] + [self.proto]

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        """Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]
        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith('<!>'):
                    new_label_words_per_label.append(word.split('<!>')[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) ->List:
        """In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)
        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [([([1] * len(ids) + [0] * (max_len - len(ids))) for ids in ids_per_label] + [[0] * max_len] * (max_num_label_words - len(ids_per_label))) for ids_per_label in all_ids]
        words_ids = [([(ids + [0] * (max_len - len(ids))) for ids in ids_per_label] + [[0] * max_len] * (max_num_label_words - len(ids_per_label))) for ids_per_label in all_ids]
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def process_hiddens(self, hiddens: 'torch.Tensor', **kwargs):
        """A whole framework to process the original logits over the vocabulary, which contains four steps:
        """
        proto_logits = self.sim(self.head(hiddens), self.proto)
        return proto_logits

    def project(self, logits: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """
        Project the labels, the return value is the normalized (sum to 1) probs of label words.

        Args:
            logits (:obj:`torch.Tensor`): The original logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """
        label_words_logits = logits[:, self.label_words_ids]
        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        label_words_logits -= 10000 * (1 - self.label_words_mask)
        return label_words_logits

    def process_logits(self, logits: 'torch.Tensor', **kwargs):
        """A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        label_words_logits = self.project(logits, **kwargs)
        if self.post_log_softmax:
            if hasattr(self, '_calibrate_logits') and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)
        label_logits = self.aggregate(label_words_logits)
        return label_logits

    def normalize(self, logits: 'torch.Tensor') ->torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def aggregate(self, label_words_logits: 'torch.Tensor') ->torch.Tensor:
        """Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1) / self.label_words_mask.sum(-1)
        return label_words_logits

    def calibrate(self, label_words_probs: 'torch.Tensor', **kwargs) ->torch.Tensor:
        """

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() == 1, 'self._calibrate_logits are not 1-d tensor'
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] and calibrate_label_words_probs.shape[0] == 1, 'shape not match'
        label_words_probs /= calibrate_label_words_probs + 1e-15
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1, keepdim=True)
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs

    def ensemble_logits(self, manual_logits, proto_logits):
        logits = torch.stack([manual_logits, proto_logits])
        logits = logits.permute(1, 0, 2)
        logits = self.scaler(logits)
        logits = torch.mean(logits, 1)
        return logits

    @staticmethod
    def scaler(logits):
        m = logits.mean(-1, keepdim=True)
        s = logits.std(-1, keepdim=True)
        return (logits - m) / s

    def process_outputs(self, outputs: 'Union[torch.Tensor, torch.Tensor]', batch: 'Union[Dict, InputFeatures]', **kwargs):
        manual_logits = self.process_logits(outputs[1])
        if self.trained is False:
            return manual_logits
        proto_logits = self.process_hiddens(outputs[0])
        if self.trained and self.multi_verb == 'proto':
            return proto_logits
        return self.ensemble_logits(manual_logits, proto_logits)

    def gather_outputs(self, outputs: 'ModelOutput'):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")
        return ret, logits

    @staticmethod
    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))

    def pcl_loss(self, v_ins):
        sim_mat = torch.exp(self.sim(v_ins, self.proto))
        num = sim_mat.shape[1]
        loss = 0.0
        for i in range(num):
            pos_score = torch.diag(sim_mat[:, i, :])
            neg_score = sim_mat[:, i, :].sum(1) - pos_score
            loss += -torch.log(pos_score / (pos_score + neg_score)).sum()
        loss = loss / (num * self.num_classes * self.num_classes)
        loss_ins = 0.0
        for i in range(v_ins.shape[0]):
            sim_instance = torch.exp(self.sim(v_ins, v_ins[i]))
            pos_ins = sim_instance[i]
            neg_ins = (sim_instance.sum(0) - pos_ins).sum(0)
            loss_ins += -torch.log(pos_ins / (pos_ins + neg_ins)).sum()
        loss_ins = loss_ins / (num * self.num_classes * num * self.num_classes)
        loss = loss + loss_ins
        return loss

    def train_proto(self, model, dataloader, device):
        model.eval()
        embeds = [[] for _ in range(self.num_classes)]
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = batch.to_dict()
                outputs = model.prompt_model(batch)
                hidden, _ = self.gather_outputs(outputs)
                outputs_at_mask = model.extract_at_mask(hidden, batch)
                for j in range(len(outputs_at_mask)):
                    label = batch['label'][j]
                    embeds[label].append(outputs_at_mask[j])
        embeds = [torch.stack(e) for e in embeds]
        embeds = torch.stack(embeds)
        instance_mean = embeds.mean(1)
        loss = 0.0
        for epoch in range(self.epochs):
            x = self.head(embeds)
            self.optimizer.zero_grad()
            loss = self.pcl_loss(x)
            loss.backward()
            self.optimizer.step()
        logger.info('Total epoch: {}. ProtoVerb loss: {}'.format(self.epochs, loss))
        self.trained = True


class PTRVerbalizer(Verbalizer):
    """
    In `PTR <https://arxiv.org/pdf/2105.11259.pdf>`_, each prompt has more than one ``<mask>`` tokens.
    Different ``<mask>`` tokens have different label words.
    The final label is predicted jointly by these label words using logic rules.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        classes (:obj:`Sequence[str]`): A sequence of classes that need to be projected.
        label_words (:obj:`Union[Sequence[Sequence[str]], Mapping[str, Sequence[str]]]`, optional): The label words that are projected by the labels.
    """

    def __init__(self, tokenizer: 'PreTrainedTokenizer', classes: 'Sequence[str]'=None, num_classes: 'Optional[int]'=None, label_words: 'Optional[Union[Sequence[Sequence[str]], Mapping[str, Sequence[str]]]]'=None):
        super().__init__(tokenizer=tokenizer, classes=classes, num_classes=num_classes)
        self.label_words = label_words

    def on_label_words_set(self):
        """
        Prepare One2oneVerbalizer for each `<mask>` separately
        """
        super().on_label_words_set()
        self.num_masks = len(self.label_words[0])
        for words in self.label_words:
            if len(words) != self.num_masks:
                raise ValueError('number of mask tokens for different classes are not consistent')
        self.sub_labels = [list(set([words[i] for words in self.label_words])) for i in range(self.num_masks)]
        self.verbalizers = nn.ModuleList([One2oneVerbalizer(tokenizer=self.tokenizer, label_words=labels, post_log_softmax=False) for labels in self.sub_labels])
        self.label_mappings = nn.Parameter(torch.LongTensor([[labels.index(words[j]) for words in self.label_words] for j, labels in enumerate(self.sub_labels)]), requires_grad=False)

    def process_logits(self, logits: 'torch.Tensor', batch: 'Union[Dict, InputFeatures]', **kwargs):
        """
        1) Process vocab logits of each `<mask>` into label logits of each `<mask>`

        2) Combine these logits into a single label logits of the whole task

        Args:
            logits (:obj:`torch.Tensor`): vocab logits of each `<mask>` (shape: `[batch_size, num_masks, vocab_size]`)

        Returns:
            :obj:`torch.Tensor`: logits (label logits of whole task (shape: `[batch_size, label_size of the whole task]`))
        """
        each_logits = [self.verbalizers[i].process_logits(logits=logits[:, i, :], batch=batch, **kwargs) for i in range(self.num_masks)]
        label_logits = [logits[:, self.label_mappings[j]] for j, logits in enumerate(each_logits)]
        logsoftmax = nn.functional.log_softmax(sum(label_logits), dim=-1)
        if 'label' in batch:
            each_logsoftmax = [nn.functional.log_softmax(logits, dim=-1)[:, self.label_mappings[j]] for j, logits in enumerate(each_logits)]
            return logsoftmax + sum(each_logsoftmax) / len(each_logits)
        return logsoftmax


class PtuningTemplate(MixedTemplate):
    """
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        prompt_encoder_type (:obj:`str`): head above the embedding layer of new tokens. Can be ``lstm`` or ``mlp``.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """
    registered_inputflag_names = ['soft_token_ids', 'loss_ids', 'shortenable_ids']

    def __init__(self, model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizer', text: 'Optional[List[str]]'=None, prompt_encoder_type: 'str'='lstm', placeholder_mapping: 'dict'={'<text_a>': 'text_a', '<text_b>': 'text_b'}):
        super().__init__(model=model, tokenizer=tokenizer, placeholder_mapping=placeholder_mapping)
        self.prompt_encoder_type = prompt_encoder_type
        self.text = text

    def on_text_set(self):
        """
        when template text was set, generate parameters needed in p-tuning input embedding phrase
        """
        super().on_text_set()
        self.num_soft_token = sum([(soft_id != 0) for soft_id in self.soft_token_ids])
        self.generate_parameters()

    def generate_parameters(self) ->None:
        """
        generate parameters needed for new tokens' embedding in P-tuning
        """
        if self.num_soft_token == 0:
            return
        self.new_embedding = nn.Embedding(self.num_soft_token, self.embedding_size)
        self.new_ids = nn.Parameter(torch.LongTensor(list(range(self.num_soft_token))), requires_grad=False)
        if self.prompt_encoder_type == 'lstm':
            self.new_lstm_head = nn.LSTM(input_size=self.embedding_size, hidden_size=self.embedding_size, num_layers=2, bidirectional=True, batch_first=True)
            self.new_mlp_head = nn.Sequential(nn.Linear(2 * self.embedding_size, self.embedding_size), nn.ReLU(), nn.Linear(self.embedding_size, self.embedding_size))
        elif self.prompt_encoder_type == 'mlp':
            self.new_mlp_head = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size), nn.ReLU(), nn.Linear(self.embedding_size, self.embedding_size))
        else:
            raise ValueError('unknown prompt_enocder_type')

    def process_batch(self, batch: 'Union[Dict, InputFeatures]') ->Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for new tokens, use a brand new embedding layer, with MLP or LSTM head
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        if self.num_soft_token != 0:
            new_embeds = self.new_embedding(self.new_ids).unsqueeze(0)
            if self.prompt_encoder_type == 'lstm':
                new_embeds = self.new_lstm_head(new_embeds)[0]
            new_embeds = self.new_mlp_head(new_embeds)
            replace_idxs = torch.nonzero(batch['soft_token_ids'] > 0).view(-1, self.num_soft_token, 2)
            for b in range(replace_idxs.shape[0]):
                for i in range(self.num_soft_token):
                    inputs_embeds[b][replace_idxs[b][i][1]] = new_embeds[0][i]
        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch


class SoftTemplate(Template):
    """This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens.
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take
    the first n_tokens similar to their implementation).
    """
    registered_inputflag_names = ['loss_ids', 'shortenable_ids']

    def __init__(self, model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizer', text: 'Optional[str]'=None, soft_embeds: 'Optional[torch.FloatTensor]'=None, num_tokens: 'int'=20, initialize_from_vocab: 'Optional[bool]'=True, random_range: 'Optional[float]'=0.5, placeholder_mapping: 'dict'={'<text_a>': 'text_a', '<text_b>': 'text_b'}):
        super().__init__(tokenizer=tokenizer, placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.raw_embedding.requires_grad_(False)
        self.model_is_encoder_decoder = model.config.is_encoder_decoder
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab
        self.text = text
        if soft_embeds is not None:
            self.soft_embeds = soft_embeds
            self.num_tokens = len(soft_embeds)
        elif self.num_tokens > 0:
            self.generate_parameters()

    def on_text_set(self):
        self.text = self.parse_text(self.text)

    def wrap_one_example(self, example) ->List[Dict]:
        if self.text is None:
            logger.warning("You didn't provide text template for softprompt. Using default template, is this intended?")
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)

    def generate_parameters(self) ->None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        if self.initialize_from_vocab:
            soft_embeds = self.raw_embedding.weight[:self.num_tokens].clone().detach()
        else:
            soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)
        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)

    def process_batch(self, batch: 'Union[Dict, InputFeatures]') ->Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        if self.num_tokens > 0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)
        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens > 0:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size, self.num_tokens), dtype=am.dtype, device=am.device), am], dim=-1)
        return batch

    def post_processing_outputs(self, outputs: 'torch.Tensor'):
        """Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        if not self.model_is_encoder_decoder:
            outputs.logits = outputs.logits[:, self.num_tokens:, :]
        return outputs


class SoftVerbalizer(Verbalizer):
    """
    The implementation of the verbalizer in `WARP <https://aclanthology.org/2021.acl-long.381/>`_

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """

    def __init__(self, tokenizer: 'Optional[PreTrainedTokenizer]', model: 'Optional[PreTrainedModel]', classes: 'Optional[List]'=None, num_classes: 'Optional[Sequence[str]]'=None, label_words: 'Optional[Union[Sequence[str], Mapping[str, str]]]'=None, prefix: 'Optional[str]'=' ', multi_token_handler: 'Optional[str]'='first'):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        head_name = [n for n, c in model.named_children()][-1]
        logger.info(f'The LM head named {head_name} was retrieved.')
        self.head = copy.deepcopy(getattr(model, head_name))
        max_loop = 5
        if not isinstance(self.head, torch.nn.Linear):
            module = self.head
            found = False
            last_layer_full_name = []
            for i in range(max_loop):
                last_layer_name = [n for n, c in module.named_children()][-1]
                last_layer_full_name.append(last_layer_name)
                parent_module = module
                module = getattr(module, last_layer_name)
                if isinstance(module, torch.nn.Linear):
                    found = True
                    break
            if not found:
                raise RuntimeError(f"Can't not retrieve a linear layer in {max_loop} loop from the plm.")
            self.original_head_last_layer = module.weight.data
            self.hidden_dims = self.original_head_last_layer.shape[-1]
            self.head_last_layer_full_name = '.'.join(last_layer_full_name)
            self.head_last_layer = torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)
            setattr(parent_module, last_layer_name, self.head_last_layer)
        else:
            self.hidden_dims = self.head.weight.shape[-1]
            self.original_head_last_layer = getattr(model, head_name).weight.data
            self.head = torch.nn.Linear(self.hidden_dims, self.num_classes, bias=False)
        if label_words is not None:
            self.label_words = label_words

    @property
    def group_parameters_1(self):
        """Include the parameters of head's layer but not the last layer
        In soft verbalizer, note that some heads may contain modules
        other than the final projection layer. The parameters of these part should be
        optimized (or freezed) together with the plm.
        """
        if isinstance(self.head, torch.nn.Linear):
            return []
        else:
            return [p for n, p in self.head.named_parameters() if self.head_last_layer_full_name not in n]

    @property
    def group_parameters_2(self):
        """Include the last layer's parameters
        """
        if isinstance(self.head, torch.nn.Linear):
            return [p for n, p in self.head.named_parameters()]
        else:
            return [p for n, p in self.head.named_parameters() if self.head_last_layer_full_name in n]

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        """Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]
        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith('<!>'):
                    new_label_words_per_label.append(word.split('<!>')[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) ->List:
        """In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        words_ids = []
        for word in self.label_words:
            if isinstance(word, list):
                logger.warning('Label word for a class is a list, only use the first word.')
            word = word[0]
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(word_ids) > 1:
                logger.warning('Word {} is split into multiple tokens: {}.                     If this is not what you expect, try using another word for this verbalizer'.format(word, self.tokenizer.convert_ids_to_tokens(word_ids)))
            words_ids.append(word_ids)
        max_len = max([len(ids) for ids in words_ids])
        words_ids_mask = [([1] * len(ids) + [0] * (max_len - len(ids))) for ids in words_ids]
        words_ids = [(ids + [0] * (max_len - len(ids))) for ids in words_ids]
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.label_words_mask = nn.Parameter(words_ids_mask, requires_grad=False)
        init_data = self.original_head_last_layer[self.label_words_ids, :] * self.label_words_mask.unsqueeze(-1)
        init_data = init_data.sum(dim=1) / self.label_words_mask.sum(dim=-1, keepdim=True)
        if isinstance(self.head, torch.nn.Linear):
            self.head.weight.data = init_data
            self.head.weight.data.requires_grad = True
        else:
            """
            getattr(self.head, self.head_last_layer_full_name).weight.data = init_data
            getattr(self.head, self.head_last_layer_full_name).weight.data.requires_grad=True # To be sure
            """
            self.head_last_layer.weight.data = init_data
            self.head_last_layer.weight.data.requires_grad = True

    def process_hiddens(self, hiddens: 'torch.Tensor', **kwargs):
        """A whole framework to process the original logits over the vocabulary, which contains four steps:
        """
        label_logits = self.head(hiddens)
        return label_logits

    def process_outputs(self, outputs: 'torch.Tensor', batch: 'Union[Dict, InputFeatures]', **kwargs):
        return self.process_hiddens(outputs)

    def gather_outputs(self, outputs: 'ModelOutput'):
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Gather outputs method for outputs' type {type(outputs)} not implemented")
        return ret

