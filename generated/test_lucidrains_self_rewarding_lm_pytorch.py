
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


from copy import deepcopy


from functools import lru_cache


from collections import namedtuple


import torch


from torch.nn import Module


from torch.nn import Dropout


import torch.nn.functional as F


from torch.cuda.amp import autocast


from torch.optim.lr_scheduler import LinearLR


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.distributed as dist


from numpy.lib.format import open_memmap


from functools import wraps


from typing import Type


from typing import Any


from torch import Tensor


from torch.nn.utils.rnn import pad_sequence


import re


from functools import partial


from random import randrange


from torch.nn import ModuleList


from torch.utils.data import ConcatDataset


import numpy as np


@lru_cache(maxsize=None)
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def exists(v):
    return v is not None


def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    log_probs = logits.log_softmax(dim=-1)
    return get_at('b n [c], b n -> b n', log_probs, seq)


def prompt_mask_from_len(lengths, seq):
    seq_len, device = seq.shape[-1], seq.device
    return torch.arange(seq_len, device=device) < rearrange(lengths, '... -> ... 1')


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def set_dropout_(model: 'Module', prob: 'float'):
    for module in model.modules():
        if isinstance(module, Dropout):
            module.p = prob


class DPODataset(Dataset):

    def __init__(self, data_folder: 'str'='./', preference_seq_memmap_file: 'str'='preference_seq.memmap.npy', prompt_len_memmap_file: 'str'='prompt_len.memmap.npy'):
        self.data_folder = Path(data_folder)
        assert self.data_folder.exists() and self.data_folder.is_dir()
        preference_seq_memmap_path = self.data_folder / preference_seq_memmap_file
        prompt_len_memmap_path = self.data_folder / prompt_len_memmap_file
        assert preference_seq_memmap_path.exists()
        assert prompt_len_memmap_path.exists()
        self.paired_sequences = open_memmap(str(preference_seq_memmap_path), dtype='int', mode='r')
        self.prompt_len = open_memmap(str(prompt_len_memmap_path), dtype='int', mode='r')
        self.seq_len = self.paired_sequences.shape[1]
        assert self.paired_sequences.shape[0] == self.prompt_len.shape[0]

    def __len__(self):
        return self.paired_sequences.shape[0]

    def __getitem__(self, idx):
        sequences = self.paired_sequences[idx].copy()
        prompt_lens = self.prompt_len[idx].copy()
        preferred_seq, unpreferred_seq = sequences
        return preferred_seq, unpreferred_seq, prompt_lens


def find_variables_from_jinja_template(template: 'str'):
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)


def create_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {'reward'}, 'reward template must include "score" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(reward='([0-9\\.]+)')

    def parse_reward_fn(llm_response: 'str') ->float:
        result = re.search(f'{reward_regex_str}', llm_response)
        if not exists(result) or result.groups == 0:
            return None
        if not result.groups(1).isnumeric():
            return None
        return float(result.groups(1))
    return parse_reward_fn


def cast_input(cast_fn):

    def decorator(fn):

        @wraps(fn)
        def inner(t, *args, **kwargs):
            t = cast_fn(t)
            return fn(t, *args, **kwargs)
        return inner
    return decorator


def cast_output(cast_fn):

    def decorator(fn):

        @wraps(fn)
        def output(*args, **kwargs):
            out = fn(*args, **kwargs)
            out = cast_fn(out)
            return out
        return output
    return decorator


def default(v, d):
    return v if exists(v) else d


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, keepdim=True, eps=1e-10):
    return (t / max(temperature, eps) + gumbel_noise(t)).argmax(dim=dim, keepdim=keepdim)


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


class FinetuneConfig:
    pass


DEFAULT_LLM_AS_JUDGE_PROMPT = """
Review the user’s question and the corresponding response using the additive 5-point
scoring system described below. Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective,
addressing the user’s question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.
User: {{ prompt }}
<response>{{ response }}</response>
After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we’ll
systematically attribute points based on the outlined criteria.
"""


DEFAULT_REWARD_REGEX_TEMPLATE = """
Score: {{ reward }}
"""


default_is_valid_reward_pair = lambda preferred_reward, unpreferred_reward: (preferred_reward != unpreferred_reward).all()


def cast_tuple(t, length=1, validate=False):
    out = t if isinstance(t, tuple) else (t,) * length
    assert not validate or len(out) == length
    return out

