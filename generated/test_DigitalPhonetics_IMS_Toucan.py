
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


import itertools


import matplotlib.pyplot as plt


import numpy


import numpy as np


import torch.multiprocessing


from torch.nn import CTCLoss


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import random


from torch.multiprocessing import Manager


from torch.multiprocessing import Process


from torch.utils.data import Dataset


import time


from torch.nn.utils.rnn import pad_sequence


from torch.optim import RAdam


from torch.utils.data.dataloader import DataLoader


import torch.nn as nn


import torch.utils.data


import torch.utils.data.distributed


from torch import nn


import torch.optim as optim


from torch.autograd import grad as torch_grad


import math


import torch.nn.functional as F


from torch.nn.utils import spectral_norm


from abc import ABC


from torch.functional import stft as torch_stft


import scipy


import torch.distributions as dist


from torch.nn import functional as F


import torch.nn.functional as torchfunc


from torch.nn import Linear


from torch.nn import Sequential


from torch.nn import Tanh


from scipy.interpolate import interp1d


from torch.nn import Conv1d


from torch.nn.utils import remove_weight_norm


from torch.nn.utils import weight_norm


from scipy import signal as sig


from torch.nn import ConvTranspose1d


from torch.nn import ModuleList


import copy


from torch.optim.lr_scheduler import MultiStepLR


from torch import pow


from torch import sin


from torch.nn import Parameter


import warnings


import typing as tp


from collections import OrderedDict


import logging


import re


import matplotlib


import pandas as pd


from torch.utils.data import ConcatDataset


from torch.optim.lr_scheduler import _LRScheduler


from math import exp


from torch.autograd import Variable


from matplotlib.lines import Line2D


import scipy.io.wavfile


import torch.cuda


def chinese_number_conversion(text):
    zhdigits = '零一二三四五六七八九'
    zhplaces = {(0): '', (1): '十', (2): '百', (3): '千', (4): '万', (8): '亿'}
    zhplace_keys = sorted(zhplaces.keys())

    def numdigits(n):
        return len(str(abs(n)))

    def _zhnum(n):
        if n < 10:
            return zhdigits[n]
        named_place_len = zhplace_keys[bisect.bisect_right(zhplace_keys, numdigits(n) - 1) - 1]
        left_part, right_part = n // 10 ** named_place_len, n % 10 ** named_place_len
        return _zhnum(left_part) + zhplaces[named_place_len] + ((zhdigits[0] if numdigits(right_part) != named_place_len else '') + _zhnum(right_part) if right_part else '')

    def zhnum(n):
        answer = ('负' if n < 0 else '') + _zhnum(abs(n))
        answer = re.sub('^一十', '十', answer)
        answer = re.sub('(?<![零十])二(?=[千万亿])', '两', answer)
        return answer
    return re.sub('\\d+', lambda x: zhnum(int(x.group())), text)


def convert_kanji_to_pinyin_mandarin(text):
    text = chinese_number_conversion(text)
    return ' '.join([x[0] for x in pinyin(text)])


def english_text_expansion(text):
    """
    Apply as small part of the tacotron style text cleaning pipeline, suitable for e.g. LJSpeech.
    See https://github.com/keithito/tacotron/
    Careful: Only apply to english datasets. Different languages need different cleaners.
    """
    _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [('Mrs.', 'misess'), ('Mr.', 'mister'), ('Dr.', 'doctor'), ('St.', 'saint'), ('Co.', 'company'), ('Jr.', 'junior'), ('Maj.', 'major'), ('Gen.', 'general'), ('Drs.', 'doctors'), ('Rev.', 'reverend'), ('Lt.', 'lieutenant'), ('Hon.', 'honorable'), ('Sgt.', 'sergeant'), ('Capt.', 'captain'), ('Esq.', 'esquire'), ('Ltd.', 'limited'), ('Col.', 'colonel'), ('Ft.', 'fort'), ('e.g.', ', for example, '), ('TTS', 'text to speech')]]
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def generate_feature_lookup():
    return {'~': {'symbol_type': 'silence'}, '#': {'symbol_type': 'end of sentence'}, '?': {'symbol_type': 'questionmark'}, '!': {'symbol_type': 'exclamationmark'}, '.': {'symbol_type': 'fullstop'}, ' ': {'symbol_type': 'word-boundary'}, 'ɜ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central', 'vowel_openness': 'open-mid', 'vowel_roundedness': 'unrounded'}, 'ə': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central', 'vowel_openness': 'mid', 'vowel_roundedness': 'unrounded'}, 'a': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'open', 'vowel_roundedness': 'unrounded'}, 'ð': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'dental', 'consonant_manner': 'fricative'}, 'ɛ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'open-mid', 'vowel_roundedness': 'unrounded'}, 'ɪ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front_central', 'vowel_openness': 'close_close-mid', 'vowel_roundedness': 'unrounded'}, 'ŋ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'velar', 'consonant_manner': 'nasal'}, 'ɔ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'open-mid', 'vowel_roundedness': 'rounded'}, 'ɒ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'open', 'vowel_roundedness': 'rounded'}, 'ɾ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'alveolar', 'consonant_manner': 'flap'}, 'ʃ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'postalveolar', 'consonant_manner': 'fricative'}, 'θ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'dental', 'consonant_manner': 'fricative'}, 'ʊ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central_back', 'vowel_openness': 'close_close-mid', 'vowel_roundedness': 'unrounded'}, 'ʌ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'open-mid', 'vowel_roundedness': 'unrounded'}, 'ʒ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'postalveolar', 'consonant_manner': 'fricative'}, 'æ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'open-mid_open', 'vowel_roundedness': 'unrounded'}, 'b': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'bilabial', 'consonant_manner': 'plosive'}, 'ʔ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'glottal', 'consonant_manner': 'plosive'}, 'd': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'alveolar', 'consonant_manner': 'plosive'}, 'e': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'close-mid', 'vowel_roundedness': 'unrounded'}, 'f': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'labiodental', 'consonant_manner': 'fricative'}, 'ɡ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'velar', 'consonant_manner': 'plosive'}, 'h': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'glottal', 'consonant_manner': 'fricative'}, 'i': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'close', 'vowel_roundedness': 'unrounded'}, 'j': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'palatal', 'consonant_manner': 'approximant'}, 'k': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'velar', 'consonant_manner': 'plosive'}, 'l': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'alveolar', 'consonant_manner': 'lateral-approximant'}, 'm': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'bilabial', 'consonant_manner': 'nasal'}, 'n': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'alveolar', 'consonant_manner': 'nasal'}, 'ɳ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'retroflex', 'consonant_manner': 'nasal'}, 'o': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'close-mid', 'vowel_roundedness': 'rounded'}, 'p': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'bilabial', 'consonant_manner': 'plosive'}, 'ɹ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'alveolar', 'consonant_manner': 'approximant'}, 'r': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'alveolar', 'consonant_manner': 'trill'}, 's': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'alveolar', 'consonant_manner': 'fricative'}, 't': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'alveolar', 'consonant_manner': 'plosive'}, 'u': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'close', 'vowel_roundedness': 'rounded'}, 'v': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'labiodental', 'consonant_manner': 'fricative'}, 'w': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'labial-velar', 'consonant_manner': 'approximant'}, 'x': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'velar', 'consonant_manner': 'fricative'}, 'z': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'alveolar', 'consonant_manner': 'fricative'}, 'ʀ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'uvular', 'consonant_manner': 'trill'}, 'ø': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'close-mid', 'vowel_roundedness': 'rounded'}, 'ç': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'palatal', 'consonant_manner': 'fricative'}, 'ɐ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central', 'vowel_openness': 'open', 'vowel_roundedness': 'unrounded'}, 'œ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'open-mid', 'vowel_roundedness': 'rounded'}, 'y': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'close', 'vowel_roundedness': 'rounded'}, 'ʏ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front_central', 'vowel_openness': 'close_close-mid', 'vowel_roundedness': 'rounded'}, 'ɑ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'open', 'vowel_roundedness': 'unrounded'}, 'c': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'palatal', 'consonant_manner': 'plosive'}, 'ɲ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'palatal', 'consonant_manner': 'nasal'}, 'ɣ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'velar', 'consonant_manner': 'fricative'}, 'ʎ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'palatal', 'consonant_manner': 'lateral-approximant'}, 'β': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'bilabial', 'consonant_manner': 'fricative'}, 'ʝ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'palatal', 'consonant_manner': 'fricative'}, 'ɟ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'palatal', 'consonant_manner': 'plosive'}, 'q': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'uvular', 'consonant_manner': 'plosive'}, 'ɕ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'alveolopalatal', 'consonant_manner': 'fricative'}, 'ɭ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'retroflex', 'consonant_manner': 'lateral-approximant'}, 'ɵ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central', 'vowel_openness': 'close-mid', 'vowel_roundedness': 'rounded'}, 'ʑ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'alveolopalatal', 'consonant_manner': 'fricative'}, 'ʋ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'labiodental', 'consonant_manner': 'approximant'}, 'ʁ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'uvular', 'consonant_manner': 'fricative'}, 'ɨ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central', 'vowel_openness': 'close', 'vowel_roundedness': 'unrounded'}, 'ʂ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'retroflex', 'consonant_manner': 'fricative'}, 'ɓ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'bilabial', 'consonant_manner': 'implosive'}, 'ʙ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'bilabial', 'consonant_manner': 'vibrant'}, 'ɗ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'dental', 'consonant_manner': 'implosive'}, 'ɖ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'retroflex', 'consonant_manner': 'plosive'}, 'χ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'uvular', 'consonant_manner': 'fricative'}, 'ʛ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'uvular', 'consonant_manner': 'implosive'}, 'ʟ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'velar', 'consonant_manner': 'lateral-approximant'}, 'ɽ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'retroflex', 'consonant_manner': 'flap'}, 'ɢ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'uvular', 'consonant_manner': 'plosive'}, 'ɠ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'velar', 'consonant_manner': 'implosive'}, 'ǂ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'alveolopalatal', 'consonant_manner': 'click'}, 'ɦ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'glottal', 'consonant_manner': 'fricative'}, 'ǁ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'alveolar', 'consonant_manner': 'click'}, 'ĩ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'close', 'vowel_roundedness': 'unrounded', 'consonant_manner': 'nasal'}, 'ʍ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'labial-velar', 'consonant_manner': 'fricative'}, 'ʕ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'pharyngal', 'consonant_manner': 'fricative'}, 'ɻ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'retroflex', 'consonant_manner': 'approximant'}, 'ʄ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'palatal', 'consonant_manner': 'implosive'}, 'ũ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'close', 'vowel_roundedness': 'rounded', 'consonant_manner': 'nasal'}, 'ɤ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'close-mid', 'vowel_roundedness': 'unrounded'}, 'ɶ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'front', 'vowel_openness': 'open', 'vowel_roundedness': 'rounded'}, 'õ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'close-mid', 'vowel_roundedness': 'rounded', 'consonant_manner': 'nasal'}, 'ʡ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'epiglottal', 'consonant_manner': 'plosive'}, 'ʈ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'retroflex', 'consonant_manner': 'plosive'}, 'ʜ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'epiglottal', 'consonant_manner': 'fricative'}, 'ɱ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'labiodental', 'consonant_manner': 'nasal'}, 'ɯ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'back', 'vowel_openness': 'close', 'vowel_roundedness': 'unrounded'}, 'ǀ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'dental', 'consonant_manner': 'click'}, 'ɸ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'bilabial', 'consonant_manner': 'fricative'}, 'ʘ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'bilabial', 'consonant_manner': 'click'}, 'ʐ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'retroflex', 'consonant_manner': 'fricative'}, 'ɰ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'velar', 'consonant_manner': 'approximant'}, 'ɘ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central', 'vowel_openness': 'close-mid', 'vowel_roundedness': 'unrounded'}, 'ħ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'pharyngal', 'consonant_manner': 'fricative'}, 'ɞ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central', 'vowel_openness': 'open-mid', 'vowel_roundedness': 'rounded'}, 'ʉ': {'symbol_type': 'phoneme', 'vowel_consonant': 'vowel', 'VUV': 'voiced', 'vowel_frontness': 'central', 'vowel_openness': 'close', 'vowel_roundedness': 'rounded'}, 'ɴ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'uvular', 'consonant_manner': 'nasal'}, 'ʢ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'epiglottal', 'consonant_manner': 'fricative'}, 'ѵ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'voiced', 'consonant_place': 'labiodental', 'consonant_manner': 'flap'}, 'ǃ': {'symbol_type': 'phoneme', 'vowel_consonant': 'consonant', 'VUV': 'unvoiced', 'consonant_place': 'postalveolar', 'consonant_manner': 'click'}}


def get_feature_to_index_lookup():
    return {'stressed': 0, 'very-high-tone': 1, 'high-tone': 2, 'mid-tone': 3, 'low-tone': 4, 'very-low-tone': 5, 'rising-tone': 6, 'falling-tone': 7, 'peaking-tone': 8, 'dipping-tone': 9, 'lengthened': 10, 'half-length': 11, 'shortened': 12, 'consonant': 13, 'vowel': 14, 'phoneme': 15, 'silence': 16, 'end of sentence': 17, 'questionmark': 18, 'exclamationmark': 19, 'fullstop': 20, 'word-boundary': 21, 'dental': 22, 'postalveolar': 23, 'velar': 24, 'palatal': 25, 'glottal': 26, 'uvular': 27, 'labiodental': 28, 'labial-velar': 29, 'alveolar': 30, 'bilabial': 31, 'alveolopalatal': 32, 'retroflex': 33, 'pharyngal': 34, 'epiglottal': 35, 'central': 36, 'back': 37, 'front_central': 38, 'front': 39, 'central_back': 40, 'mid': 41, 'close-mid': 42, 'close': 43, 'open-mid': 44, 'close_close-mid': 45, 'open-mid_open': 46, 'open': 47, 'rounded': 48, 'unrounded': 49, 'plosive': 50, 'nasal': 51, 'approximant': 52, 'trill': 53, 'flap': 54, 'fricative': 55, 'lateral-approximant': 56, 'implosive': 57, 'vibrant': 58, 'click': 59, 'ejective': 60, 'aspirated': 61, 'unvoiced': 62, 'voiced': 63}


def generate_feature_table():
    ipa_to_phonemefeats = generate_feature_lookup()
    feat_types = set()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            [feat_types.add(feat) for feat in ipa_to_phonemefeats[ipa].keys()]
    feat_to_val_set = dict()
    for feat in feat_types:
        feat_to_val_set[feat] = set()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            for feat in ipa_to_phonemefeats[ipa]:
                feat_to_val_set[feat].add(ipa_to_phonemefeats[ipa][feat])
    value_list = set()
    for val_set in [feat_to_val_set[feat] for feat in feat_to_val_set]:
        for value in val_set:
            value_list.add(value)
    value_to_index = get_feature_to_index_lookup()
    phone_to_vector = dict()
    for ipa in ipa_to_phonemefeats:
        if len(ipa) == 1:
            phone_to_vector[ipa] = [0] * (15 + sum([len(values) for values in [feat_to_val_set[feat] for feat in feat_to_val_set]]))
            for feat in ipa_to_phonemefeats[ipa]:
                if ipa_to_phonemefeats[ipa][feat] in value_to_index:
                    phone_to_vector[ipa][value_to_index[ipa_to_phonemefeats[ipa][feat]]] = 1
            if phone_to_vector[ipa][value_to_index['phoneme']] != 1:
                phone_to_vector[ipa][value_to_index['silence']] = 1
    for feat in feat_to_val_set:
        for value in feat_to_val_set[feat]:
            if value not in value_to_index:
                None
    return phone_to_vector


def get_phone_to_id():
    """
    for the states of the ctc loss and dijkstra/mas in the aligner
    cannot be extracted trivially from above because sets are unordered and the IDs need to be consistent
    """
    phone_to_id = dict()
    for index, phone in enumerate('~#?!ǃ.ɜəaðɛɪŋɔɒɾʃθʊʌʒæbʔdefghijklmnɳopɡɹrstuvwxzʀøçɐœyʏɑcɲɣʎβʝɟqɕɭɵʑʋʁɨʂɓʙɗɖχʛʟɽɢɠǂɦǁĩʍʕɻʄũɤɶõʡʈʜɱɯǀɸʘʐɰɘħɞʉɴʢѵ'):
        phone_to_id[phone] = index
    phone_to_id['#'] = phone_to_id['~']
    phone_to_id['?'] = phone_to_id['~']
    phone_to_id['!'] = phone_to_id['~']
    phone_to_id['.'] = phone_to_id['~']
    return phone_to_id


def remove_french_spacing(text):
    text = text.replace(' »', '"').replace('« ', '"')
    for punc in ['!', ';', ':', '.', ',', '?', '-']:
        text = text.replace(f' {punc}', punc)
    return text


class ArticulatoryCombinedTextFrontend:

    def __init__(self, language, use_explicit_eos=True, use_lexical_stress=True, silent=True, add_silence_to_end=True, use_word_boundaries=True, device='cpu'):
        """
        Mostly preparing ID lookups
        """
        self.language = language
        self.use_explicit_eos = use_explicit_eos
        self.use_stress = use_lexical_stress
        self.add_silence_to_end = add_silence_to_end
        self.use_word_boundaries = use_word_boundaries
        register_to_height = {'˥': 5, '˦': 4, '˧': 3, '˨': 2, '˩': 1}
        self.rising_perms = list()
        self.falling_perms = list()
        self.peaking_perms = list()
        self.dipping_perms = list()
        for first_tone in ['˥', '˦', '˧', '˨', '˩']:
            for second_tone in ['˥', '˦', '˧', '˨', '˩']:
                if register_to_height[first_tone] > register_to_height[second_tone]:
                    self.falling_perms.append(first_tone + second_tone)
                else:
                    self.rising_perms.append(first_tone + second_tone)
                for third_tone in ['˥', '˦', '˧', '˨', '˩']:
                    if register_to_height[first_tone] > register_to_height[second_tone] < register_to_height[third_tone]:
                        self.dipping_perms.append(first_tone + second_tone + third_tone)
                    elif register_to_height[first_tone] < register_to_height[second_tone] > register_to_height[third_tone]:
                        self.peaking_perms.append(first_tone + second_tone + third_tone)
        if language == 'eng' or language == 'en-us':
            self.g2p_lang = 'en-us'
            self.expand_abbreviations = english_text_expansion
            self.phonemizer = 'espeak'
        elif language == 'deu':
            self.g2p_lang = 'de'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ell':
            self.g2p_lang = 'el'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'spa':
            self.g2p_lang = 'es'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'spa-lat':
            self.g2p_lang = 'es-419'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'fin':
            self.g2p_lang = 'fi'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'rus':
            self.g2p_lang = 'ru'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'hun':
            self.g2p_lang = 'hu'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'nld':
            self.g2p_lang = 'nl'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'fra':
            self.g2p_lang = 'fr-fr'
            self.expand_abbreviations = remove_french_spacing
            self.phonemizer = 'espeak'
        elif language == 'fr-be':
            self.g2p_lang = 'fr-be'
            self.expand_abbreviations = remove_french_spacing
            self.phonemizer = 'espeak'
        elif language == 'fr-sw':
            self.g2p_lang = 'fr-ch'
            self.expand_abbreviations = remove_french_spacing
            self.phonemizer = 'espeak'
        elif language == 'ita':
            self.g2p_lang = 'it'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'por':
            self.g2p_lang = 'pt'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'pt-br':
            self.g2p_lang = 'pt-br'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'pol':
            self.g2p_lang = 'pl'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'cmn':
            self.g2p_lang = 'cmn'
            self.expand_abbreviations = convert_kanji_to_pinyin_mandarin
            self.phonemizer = 'dragonmapper'
        elif language == 'vie':
            self.g2p_lang = 'vi'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'vi-ctr':
            self.g2p_lang = 'vi-vn-x-central'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'vi-so':
            self.g2p_lang = 'vi-vn-x-south'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ukr':
            self.g2p_lang = 'uk'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'pes':
            self.g2p_lang = 'fa'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'afr':
            self.g2p_lang = 'af'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'aln':
            self.g2p_lang = 'sq'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'amh':
            self.g2p_lang = 'am'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'arb':
            self.g2p_lang = 'ar'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'arg':
            self.g2p_lang = 'an'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'hye':
            self.g2p_lang = 'hy'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'hyw':
            self.g2p_lang = 'hyw'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'azj':
            self.g2p_lang = 'az'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'bak':
            self.g2p_lang = 'ba'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'eus':
            self.g2p_lang = 'eu'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'bel':
            self.g2p_lang = 'be'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ben':
            self.g2p_lang = 'bn'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'bpy':
            self.g2p_lang = 'bpy'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'bos':
            self.g2p_lang = 'bs'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'bul':
            self.g2p_lang = 'bg'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'mya':
            self.g2p_lang = 'my'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'chr':
            self.g2p_lang = 'chr'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'yue':
            self.g2p_lang = 'yue'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'hak':
            self.g2p_lang = 'hak'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'haw':
            self.g2p_lang = 'haw'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'hrv':
            self.g2p_lang = 'hr'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ces':
            self.g2p_lang = 'cs'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'dan':
            self.g2p_lang = 'da'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ekk':
            self.g2p_lang = 'et'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'gle':
            self.g2p_lang = 'ga'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'gla':
            self.g2p_lang = 'gd'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'en-sc':
            self.g2p_lang = 'en-gb-scotland'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'kat':
            self.g2p_lang = 'ka'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'kal':
            self.g2p_lang = 'kl'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'guj':
            self.g2p_lang = 'gu'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'heb':
            self.g2p_lang = 'he'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'hin':
            self.g2p_lang = 'hi'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'isl':
            self.g2p_lang = 'is'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ind':
            self.g2p_lang = 'id'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'jpn':
            self.kakasi = pykakasi.Kakasi()
            self.expand_abbreviations = lambda x: ' '.join([chunk['hepburn'] for chunk in self.kakasi.convert(x)])
            self.g2p_lang = language
            self.phonemizer = 'transphone'
            self.transphone = read_g2p(device=device)
        elif language == 'kan':
            self.g2p_lang = 'kn'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'knn':
            self.g2p_lang = 'kok'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'kor':
            self.g2p_lang = 'ko'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ckb':
            self.g2p_lang = 'ku'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'kaz':
            self.g2p_lang = 'kk'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'kir':
            self.g2p_lang = 'ky'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'lat':
            self.g2p_lang = 'la'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ltz':
            self.g2p_lang = 'lb'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'lvs':
            self.g2p_lang = 'lv'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'lit':
            self.g2p_lang = 'lt'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'mri':
            self.g2p_lang = 'mi'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'mkd':
            self.g2p_lang = 'mk'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'zlm':
            self.g2p_lang = 'ms'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'mal':
            self.g2p_lang = 'ml'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'mlt':
            self.g2p_lang = 'mt'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'mar':
            self.g2p_lang = 'mr'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'nci':
            self.g2p_lang = 'nci'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'npi':
            self.g2p_lang = 'ne'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'nob':
            self.g2p_lang = 'nb'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'nog':
            self.g2p_lang = 'nog'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ory':
            self.g2p_lang = 'or'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'gaz':
            self.g2p_lang = 'om'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'pap':
            self.g2p_lang = 'pap'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'pan':
            self.g2p_lang = 'pa'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'ron':
            self.g2p_lang = 'ro'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'lav':
            self.g2p_lang = 'ru-lv'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'srp':
            self.g2p_lang = 'sr'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'tsn':
            self.g2p_lang = 'tn'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'snd':
            self.g2p_lang = 'sd'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'slk':
            self.g2p_lang = 'sk'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'slv':
            self.g2p_lang = 'sl'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'smj':
            self.g2p_lang = 'smj'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'swh':
            self.g2p_lang = 'sw'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'swe':
            self.g2p_lang = 'sv'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'tam':
            self.g2p_lang = 'ta'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'tha':
            self.g2p_lang = 'th'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'tuk':
            self.g2p_lang = 'tk'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'tat':
            self.g2p_lang = 'tt'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'tel':
            self.g2p_lang = 'te'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'tur':
            self.g2p_lang = 'tr'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'uig':
            self.g2p_lang = 'ug'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'urd':
            self.g2p_lang = 'ur'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'uzn':
            self.g2p_lang = 'uz'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        elif language == 'cym':
            self.g2p_lang = 'cy'
            self.expand_abbreviations = lambda x: x
            self.phonemizer = 'espeak'
        else:
            None
            self.g2p_lang = language
            self.phonemizer = 'transphone'
            self.expand_abbreviations = lambda x: x
            self.transphone = read_g2p(device=device)
        if self.phonemizer == 'espeak':
            try:
                self.phonemizer_backend = EspeakBackend(language=self.g2p_lang, punctuation_marks=';:,.!?¡¿—…()"«»“”~/。【】、‥،؟“”؛', preserve_punctuation=True, language_switch='remove-flags', with_stress=self.use_stress, logger=logging.getLogger(__file__))
            except RuntimeError:
                None
                self.g2p_lang = self.language
                self.phonemizer = 'transphone'
                self.expand_abbreviations = lambda x: x
                self.transphone = read_g2p()
        self.phone_to_vector = generate_feature_table()
        self.phone_to_id = get_phone_to_id()
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}
        self.text_vector_to_phone_cache = dict()

    @staticmethod
    def get_example_sentence(lang):
        if lang == 'eng':
            return 'This is a complex sentence, it even has a pause!'
        elif lang == 'deu':
            return 'Dies ist ein komplexer Satz, er hat sogar eine Pause!'
        elif lang == 'ell':
            return 'Αυτή είναι μια σύνθετη πρόταση, έχει ακόμη και παύση!'
        elif lang == 'spa':
            return 'Esta es una oración compleja, ¡incluso tiene una pausa!'
        elif lang == 'fin':
            return 'Tämä on monimutkainen lause, sillä on jopa tauko!'
        elif lang == 'rus':
            return 'Это сложное предложение, в нем даже есть пауза!'
        elif lang == 'hun':
            return 'Ez egy összetett mondat, még szünet is van benne!'
        elif lang == 'nld':
            return 'Dit is een complexe zin, er zit zelfs een pauze in!'
        elif lang == 'fra':
            return "C'est une phrase complexe, elle a même une pause !"
        elif lang == 'por':
            return 'Esta é uma frase complexa, tem até uma pausa!'
        elif lang == 'pol':
            return 'To jest zdanie złożone, ma nawet pauzę!'
        elif lang == 'ita':
            return 'Questa è una frase complessa, ha anche una pausa!'
        elif lang == 'cmn':
            return '这是一个复杂的句子，它甚至包含一个停顿。'
        elif lang == 'vie':
            return 'Đây là một câu phức tạp, nó thậm chí còn chứa một khoảng dừng.'
        else:
            None
            return None

    def string_to_tensor(self, text, view=False, device='cpu', handle_missing=True, input_phonemes=False):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence as articulatory features
        """
        if input_phonemes:
            phones = text
        else:
            phones = self.get_phone_string(text=text, include_eos_symbol=True, for_feature_extraction=True)
        phones = phones.replace('ɚ', 'ə').replace('ᵻ', 'ɨ')
        if view:
            None
        phones_vector = list()
        stressed_flag = False
        for char in phones:
            if char.strip() == 'ˈ':
                stressed_flag = True
            elif char.strip() == 'ː':
                phones_vector[-1][get_feature_to_index_lookup()['lengthened']] = 1
            elif char.strip() == 'ˑ':
                phones_vector[-1][get_feature_to_index_lookup()['half-length']] = 1
            elif char.strip() == '̆':
                phones_vector[-1][get_feature_to_index_lookup()['shortened']] = 1
            elif char.strip() == '̃' and phones_vector[-1][get_feature_to_index_lookup()['nasal']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['nasal']] = 2
            elif char.strip() == '̧' != phones_vector[-1][get_feature_to_index_lookup()['palatal']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['palatal']] = 2
            elif char.strip() == 'ʷ' and phones_vector[-1][get_feature_to_index_lookup()['labial-velar']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['labial-velar']] = 2
            elif char.strip() == 'ʰ' and phones_vector[-1][get_feature_to_index_lookup()['aspirated']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['aspirated']] = 2
            elif char.strip() == 'ˠ' and phones_vector[-1][get_feature_to_index_lookup()['velar']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['velar']] = 2
            elif char.strip() == 'ˁ' and phones_vector[-1][get_feature_to_index_lookup()['pharyngal']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['pharyngal']] = 2
            elif char.strip() == 'ˀ' and phones_vector[-1][get_feature_to_index_lookup()['glottal']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['glottal']] = 2
            elif char.strip() == 'ʼ' and phones_vector[-1][get_feature_to_index_lookup()['ejective']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['ejective']] = 2
            elif char.strip() == '̹' and phones_vector[-1][get_feature_to_index_lookup()['rounded']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['rounded']] = 2
            elif char.strip() == '̞' and phones_vector[-1][get_feature_to_index_lookup()['open']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['open']] = 2
            elif char.strip() == '̪' and phones_vector[-1][get_feature_to_index_lookup()['dental']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['dental']] = 2
            elif char.strip() == '̬' and phones_vector[-1][get_feature_to_index_lookup()['voiced']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['voiced']] = 2
            elif char.strip() == '̝' and phones_vector[-1][get_feature_to_index_lookup()['close']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['close']] = 2
            elif char.strip() == '̰' and phones_vector[-1][get_feature_to_index_lookup()['glottal']] != 1 and phones_vector[-1][get_feature_to_index_lookup()['epiglottal']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['glottal']] = 2
                phones_vector[-1][get_feature_to_index_lookup()['epiglottal']] = 2
            elif char.strip() == '̈' and phones_vector[-1][get_feature_to_index_lookup()['central']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['central']] = 2
            elif char.strip() == '̜' and phones_vector[-1][get_feature_to_index_lookup()['unrounded']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['unrounded']] = 2
            elif char.strip() == '̥' and phones_vector[-1][get_feature_to_index_lookup()['unvoiced']] != 1:
                phones_vector[-1][get_feature_to_index_lookup()['unvoiced']] = 2
            elif char.strip() == '˥':
                phones_vector[-1][get_feature_to_index_lookup()['very-high-tone']] = 1
            elif char.strip() == '˦':
                phones_vector[-1][get_feature_to_index_lookup()['high-tone']] = 1
            elif char.strip() == '˧':
                phones_vector[-1][get_feature_to_index_lookup()['mid-tone']] = 1
            elif char.strip() == '˨':
                phones_vector[-1][get_feature_to_index_lookup()['low-tone']] = 1
            elif char.strip() == '˩':
                phones_vector[-1][get_feature_to_index_lookup()['very-low-tone']] = 1
            elif char.strip() == '⭧':
                phones_vector[-1][get_feature_to_index_lookup()['rising-tone']] = 1
            elif char.strip() == '⭨':
                phones_vector[-1][get_feature_to_index_lookup()['falling-tone']] = 1
            elif char.strip() == '⮁':
                phones_vector[-1][get_feature_to_index_lookup()['peaking-tone']] = 1
            elif char.strip() == '⮃':
                phones_vector[-1][get_feature_to_index_lookup()['dipping-tone']] = 1
            else:
                if handle_missing:
                    try:
                        phones_vector.append(self.phone_to_vector[char].copy())
                    except KeyError:
                        None
                else:
                    phones_vector.append(self.phone_to_vector[char].copy())
                if stressed_flag:
                    stressed_flag = False
                    phones_vector[-1][get_feature_to_index_lookup()['stressed']] = 1
        return torch.Tensor(phones_vector, device=device)

    def get_phone_string(self, text, include_eos_symbol=True, for_feature_extraction=False, for_plot_labels=False):
        if text == '':
            return ''
        utt = self.expand_abbreviations(text)
        if self.phonemizer == 'espeak':
            try:
                phones = self.phonemizer_backend.phonemize([utt], strip=True)[0]
            except:
                None
                self.g2p_lang = self.language
                self.phonemizer = 'transphone'
                self.expand_abbreviations = lambda x: x
                self.transphone = read_g2p()
                return self.get_phone_string(text, include_eos_symbol, for_feature_extraction, for_plot_labels)
        elif self.phonemizer == 'transphone':
            replacements = [('。', '~'), ('，', '~'), ('【', '~'), ('】', '~'), ('、', '~'), ('‥', '~'), ('؟', '~'), ('،', '~'), ('“', '~'), ('”', '~'), ('؛', '~'), ('《', '~'), ('》', '~'), ('？', '~'), ('！', '~'), (' ：', '~'), (' ；', '~'), ('－', '~'), ('·', ' '), ('`', ''), ('"', '~'), (' - ', '~ '), ('- ', '~ '), ('-', ''), ('…', '~'), (':', '~'), (';', '~'), (',', '~')]
            for replacement in replacements:
                utt = utt.replace(replacement[0], replacement[1])
            utt = re.sub('~+', '~', utt)
            utt = re.sub('\\s+', ' ', utt)
            utt = re.sub('\\.+', '.', utt)
            chunk_list = list()
            for chunk in utt.split('~'):
                word_list = list()
                for word_by_whitespace in chunk.split():
                    word_list.append(self.transphone.inference(word_by_whitespace, self.g2p_lang))
                chunk_list.append(' '.join([''.join(word) for word in word_list]))
            phones = '~ '.join(chunk_list)
        elif self.phonemizer == 'dragonmapper':
            phones = pinyin_to_ipa(utt)
        if self.g2p_lang == 'vi' or self.g2p_lang == 'vi-vn-x-central' or self.g2p_lang == 'vi-vn-x-south':
            phones = phones.replace('1', '˧')
            phones = phones.replace('2', '˨˩')
            phones = phones.replace('ɜ', '˧˥')
            phones = phones.replace('3', '˧˥')
            phones = phones.replace('4', '˦˧˥')
            phones = phones.replace('5', '˧˩˧')
            phones = phones.replace('6', '˧˩˨ʔ')
            phones = phones.replace('7', '˧')
        elif self.g2p_lang == 'yue':
            phones = phones.replace('1', '˥')
            phones = phones.replace('2', '˧˥')
            phones = phones.replace('3', '˧')
            phones = phones.replace('4', '˧˩')
            phones = phones.replace('5', '˩˧')
            phones = phones.replace('6', '˨')
        return self.postprocess_phoneme_string(phones, for_feature_extraction, include_eos_symbol, for_plot_labels)

    def postprocess_phoneme_string(self, phoneme_string, for_feature_extraction, include_eos_symbol, for_plot_labels):
        """
        Takes as input a phoneme string and processes it to work best with the way we represent phonemes as featurevectors
        """
        replacements = [('。', '.'), ('，', ','), ('【', '"'), ('】', '"'), ('、', ','), ('‥', '…'), ('؟', '?'), ('،', ','), ('“', '"'), ('”', '"'), ('؛', ','), ('《', '"'), ('》', '"'), ('？', '?'), ('！', '!'), (' ：', ':'), (' ；', ';'), ('－', '-'), ('·', ' '), ('/', ' '), ('—', ''), ('(', '~'), (')', '~'), ('...', '…'), ('\n', ', '), ('\t', ' '), ('¡', ''), ('¿', ''), ('«', '"'), ('»', '"'), ('N', 'ŋ'), ('ɫ', 'l'), ('ɚ', 'ə'), ('g', 'ɡ'), ('ε', 'e'), ('ʦ', 'ts'), ('ˤ', 'ˁ'), ('ᵻ', 'ɨ'), ('ɧ', 'ç'), ('ɥ', 'j'), ('ɬ', 's'), ('ɮ', 'z'), ('ɺ', 'ɾ'), ('ʲ', 'j'), ('ˌ', ''), ('̋', '˥'), ('́', '˦'), ('̄', '˧'), ('̀', '˨'), ('̏', '˩'), ('̂', '⭨'), ('̌', '⭧'), ('꜖', '˩'), ('꜕', '˨'), ('꜔', '˧'), ('꜓', '˦'), ('꜒', '˥'), ('"', '~'), (' - ', '~ '), ('- ', '~ '), ('-', ''), ('…', '.'), (':', '~'), (';', '~'), (',', '~')]
        unsupported_ipa_characters = {'̙', '̯', '̤', '̩', '̠', '̟', 'ꜜ', '̽', '|', '•', '↘', '‖', '‿', 'ᷝ', 'ᷠ', '̚', '↗', 'ꜛ', '̻', '̘', '͡', '̺'}
        for char in unsupported_ipa_characters:
            replacements.append((char, ''))
        if not for_feature_extraction:
            replacements = replacements + [('ˈ', ''), ('ː', ''), ('ˑ', ''), ('̆', ''), ('˥', ''), ('˦', ''), ('˧', ''), ('˨', ''), ('˩', ''), ('̌', ''), ('̂', ''), ('⭧', ''), ('⭨', ''), ('⮃', ''), ('⮁', ''), ('̃', ''), ('̧', ''), ('ʷ', ''), ('ʰ', ''), ('ˠ', ''), ('ˁ', ''), ('ˀ', ''), ('ʼ', ''), ('̹', ''), ('̞', ''), ('̪', ''), ('̬', ''), ('̝', ''), ('̰', ''), ('̈', ''), ('̜', ''), ('̥', '')]
        for replacement in replacements:
            phoneme_string = phoneme_string.replace(replacement[0], replacement[1])
        phones = re.sub('~+', '~', phoneme_string)
        phones = re.sub('\\s+', ' ', phones)
        phones = re.sub('\\.+', '.', phones)
        phones = phones.lstrip('~').rstrip('~')
        for peaking_perm in self.peaking_perms:
            phones = phones.replace(peaking_perm, '⮁'.join(peaking_perm))
        for dipping_perm in self.dipping_perms:
            phones = phones.replace(dipping_perm, '⮃'.join(dipping_perm))
        for rising_perm in self.rising_perms:
            phones = phones.replace(rising_perm, '⭧'.join(rising_perm))
        for falling_perm in self.falling_perms:
            phones = phones.replace(falling_perm, '⭨'.join(falling_perm))
        if self.add_silence_to_end:
            phones += '~'
        if include_eos_symbol:
            phones += '#'
        if not self.use_word_boundaries:
            phones = phones.replace(' ', '')
        if for_plot_labels:
            phones = phones.replace(' ', '|')
        phones = '~' + phones
        phones = re.sub('~+', '~', phones)
        return phones

    def text_vectors_to_id_sequence(self, text_vector):
        tokens = list()
        for vector in text_vector:
            if vector[get_feature_to_index_lookup()['word-boundary']] == 0:
                features = vector.cpu().numpy().tolist()
                immutable_vector = tuple(features)
                if immutable_vector in self.text_vector_to_phone_cache:
                    tokens.append(self.phone_to_id[self.text_vector_to_phone_cache[immutable_vector]])
                    continue
                features = features[13:]
                for index in range(len(features)):
                    if features[index] == 2:
                        features[index] = 0
                for phone in self.phone_to_vector:
                    if features == self.phone_to_vector[phone][13:]:
                        tokens.append(self.phone_to_id[phone])
                        self.text_vector_to_phone_cache[immutable_vector] = phone
                        break
        return tokens


class MelSpectrogram(torch.nn.Module):

    def __init__(self, fs=24000, fft_size=1536, hop_size=384, win_length=None, window='hann', num_mels=100, fmin=60, fmax=None, center=True, normalized=False, onesided=True, eps=1e-10, log_base=10.0):
        super().__init__()
        self.fft_size = fft_size
        if win_length is None:
            self.win_length = fft_size
        else:
            self.win_length = win_length
        self.hop_size = hop_size
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        if window is not None and not hasattr(torch, f'{window}_window'):
            raise ValueError(f'{window} window is not implemented')
        self.window = window
        self.eps = eps
        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        melmat = librosa.filters.mel(sr=fs, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)
        self.register_buffer('melmat', torch.from_numpy(melmat.T).float())
        self.stft_params = {'n_fft': self.fft_size, 'win_length': self.win_length, 'hop_length': self.hop_size, 'center': self.center, 'normalized': self.normalized, 'onesided': self.onesided}
        self.stft_params['return_complex'] = False
        self.log_base = log_base
        if self.log_base is None:
            self.log = torch.log
        elif self.log_base == 2.0:
            self.log = torch.log2
        elif self.log_base == 10.0:
            self.log = torch.log10
        else:
            raise ValueError(f'log_base: {log_base} is not supported.')

    def forward(self, x):
        """
        Calculate Mel-spectrogram.

        Args:
            x (Tensor): Input waveform tensor (B, T) or (B, 1, T).

        Returns:
            Tensor: Mel-spectrogram (B, #mels, #frames).
        """
        if x.dim() == 3:
            x = x.reshape(-1, x.size(2))
        if self.window is not None:
            window_func = getattr(torch, f'{self.window}_window')
            window = window_func(self.win_length, dtype=x.dtype, device=x.device)
        else:
            window = None
        x_stft = torch.stft(x, window=window, **self.stft_params)
        x_stft = x_stft.transpose(1, 2)
        x_power = x_stft[..., 0] ** 2 + x_stft[..., 1] ** 2
        x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps))
        x_mel = torch.matmul(x_amp, self.melmat)
        x_mel = torch.clamp(x_mel, min=self.eps)
        return self.log(x_mel).transpose(1, 2)


class LogMelSpec(torch.nn.Module):

    def __init__(self, sr, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = MelSpectrogram(sample_rate=sr, n_fft=1024, win_length=1024, hop_length=256, f_min=40.0, f_max=sr // 2, pad=0, n_mels=128, power=2.0, normalized=False, center=True, pad_mode='reflect', mel_scale='htk')

    def forward(self, audio):
        melspec = self.spec(audio.float())
        zero_mask = melspec == 0
        melspec[zero_mask] = 1e-08
        logmelspec = torch.log10(melspec)
        return logmelspec


class AudioPreprocessor:

    def __init__(self, input_sr, output_sr=None, cut_silence=False, do_loudnorm=False, device='cpu'):
        """
        The parameters are by default set up to do well
        on a 16kHz signal. A different sampling rate may
        require different hop_length and n_fft (e.g.
        doubling frequency --> doubling hop_length and
        doubling n_fft)
        """
        self.cut_silence = cut_silence
        self.do_loudnorm = do_loudnorm
        self.device = device
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.meter = pyln.Meter(input_sr)
        self.final_sr = input_sr
        self.wave_to_spectrogram = LogMelSpec(output_sr if output_sr is not None else input_sr)
        if cut_silence:
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            self.silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False, verbose=False)
            self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
            torch.set_grad_enabled(True)
            self.silero_model = self.silero_model
        if output_sr is not None and output_sr != input_sr:
            self.resample = Resample(orig_freq=input_sr, new_freq=output_sr)
            self.final_sr = output_sr
        else:
            self.resample = lambda x: x

    def cut_leading_and_trailing_silence(self, audio):
        """
        https://github.com/snakers4/silero-vad
        """
        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(audio, self.silero_model, sampling_rate=self.final_sr)
        try:
            result = audio[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]
            return result
        except IndexError:
            None
        return audio

    def normalize_loudness(self, audio):
        """
        normalize the amplitudes according to
        their decibels, so this should turn any
        signal with different magnitudes into
        the same magnitude by analysing loudness
        """
        try:
            loudness = self.meter.integrated_loudness(audio)
        except ValueError:
            return audio
        loud_normed = pyln.normalize.loudness(audio, loudness, -30.0)
        peak = numpy.amax(numpy.abs(loud_normed))
        peak_normed = numpy.divide(loud_normed, peak)
        return peak_normed

    def normalize_audio(self, audio):
        """
        one function to apply them all in an
        order that makes sense.
        """
        if self.do_loudnorm:
            audio = self.normalize_loudness(audio)
        audio = torch.tensor(audio, device=self.device, dtype=torch.float32)
        audio = self.resample(audio)
        if self.cut_silence:
            audio = self.cut_leading_and_trailing_silence(audio)
        return audio

    def audio_to_mel_spec_tensor(self, audio, normalize=False, explicit_sampling_rate=None):
        """
        explicit_sampling_rate is for when
        normalization has already been applied
        and that included resampling. No way
        to detect the current input_sr of the incoming
        audio
        """
        if type(audio) != torch.tensor and type(audio) != torch.Tensor:
            audio = torch.tensor(audio, device=self.device)
        if explicit_sampling_rate is None or explicit_sampling_rate == self.output_sr:
            return self.wave_to_spectrogram(audio.float())
        else:
            if explicit_sampling_rate != self.input_sr:
                None
                self.resample = Resample(orig_freq=explicit_sampling_rate, new_freq=self.output_sr)
                self.input_sr = explicit_sampling_rate
            audio = self.resample(audio.float())
            return self.wave_to_spectrogram(audio)


MODEL_DIR = 'Models/'


class StyleAdaptiveLayerNorm(nn.Module):

    def __init__(self, in_channels, cond_channels):
        """
        Style Adaptive Layer Normalization (SALN) module.

        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channels = in_channels
        self.saln = nn.Linear(cond_channels, in_channels * 2, 1)
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.saln.bias.data[:self.in_channels], 1)
        nn.init.constant_(self.saln.bias.data[self.in_channels:], 0)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.saln(c.unsqueeze(1)), chunks=2, dim=-1)
        return gamma * self.norm(x) + beta


class ConvNeXtBlock(nn.Module):

    def __init__(self, in_channels, out_channels, filter_channels, gin_channels):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels + out_channels, in_channels + out_channels, kernel_size=7, padding=3, groups=in_channels + out_channels)
        self.norm = StyleAdaptiveLayerNorm(in_channels + out_channels, gin_channels)
        self.pwconv = nn.Sequential(nn.Linear(in_channels + out_channels, filter_channels), nn.GELU(), nn.Linear(filter_channels, in_channels + out_channels))

    def forward(self, x, c, x_mask) ->torch.Tensor:
        residual = x
        x = self.dwconv(x) * x_mask
        if c is not None:
            x = self.norm(x.transpose(1, 2), c)
        else:
            x = x.transpose(1, 2)
        x = self.pwconv(x).transpose(1, 2)
        x = residual + x
        return x * x_mask


class FFN(nn.Module):

    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)
        self.act1 = nn.GELU(approximate='tanh')

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $rac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: 'int', base: 'int'=10000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\\Theta$
        """
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: 'torch.Tensor'):
        """
        Cache $\\cos$ and $\\sin$ values
        """
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        theta = 1.0 / self.base ** (torch.arange(0, self.d, 2).float() / self.d)
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: 'torch.Tensor'):
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: 'torch.Tensor'):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        x = x.permute(2, 0, 1, 3)
        self._build_cache(x)
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)
        x_rope = x_rope * self.cos_cached[:x.shape[0]] + neg_half_x * self.sin_cached[:x.shape[0]]
        return torch.cat((x_rope, x_pass), dim=-1).permute(1, 2, 0, 3)


class MultiHeadAttention(nn.Module):

    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.k_channels * 0.5)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        x = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = *key.size(), query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        query = self.query_rotary_pe(query)
        key = self.key_rotary_pe(key)
        output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=self.p_dropout if self.training else 0)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output


class DiTConVBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_channels, out_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels + out_channels, elementwise_affine=False, eps=1e-06)
        self.attn = MultiHeadAttention(hidden_channels + out_channels, hidden_channels + out_channels, num_heads, p_dropout)
        self.norm2 = nn.LayerNorm(hidden_channels + out_channels, elementwise_affine=False, eps=1e-06)
        self.mlp = FFN(hidden_channels + out_channels, hidden_channels + out_channels, filter_channels, kernel_size, p_dropout=p_dropout)
        self.adaLN_modulation = nn.Sequential(nn.Linear(gin_channels, hidden_channels + out_channels) if gin_channels != hidden_channels + out_channels else nn.Identity(), nn.SiLU(), nn.Linear(hidden_channels + out_channels, 6 * (hidden_channels + out_channels), bias=True))

    def forward(self, x, c, x_mask):
        """
        Args:
            x : [batch_size, channel, time]
            c : [batch_size, channel]
            x_mask : [batch_size, 1, time]
        return the same shape as x
        """
        x = x * x_mask
        attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1)
        if c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).unsqueeze(2).chunk(6, dim=1)
            x = x + gate_msa * self.attn(self.modulate(self.norm1(x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa), attn_mask) * x_mask
            x = x + gate_mlp * self.mlp(self.modulate(self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp), x_mask) * x_mask
        else:
            x = x + self.attn(self.norm1(x.transpose(1, 2)).transpose(1, 2), attn_mask)
            x = x + self.mlp(self.norm1(x.transpose(1, 2)).transpose(1, 2), x_mask)
        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """

    def __init__(self, in_channels, out_channels, cond_channels):
        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(cond_channels, (in_channels + out_channels) * 2, 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c.unsqueeze(2)), chunks=2, dim=1)
        return gamma * x + beta


class DitWrapper(nn.Module):
    """ add FiLM layer to condition time embedding to DiT """

    def __init__(self, hidden_channels, out_channels, filter_channels, num_heads, kernel_size=3, p_dropout=0.1, gin_channels=0, time_channels=0):
        super().__init__()
        if gin_channels is None:
            gin_channels = 0
        self.time_fusion = FiLMLayer(hidden_channels, out_channels, time_channels)
        self.conv1 = ConvNeXtBlock(hidden_channels, out_channels, filter_channels, gin_channels)
        self.conv2 = ConvNeXtBlock(hidden_channels, out_channels, filter_channels, gin_channels)
        self.conv3 = ConvNeXtBlock(hidden_channels, out_channels, filter_channels, gin_channels)
        self.block = DiTConVBlock(hidden_channels, out_channels, hidden_channels, num_heads, kernel_size, p_dropout, gin_channels)

    def forward(self, x, c, t, x_mask):
        x = self.time_fusion(x, t) * x_mask
        x = self.conv1(x, c, x_mask)
        x = self.conv2(x, c, x_mask)
        x = self.conv3(x, c, x_mask)
        x = self.block(x, c, x_mask)
        return x


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, 'SinusoidalPosEmb requires dim to be even'

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):

    def __init__(self, in_channels, out_channels, filter_channels):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_channels, filter_channels), nn.SiLU(inplace=True), nn.Linear(filter_channels, out_channels))

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):

    def __init__(self, hidden_channels, out_channels, filter_channels, dropout=0.05, n_layers=1, n_heads=4, kernel_size=3, gin_channels=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(hidden_channels, hidden_channels, filter_channels)
        self.blocks = nn.ModuleList([DitWrapper(hidden_channels, out_channels, filter_channels, n_heads, kernel_size, dropout, gin_channels, hidden_channels) for _ in range(n_layers)])
        self.final_proj = nn.Conv1d(hidden_channels + out_channels, out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, mask, mu, t, c):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            c (_type_): shape (batch_size, gin_channels)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        t = self.time_mlp(self.time_embeddings(t))
        x = torch.cat((x, mu), dim=1)
        for block in self.blocks:
            x = block(x, c, t, mask)
        output = self.final_proj(x * mask)
        return output * mask


def plot_spec_tensor(spec, save_path, name, title=None):
    fig, spec_plot_axis = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    spec_plot_axis.imshow(spec.detach().cpu().numpy(), origin='lower', cmap='GnBu')
    spec_plot_axis.yaxis.set_visible(False)
    spec_plot_axis.set_aspect('auto')
    if title is not None:
        spec_plot_axis.set_title(title)
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.95 if title is None else 0.85, wspace=0.0, hspace=0.0)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{name}.png'), dpi=100)
    plt.clf()
    plt.close()


def create_plot_of_all_solutions(sol, fps=8):
    gif_collector = list()
    for step_index, solution in enumerate(sol):
        unbatched_solution = solution[0]
        plot_spec_tensor(unbatched_solution, 'tmp', step_index, title=step_index + 1)
        gif_collector.append(imageio.v2.imread(f'tmp/{step_index}.png'))
    for _ in range(fps * 2):
        gif_collector.append(gif_collector[-1])
    imageio.mimsave('tmp/animation.gif', gif_collector, fps=fps, loop=0)


class CFMDecoder(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.gin_channels = gin_channels
        self.sigma_min = 0.0001
        self.estimator = Decoder(hidden_channels, out_channels, filter_channels, p_dropout, n_layers, n_heads, kernel_size, gin_channels)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, c=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            c (torch.Tensor, optional): shape: (batch_size, gin_channels)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        size = list(mu.size())
        size[1] = self.out_channels
        z = torch.randn(size=size) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, c=c)

    def solve_euler(self, x, t_span, mu, mask, c, plot_solutions=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.
                shape: (batch_size, gin_channels)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, c)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        if plot_solutions:
            create_plot_of_all_solutions(sol)
        return sol[-1]

    def compute_loss(self, x1, mask, mu, c):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), c), u, reduction='sum') / (torch.sum(mask) * u.shape[1])
        return loss, y


class AdaIN1d(nn.Module):
    """
    MIT Licensed

    Copyright (c) 2022 Aaron (Yinghao) Li
    https://github.com/yl4579/StyleTTS/blob/main/models.py
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma.transpose(1, 2)) * self.norm(x.transpose(1, 2)).transpose(1, 2) + beta.transpose(1, 2)


class ConditionalLayerNorm(nn.Module):

    def __init__(self, hidden_dim, speaker_embedding_dim, dim=-1):
        super(ConditionalLayerNorm, self).__init__()
        self.dim = dim
        if isinstance(hidden_dim, int):
            self.normal_shape = hidden_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.W_scale = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.normal_shape), nn.Tanh(), nn.Linear(self.normal_shape, self.normal_shape))
        self.W_bias = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.normal_shape), nn.Tanh(), nn.Linear(self.normal_shape, self.normal_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale[0].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[2].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[0].bias, 1.0)
        torch.nn.init.constant_(self.W_scale[2].bias, 1.0)
        torch.nn.init.constant_(self.W_bias[0].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[2].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[0].bias, 0.0)
        torch.nn.init.constant_(self.W_bias[2].bias, 0.0)

    def forward(self, x, speaker_embedding):
        if self.dim != -1:
            x = x.transpose(-1, self.dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y = scale.unsqueeze(1) * ((x - mean) / var) + bias.unsqueeze(1)
        if self.dim != -1:
            y = y.transpose(-1, self.dim)
        return y


class ConvolutionModule(nn.Module):
    """
    ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=bias)
        self.norm = nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm1d(channels))
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

    def forward(self, x):
        """
        Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class LayerNorm(torch.nn.LayerNorm):
    """
    Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1, eps=1e-12):
        """
        Construct an LayerNorm object.
        """
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class EncoderLayer(nn.Module):
    """
    Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(self, size, self_attn, feed_forward, feed_forward_macaron, conv_module, dropout_rate, normalize_before=True, concat_after=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)
        self.norm_mha = LayerNorm(size)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)
            self.norm_final = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """
        Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]
        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)
        if self.conv_module is not None:
            x = self.norm_final(x)
        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        if pos_emb is not None:
            return (x, pos_emb), mask
        return x, mask


class MultiLayeredConv1d(torch.nn.Module):
    """
    Multi-layered conv1d for Transformer block.

    This is a module of multi-layered conv1d designed
    to replace positionwise feed-forward network
    in Transformer block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """
        Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Conv1d(hidden_chans, in_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).

        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).
        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x).transpose(-1, 1)).transpose(-1, 1)


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """
        Construct an MultiHeadedAttention object.
        """
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """
        Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """
        Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask):
        """
        Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """
    Multi-Head Attention layer with relative position encoding.
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """
        Compute relative positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, :x.size(-1) // 2 + 1]
        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]
        return x

    def forward(self, query, key, value, pos_emb, mask):
        """
        Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """
        Construct an PositionalEncoding object.
        """
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe_positive = torch.zeros(x.size(1), self.d_model, device=x.device)
        pe_negative = torch.zeros(x.size(1), self.d_model, device=x.device)
        position = torch.arange(0, x.size(1), dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device) * -(math.log(10000.0) / self.d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe

    def forward(self, x):
        """
        Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, self.pe.size(1) // 2 - x.size(1) + 1:self.pe.size(1) // 2 + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class Swish(torch.nn.Module):
    """
    Construct a Swish activation function for Conformer.
    """

    def forward(self, x):
        """
        Return Swish activation function.
        """
        return x * torch.sigmoid(x)


def integrate_with_utt_embed(hs, utt_embeddings, projection, embedding_training):
    if not embedding_training:
        embeddings_expanded = torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = projection(torch.cat([hs, embeddings_expanded], dim=-1))
    else:
        hs = projection(hs, utt_embeddings)
    return hs


class MultiSequential(torch.nn.Sequential):
    """
    Multi-input multi-output torch.nn.Sequential.
    """

    def forward(self, *args):
        """
        Repeat.
        """
        for m in self:
            args = m(*args)
        return args


def repeat(N, fn):
    """
    Repeat module N times.

    Args:
        N (int): Number of repeat time.
        fn (Callable): Function to generate module.

    Returns:
        MultiSequential: Repeated model instance.
    """
    return MultiSequential(*[fn(n) for n in range(N)])


class Conformer(torch.nn.Module):
    """
    Conformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernel size of convolution module.

    """

    def __init__(self, conformer_type, attention_dim=256, attention_heads=4, linear_units=2048, num_blocks=6, dropout_rate=0.1, positional_dropout_rate=0.1, attention_dropout_rate=0.0, input_layer='conv2d', normalize_before=True, concat_after=False, positionwise_conv_kernel_size=1, macaron_style=False, use_cnn_module=False, cnn_module_kernel=31, zero_triu=False, utt_embed=None, lang_embs=None, lang_emb_size=16, use_output_norm=True, embedding_integration='AdaIN'):
        super(Conformer, self).__init__()
        activation = Swish()
        self.conv_subsampling_factor = 1
        self.use_output_norm = use_output_norm
        if isinstance(input_layer, torch.nn.Module):
            self.embed = input_layer
            self.art_embed_norm = LayerNorm(attention_dim)
            self.pos_enc = RelPositionalEncoding(attention_dim, positional_dropout_rate)
        elif input_layer is None:
            self.embed = None
            self.pos_enc = torch.nn.Sequential(RelPositionalEncoding(attention_dim, positional_dropout_rate))
        else:
            raise ValueError('unknown input_layer: ' + input_layer)
        if self.use_output_norm:
            self.output_norm = LayerNorm(attention_dim)
        self.utt_embed = utt_embed
        self.conformer_type = conformer_type
        self.use_conditional_layernorm_embedding_integration = embedding_integration in ['AdaIN', 'ConditionalLayerNorm']
        if utt_embed is not None:
            if conformer_type == 'encoder':
                if embedding_integration == 'AdaIN':
                    self.encoder_embedding_projection = AdaIN1d(style_dim=utt_embed, num_features=attention_dim)
                elif embedding_integration == 'ConditionalLayerNorm':
                    self.encoder_embedding_projection = ConditionalLayerNorm(speaker_embedding_dim=utt_embed, hidden_dim=attention_dim)
                else:
                    self.encoder_embedding_projection = torch.nn.Linear(attention_dim + utt_embed, attention_dim)
            elif embedding_integration == 'AdaIN':
                self.decoder_embedding_projections = repeat(num_blocks, lambda lnum: AdaIN1d(style_dim=utt_embed, num_features=attention_dim))
            elif embedding_integration == 'ConditionalLayerNorm':
                self.decoder_embedding_projections = repeat(num_blocks, lambda lnum: ConditionalLayerNorm(speaker_embedding_dim=utt_embed, hidden_dim=attention_dim))
            else:
                self.decoder_embedding_projections = repeat(num_blocks, lambda lnum: torch.nn.Linear(attention_dim + utt_embed, attention_dim))
        if lang_embs is not None:
            self.language_embedding = torch.nn.Embedding(num_embeddings=lang_embs, embedding_dim=lang_emb_size)
            if lang_emb_size == attention_dim:
                self.language_embedding_projection = lambda x: x
            else:
                self.language_embedding_projection = torch.nn.Linear(lang_emb_size, attention_dim)
            self.language_emb_norm = LayerNorm(attention_dim)
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = attention_heads, attention_dim, attention_dropout_rate, zero_triu
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate
        convolution_layer = ConvolutionModule
        convolution_layer_args = attention_dim, cnn_module_kernel, activation
        self.encoders = repeat(num_blocks, lambda lnum: EncoderLayer(attention_dim, encoder_selfattn_layer(*encoder_selfattn_layer_args), positionwise_layer(*positionwise_layer_args), positionwise_layer(*positionwise_layer_args) if macaron_style else None, convolution_layer(*convolution_layer_args) if use_cnn_module else None, dropout_rate, normalize_before, concat_after))

    def forward(self, xs, masks, utterance_embedding=None, lang_ids=None):
        """
        Encode input sequence.
        Args:
            utterance_embedding: embedding containing lots of conditioning signals
            lang_ids: ids of the languages per sample in the batch
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """
        if self.embed is not None:
            xs = self.embed(xs)
            xs = self.art_embed_norm(xs)
        if lang_ids is not None:
            lang_embs = self.language_embedding(lang_ids)
            projected_lang_embs = self.language_embedding_projection(lang_embs).unsqueeze(-1).transpose(1, 2)
            projected_lang_embs = self.language_emb_norm(projected_lang_embs)
            xs = xs + projected_lang_embs
        xs = self.pos_enc(xs)
        for encoder_index, encoder in enumerate(self.encoders):
            if self.utt_embed:
                if isinstance(xs, tuple):
                    x, pos_emb = xs[0], xs[1]
                    if self.conformer_type != 'encoder':
                        x = integrate_with_utt_embed(hs=x, utt_embeddings=utterance_embedding, projection=self.decoder_embedding_projections[encoder_index], embedding_training=self.use_conditional_layernorm_embedding_integration)
                    xs = x, pos_emb
                elif self.conformer_type != 'encoder':
                    xs = integrate_with_utt_embed(hs=xs, utt_embeddings=utterance_embedding, projection=self.decoder_embedding_projections[encoder_index], embedding_training=self.use_conditional_layernorm_embedding_integration)
            xs, masks = encoder(xs, masks)
        if isinstance(xs, tuple):
            xs = xs[0]
        if self.utt_embed and self.conformer_type == 'encoder':
            xs = integrate_with_utt_embed(hs=xs, utt_embeddings=utterance_embedding, projection=self.encoder_embedding_projection, embedding_training=self.use_conditional_layernorm_embedding_integration)
        elif self.use_output_norm:
            xs = self.output_norm(xs)
        return xs, masks


def pad_list(xs, pad_value):
    """
    Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


class LengthRegulator(torch.nn.Module, ABC):
    """
    Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """
        Initialize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.
        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, xs, ds, alpha=1.0):
        """
        Calculate forward propagation.
        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            alpha (float, optional): Alpha value to control speed of speech.
        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).
        """
        if alpha != 1.0:
            assert alpha > 0
            ds = torch.round(ds.float() * alpha).long()
        if ds.sum() == 0:
            ds[ds.sum(dim=1).eq(0)] = 1
        return pad_list([self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)], self.pad_value)

    def _repeat_one_sequence(self, x, d):
        """
        Repeat each frame according to duration
        """
        d = torch.clamp(d, min=0)
        return torch.repeat_interleave(x, d, dim=0)


def make_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    if length_dim == 0:
        raise ValueError('length_dim cannot be 0: {}'.format(length_dim))
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)
    if device is not None:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=device)
    else:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)
        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    return ~make_pad_mask(lengths, xs, length_dim, device=device)


class StochasticToucanTTSLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_criterion = torch.nn.L1Loss(reduction='none')

    def forward(self, predicted_features, gold_features, features_lengths):
        """
        Args:
            predicted_features (Tensor): Batch of outputs (B, Lmax, odim).
            gold_features (Tensor): Batch of target features (B, Lmax, odim).
            features_lengths (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
        """
        l1_loss = self.l1_criterion(predicted_features, gold_features)
        out_masks = make_non_pad_mask(features_lengths).unsqueeze(-1)
        out_masks = torch.nn.functional.pad(out_masks.transpose(1, 2), [0, gold_features.size(1) - out_masks.size(1), 0, 0, 0, 0], value=False).transpose(1, 2)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= gold_features.size(0) * gold_features.size(2)
        l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
        return l1_loss


def initialize(model, init):
    """
    Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Args:
        model: Target.
        init: Method of initialization.
    """
    for p in model.parameters():
        if p.dim() > 1:
            if init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(p.data)
            elif init == 'xavier_normal':
                torch.nn.init.xavier_normal_(p.data)
            elif init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity='relu')
            elif init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(p.data, nonlinearity='relu')
            else:
                raise ValueError('Unknown initialization: ' + init)
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.LayerNorm, Modules.GeneralLayers.ConditionalLayerNorm.ConditionalLayerNorm, Modules.GeneralLayers.ConditionalLayerNorm.SequentialWrappableConditionalLayerNorm)):
            m.reset_parameters()


class ToucanTTS(torch.nn.Module):
    """
    ToucanTTS module, which is based on a FastSpeech 2 module,
    but with lots of designs from different architectures accumulated
    and some major components added to put a large focus on
    multilinguality and controllability.

    Contributions inspired from elsewhere:
    - The Decoder is a flow matching network, like in Matcha-TTS and StableTTS
    - Pitch and energy values are averaged per-phone, as in FastPitch to enable great controllability
    - The encoder and decoder are Conformers, like in ESPnet

    """

    def __init__(self, input_feature_dimensions=64, spec_channels=128, attention_dimension=384, attention_heads=4, positionwise_conv_kernel_size=1, use_scaled_positional_encoding=True, init_type='xavier_uniform', use_macaron_style_in_conformer=True, use_cnn_in_conformer=True, encoder_layers=6, encoder_units=1536, encoder_normalize_before=True, encoder_concat_after=False, conformer_encoder_kernel_size=7, transformer_enc_dropout_rate=0.1, transformer_enc_positional_dropout_rate=0.1, transformer_enc_attn_dropout_rate=0.1, decoder_layers=6, decoder_units=1536, decoder_concat_after=False, conformer_decoder_kernel_size=31, decoder_normalize_before=True, transformer_dec_dropout_rate=0.1, transformer_dec_positional_dropout_rate=0.1, transformer_dec_attn_dropout_rate=0.1, prosody_channels=8, duration_predictor_layers=3, duration_predictor_kernel_size=5, duration_predictor_dropout_rate=0.2, pitch_predictor_layers=3, pitch_predictor_kernel_size=5, pitch_predictor_dropout=0.2, pitch_embed_kernel_size=1, pitch_embed_dropout=0.0, energy_predictor_layers=2, energy_predictor_kernel_size=3, energy_predictor_dropout=0.2, energy_embed_kernel_size=1, energy_embed_dropout=0.0, cfm_filter_channels=256, cfm_heads=4, cfm_layers=3, cfm_kernel_size=5, cfm_p_dropout=0.1, utt_embed_dim=192, lang_embs=8000, lang_emb_size=32, integrate_language_embedding_into_encoder_out=True, embedding_integration='AdaIN'):
        super().__init__()
        self.config = {'input_feature_dimensions': input_feature_dimensions, 'attention_dimension': attention_dimension, 'attention_heads': attention_heads, 'positionwise_conv_kernel_size': positionwise_conv_kernel_size, 'use_scaled_positional_encoding': use_scaled_positional_encoding, 'init_type': init_type, 'use_macaron_style_in_conformer': use_macaron_style_in_conformer, 'use_cnn_in_conformer': use_cnn_in_conformer, 'encoder_layers': encoder_layers, 'encoder_units': encoder_units, 'encoder_normalize_before': encoder_normalize_before, 'encoder_concat_after': encoder_concat_after, 'conformer_encoder_kernel_size': conformer_encoder_kernel_size, 'transformer_enc_dropout_rate': transformer_enc_dropout_rate, 'transformer_enc_positional_dropout_rate': transformer_enc_positional_dropout_rate, 'transformer_enc_attn_dropout_rate': transformer_enc_attn_dropout_rate, 'decoder_layers': decoder_layers, 'decoder_units': decoder_units, 'decoder_concat_after': decoder_concat_after, 'conformer_decoder_kernel_size': conformer_decoder_kernel_size, 'decoder_normalize_before': decoder_normalize_before, 'transformer_dec_dropout_rate': transformer_dec_dropout_rate, 'transformer_dec_positional_dropout_rate': transformer_dec_positional_dropout_rate, 'transformer_dec_attn_dropout_rate': transformer_dec_attn_dropout_rate, 'duration_predictor_layers': duration_predictor_layers, 'duration_predictor_kernel_size': duration_predictor_kernel_size, 'duration_predictor_dropout_rate': duration_predictor_dropout_rate, 'pitch_predictor_layers': pitch_predictor_layers, 'pitch_predictor_kernel_size': pitch_predictor_kernel_size, 'pitch_predictor_dropout': pitch_predictor_dropout, 'pitch_embed_kernel_size': pitch_embed_kernel_size, 'pitch_embed_dropout': pitch_embed_dropout, 'energy_predictor_layers': energy_predictor_layers, 'energy_predictor_kernel_size': energy_predictor_kernel_size, 'energy_predictor_dropout': energy_predictor_dropout, 'energy_embed_kernel_size': energy_embed_kernel_size, 'energy_embed_dropout': energy_embed_dropout, 'spec_channels': spec_channels, 'cfm_filter_channels': cfm_filter_channels, 'prosody_channels': prosody_channels, 'cfm_heads': cfm_heads, 'cfm_layers': cfm_layers, 'cfm_kernel_size': cfm_kernel_size, 'cfm_p_dropout': cfm_p_dropout, 'utt_embed_dim': utt_embed_dim, 'lang_embs': lang_embs, 'lang_emb_size': lang_emb_size, 'embedding_integration': embedding_integration, 'integrate_language_embedding_into_encoder_out': integrate_language_embedding_into_encoder_out}
        if lang_embs is None or lang_embs == 0:
            lang_embs = None
            integrate_language_embedding_into_encoder_out = False
        if integrate_language_embedding_into_encoder_out:
            utt_embed_dim = utt_embed_dim + lang_emb_size
        self.input_feature_dimensions = input_feature_dimensions
        self.attention_dimension = attention_dimension
        self.use_scaled_pos_enc = use_scaled_positional_encoding
        self.multilingual_model = lang_embs is not None
        self.multispeaker_model = utt_embed_dim is not None
        self.integrate_language_embedding_into_encoder_out = integrate_language_embedding_into_encoder_out
        self.use_conditional_layernorm_embedding_integration = embedding_integration in ['AdaIN', 'ConditionalLayerNorm']
        articulatory_feature_embedding = Sequential(Linear(input_feature_dimensions, 100), Tanh(), Linear(100, attention_dimension))
        self.encoder = Conformer(conformer_type='encoder', attention_dim=attention_dimension, attention_heads=attention_heads, linear_units=encoder_units, num_blocks=encoder_layers, input_layer=articulatory_feature_embedding, dropout_rate=transformer_enc_dropout_rate, positional_dropout_rate=transformer_enc_positional_dropout_rate, attention_dropout_rate=transformer_enc_attn_dropout_rate, normalize_before=encoder_normalize_before, concat_after=encoder_concat_after, positionwise_conv_kernel_size=positionwise_conv_kernel_size, macaron_style=use_macaron_style_in_conformer, use_cnn_module=True, cnn_module_kernel=conformer_encoder_kernel_size, zero_triu=False, utt_embed=utt_embed_dim, lang_embs=lang_embs, lang_emb_size=lang_emb_size, use_output_norm=True, embedding_integration=embedding_integration)
        self.pitch_embed = Sequential(torch.nn.Conv1d(in_channels=1, out_channels=attention_dimension, kernel_size=pitch_embed_kernel_size, padding=(pitch_embed_kernel_size - 1) // 2), torch.nn.Dropout(pitch_embed_dropout))
        self.energy_embed = Sequential(torch.nn.Conv1d(in_channels=1, out_channels=attention_dimension, kernel_size=energy_embed_kernel_size, padding=(energy_embed_kernel_size - 1) // 2), torch.nn.Dropout(energy_embed_dropout))
        self.length_regulator = LengthRegulator()
        self.decoder = Conformer(conformer_type='decoder', attention_dim=attention_dimension, attention_heads=attention_heads, linear_units=decoder_units, num_blocks=decoder_layers, input_layer=None, dropout_rate=transformer_dec_dropout_rate, positional_dropout_rate=transformer_dec_positional_dropout_rate, attention_dropout_rate=transformer_dec_attn_dropout_rate, normalize_before=decoder_normalize_before, concat_after=decoder_concat_after, positionwise_conv_kernel_size=positionwise_conv_kernel_size, macaron_style=use_macaron_style_in_conformer, use_cnn_module=use_cnn_in_conformer, cnn_module_kernel=conformer_decoder_kernel_size, use_output_norm=embedding_integration not in ['AdaIN', 'ConditionalLayerNorm'], utt_embed=utt_embed_dim, embedding_integration=embedding_integration)
        self.output_projection = torch.nn.Linear(attention_dimension, spec_channels)
        self.pitch_latent_reduction = torch.nn.Linear(attention_dimension, prosody_channels)
        self.energy_latent_reduction = torch.nn.Linear(attention_dimension, prosody_channels)
        self.duration_latent_reduction = torch.nn.Linear(attention_dimension, prosody_channels)
        self._reset_parameters(init_type=init_type)
        if lang_embs is not None:
            torch.nn.init.normal_(self.encoder.language_embedding.weight, mean=0, std=attention_dimension ** -0.5)
        self.duration_predictor = CFMDecoder(hidden_channels=prosody_channels, out_channels=1, filter_channels=prosody_channels, n_heads=1, n_layers=duration_predictor_layers, kernel_size=duration_predictor_kernel_size, p_dropout=duration_predictor_dropout_rate, gin_channels=utt_embed_dim)
        self.pitch_predictor = CFMDecoder(hidden_channels=prosody_channels, out_channels=1, filter_channels=prosody_channels, n_heads=1, n_layers=pitch_predictor_layers, kernel_size=pitch_predictor_kernel_size, p_dropout=pitch_predictor_dropout, gin_channels=utt_embed_dim)
        self.energy_predictor = CFMDecoder(hidden_channels=prosody_channels, out_channels=1, filter_channels=prosody_channels, n_heads=1, n_layers=energy_predictor_layers, kernel_size=energy_predictor_kernel_size, p_dropout=energy_predictor_dropout, gin_channels=utt_embed_dim)
        self.flow_matching_decoder = CFMDecoder(hidden_channels=spec_channels, out_channels=spec_channels, filter_channels=cfm_filter_channels, n_heads=cfm_heads, n_layers=cfm_layers, kernel_size=cfm_kernel_size, p_dropout=cfm_p_dropout, gin_channels=utt_embed_dim)
        self.criterion = StochasticToucanTTSLoss()

    def forward(self, text_tensors, text_lengths, gold_speech, speech_lengths, gold_durations, gold_pitch, gold_energy, utterance_embedding, return_feats=False, lang_ids=None, run_stochastic=True):
        """
        Args:
            return_feats (Boolean): whether to return the predicted spectrogram
            text_tensors (LongTensor): Batch of padded text vectors (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            gold_speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of padded durations (B, Tmax + 1).
            gold_pitch (Tensor): Batch of padded token-averaged pitch (B, Tmax + 1, 1).
            gold_energy (Tensor): Batch of padded token-averaged energy (B, Tmax + 1, 1).
            lang_ids (LongTensor): The language IDs used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Batch of embeddings to condition the TTS on, if the model is multispeaker
            run_stochastic (Bool): Whether to detach the inputs to the normalizing flow for stability.
        """
        outs, stochastic_loss, duration_loss, pitch_loss, energy_loss = self._forward(text_tensors=text_tensors, text_lengths=text_lengths, gold_speech=gold_speech, speech_lengths=speech_lengths, gold_durations=gold_durations, gold_pitch=gold_pitch, gold_energy=gold_energy, utterance_embedding=utterance_embedding, is_inference=False, lang_ids=lang_ids, run_stochastic=run_stochastic)
        regression_loss = self.criterion(predicted_features=outs, gold_features=gold_speech, features_lengths=speech_lengths)
        if return_feats:
            return regression_loss, stochastic_loss, duration_loss, pitch_loss, energy_loss, outs
        return regression_loss, stochastic_loss, duration_loss, pitch_loss, energy_loss

    def _forward(self, text_tensors, text_lengths, gold_speech=None, speech_lengths=None, gold_durations=None, gold_pitch=None, gold_energy=None, is_inference=False, utterance_embedding=None, lang_ids=None, run_stochastic=False):
        text_tensors = torch.clamp(text_tensors, max=1.0)
        if not self.multilingual_model:
            lang_ids = None
        if not self.multispeaker_model:
            utterance_embedding = None
        if utterance_embedding is not None:
            utterance_embedding = torch.nn.functional.normalize(utterance_embedding)
            if self.integrate_language_embedding_into_encoder_out and lang_ids is not None:
                lang_embs = self.encoder.language_embedding(lang_ids)
                lang_embs = torch.nn.functional.normalize(lang_embs)
                utterance_embedding = torch.cat([lang_embs, utterance_embedding], dim=1).detach()
        text_masks = make_non_pad_mask(text_lengths, device=text_lengths.device).unsqueeze(-2)
        encoded_texts, _ = self.encoder(text_tensors, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids)
        if is_inference:
            reduced_pitch_space = torchfunc.dropout(self.pitch_latent_reduction(encoded_texts), p=0.1).transpose(1, 2)
            pitch_predictions = self.pitch_predictor(mu=reduced_pitch_space, mask=text_masks.float(), n_timesteps=10, temperature=1.0, c=utterance_embedding)
            embedded_pitch_curve = self.pitch_embed(pitch_predictions).transpose(1, 2)
            reduced_energy_space = torchfunc.dropout(self.energy_latent_reduction(encoded_texts + embedded_pitch_curve), p=0.1).transpose(1, 2)
            energy_predictions = self.energy_predictor(mu=reduced_energy_space, mask=text_masks.float(), n_timesteps=10, temperature=1.0, c=utterance_embedding)
            embedded_energy_curve = self.energy_embed(energy_predictions).transpose(1, 2)
            reduced_duration_space = torchfunc.dropout(self.duration_latent_reduction(encoded_texts + embedded_pitch_curve + embedded_energy_curve), p=0.1).transpose(1, 2)
            predicted_durations = self.duration_predictor(mu=reduced_duration_space, mask=text_masks.float(), n_timesteps=10, temperature=1.0, c=utterance_embedding)
            predicted_durations = torch.clamp(torch.ceil(predicted_durations), min=0.0).long().squeeze(1)
            for phoneme_index, phoneme_vector in enumerate(text_tensors.squeeze(0)):
                if phoneme_vector[get_feature_to_index_lookup()['word-boundary']] == 1:
                    predicted_durations[0][phoneme_index] = 0
            enriched_encoded_texts = encoded_texts + embedded_pitch_curve + embedded_energy_curve
            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, predicted_durations)
        else:
            reduced_pitch_space = torchfunc.dropout(self.pitch_latent_reduction(encoded_texts), p=0.1).transpose(1, 2)
            pitch_loss, _ = self.pitch_predictor.compute_loss(mu=reduced_pitch_space, x1=gold_pitch.transpose(1, 2), mask=text_masks.float(), c=utterance_embedding)
            embedded_pitch_curve = self.pitch_embed(gold_pitch.transpose(1, 2)).transpose(1, 2)
            reduced_energy_space = torchfunc.dropout(self.energy_latent_reduction(encoded_texts + embedded_pitch_curve), p=0.1).transpose(1, 2)
            energy_loss, _ = self.energy_predictor.compute_loss(mu=reduced_energy_space, x1=gold_energy.transpose(1, 2), mask=text_masks.float(), c=utterance_embedding)
            embedded_energy_curve = self.energy_embed(gold_energy.transpose(1, 2)).transpose(1, 2)
            reduced_duration_space = torchfunc.dropout(self.duration_latent_reduction(encoded_texts + embedded_pitch_curve + embedded_energy_curve), p=0.1).transpose(1, 2)
            duration_loss, _ = self.duration_predictor.compute_loss(mu=reduced_duration_space, x1=gold_durations.unsqueeze(-1).transpose(1, 2).float(), mask=text_masks.float(), c=utterance_embedding)
            enriched_encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
            upsampled_enriched_encoded_texts = self.length_regulator(enriched_encoded_texts, gold_durations)
        decoder_masks = make_non_pad_mask(speech_lengths, device=speech_lengths.device).unsqueeze(-2) if speech_lengths is not None and not is_inference else None
        decoded_speech, _ = self.decoder(upsampled_enriched_encoded_texts, decoder_masks, utterance_embedding=utterance_embedding)
        preliminary_spectrogram = self.output_projection(decoded_speech)
        if is_inference:
            if run_stochastic:
                refined_codec_frames = self.flow_matching_decoder(mu=preliminary_spectrogram.transpose(1, 2), mask=make_non_pad_mask([len(decoded_speech[0])], device=decoded_speech.device).unsqueeze(-2).float(), n_timesteps=15, temperature=0.2, c=None).transpose(1, 2)
            else:
                refined_codec_frames = preliminary_spectrogram
            return refined_codec_frames, predicted_durations.squeeze(), pitch_predictions.squeeze(), energy_predictions.squeeze()
        else:
            if run_stochastic:
                stochastic_loss, _ = self.flow_matching_decoder.compute_loss(x1=gold_speech.transpose(1, 2), mask=decoder_masks.float(), mu=preliminary_spectrogram.transpose(1, 2).detach(), c=None)
            else:
                stochastic_loss = None
            return preliminary_spectrogram, stochastic_loss, duration_loss, pitch_loss, energy_loss

    @torch.inference_mode()
    def inference(self, text, speech=None, utterance_embedding=None, return_duration_pitch_energy=False, lang_id=None, run_stochastic=True):
        """
        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            return_duration_pitch_energy (Boolean): whether to return the list of predicted durations for nicer plotting
            lang_id (LongTensor): The language ID used to access the language embedding table, if the model is multilingual
            utterance_embedding (Tensor): Embedding to condition the TTS on, if the model is multispeaker
            run_stochastic (bool): whether to use the output of the stochastic or of the out_projection to generate codec frames
        """
        self.eval()
        ilens = torch.tensor([text.shape[0]], dtype=torch.long, device=text.device)
        text_pseudobatched, speech_pseudobatched = text.unsqueeze(0), None
        if speech is not None:
            speech_pseudobatched = speech.unsqueeze(0)
        utterance_embeddings = utterance_embedding.unsqueeze(0) if utterance_embedding is not None else None
        outs, duration_predictions, pitch_predictions, energy_predictions = self._forward(text_pseudobatched, ilens, speech_pseudobatched, is_inference=True, utterance_embedding=utterance_embeddings, lang_ids=lang_id, run_stochastic=run_stochastic)
        self.train()
        if return_duration_pitch_energy:
            return outs.squeeze().transpose(0, 1), duration_predictions, pitch_predictions, energy_predictions
        return outs.squeeze().transpose(0, 1)

    def _reset_parameters(self, init_type='xavier_uniform'):
        if init_type != 'pytorch':
            initialize(self, init_type)

    def reset_postnet(self, init_type='xavier_uniform'):
        initialize(self.flow_matching_decoder, init_type)


def float2pcm(sig, dtype='int16'):
    """
    https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def load_json_from_path(path):
    with open(path, 'r', encoding='utf8') as f:
        obj = json.loads(f.read())
    return obj


def get_language_id(language):
    try:
        iso_codes_to_ids = load_json_from_path('Preprocessing/multilinguality/iso_lookup.json')[-1]
    except FileNotFoundError:
        try:
            iso_codes_to_ids = load_json_from_path(str(Path(__file__).parent / 'multilinguality/iso_lookup.json'))[-1]
        except FileNotFoundError:
            iso_codes_to_ids = load_json_from_path('iso_lookup.json')[-1]
    if language not in iso_codes_to_ids:
        None
        return None
    return torch.LongTensor([iso_codes_to_ids[language]])


class BatchNormConv(torch.nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int'):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bnorm = torch.nn.SyncBatchNorm.convert_sync_batchnorm(torch.nn.BatchNorm1d(out_channels))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bnorm(x)
        x = x.transpose(1, 2)
        return x


def binarize_alignment(alignment_prob):
    """
    # Implementation by:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/alignment.py
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/attn_loss_function.py

    Binarizes alignment with MAS.
    """
    opt = np.zeros_like(alignment_prob)
    alignment_prob = alignment_prob + (np.abs(alignment_prob).max() + 1.0)
    alignment_prob * alignment_prob * (1.0 / alignment_prob.max())
    attn_map = np.log(alignment_prob)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):
            prev_log = log_p[i - 1, j]
            prev_j = j
            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1
            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


class Aligner(torch.nn.Module):

    def __init__(self, n_features=128, num_symbols=145, conv_dim=512, lstm_dim=512):
        super().__init__()
        self.convs = torch.nn.ModuleList([BatchNormConv(n_features, conv_dim, 3), torch.nn.Dropout(p=0.5), BatchNormConv(conv_dim, conv_dim, 3), torch.nn.Dropout(p=0.5), BatchNormConv(conv_dim, conv_dim, 3), torch.nn.Dropout(p=0.5), BatchNormConv(conv_dim, conv_dim, 3), torch.nn.Dropout(p=0.5), BatchNormConv(conv_dim, conv_dim, 3), torch.nn.Dropout(p=0.5)])
        self.rnn1 = torch.nn.LSTM(conv_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.rnn2 = torch.nn.LSTM(2 * lstm_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(2 * lstm_dim, num_symbols)
        self.tf = ArticulatoryCombinedTextFrontend(language='eng')
        self.ctc_loss = CTCLoss(blank=144, zero_infinity=True)
        self.vector_to_id = dict()

    def forward(self, x, lens=None):
        for conv in self.convs:
            x = conv(x)
        if lens is not None:
            x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        if lens is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.proj(x)
        if lens is not None:
            out_masks = make_non_pad_mask(lens).unsqueeze(-1)
            x = x * out_masks.float()
        return x

    @torch.inference_mode()
    def inference(self, features, tokens, save_img_for_debug=None, train=False, pathfinding='MAS', return_ctc=False):
        if not train:
            tokens_indexed = self.tf.text_vectors_to_id_sequence(text_vector=tokens)
            tokens = np.asarray(tokens_indexed)
        else:
            tokens = tokens.cpu().detach().numpy()
        pred = self(features.unsqueeze(0))
        if return_ctc:
            ctc_loss = self.ctc_loss(pred.transpose(0, 1).log_softmax(2), torch.LongTensor(tokens), torch.LongTensor([len(pred[0])]), torch.LongTensor([len(tokens)])).item()
        pred = pred.squeeze().cpu().detach().numpy()
        pred_max = pred[:, tokens]
        alignment_matrix = binarize_alignment(pred_max)
        if save_img_for_debug is not None:
            phones = list()
            for index in tokens:
                phones.append(self.tf.id_to_phone[index])
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
            ax.imshow(alignment_matrix, interpolation='nearest', aspect='auto', origin='lower', cmap='cividis')
            ax.set_ylabel('Mel-Frames')
            ax.set_xticks(range(len(pred_max[0])))
            ax.set_xticklabels(labels=phones)
            ax.set_title('MAS Path')
            plt.tight_layout()
            fig.savefig(save_img_for_debug)
            fig.clf()
            plt.close()
        if return_ctc:
            return alignment_matrix, ctc_loss
        return alignment_matrix


class Reconstructor(torch.nn.Module):

    def __init__(self, n_features=128, num_symbols=145, speaker_embedding_dim=192, hidden_dim=256):
        super().__init__()
        self.in_proj = torch.nn.Linear(num_symbols + speaker_embedding_dim, hidden_dim)
        self.hidden_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Linear(hidden_dim, n_features)
        self.l1_criterion = torch.nn.L1Loss(reduction='none')

    def forward(self, x, lens, ys):
        x = self.in_proj(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.hidden_proj(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.out_proj(x)
        out_masks = make_non_pad_mask(lens).unsqueeze(-1)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= ys.size(0) * ys.size(2)
        return self.l1_criterion(x, ys).mul(out_weights).masked_select(out_masks).sum()


class ResNetBlock(nn.Module):

    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = fin != fout
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.fhidden)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s


class ResNet_G(nn.Module):

    def __init__(self, data_dim, z_dim, size, nfilter=64, nfilter_max=512, bn=True, res_ratio=0.1, **kwargs):
        super().__init__()
        self.input_dim = z_dim
        self.output_dim = z_dim
        self.dropout_rate = 0
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.bn = bn
        self.z_dim = z_dim
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** (nlayers + 1))
        self.fc = nn.Linear(z_dim, self.nf0 * s0 * s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2 ** i, nf_max)
            blocks += [ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio), nn.Upsample(scale_factor=2)]
        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio), ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)]
        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
        self.fc_out = nn.Linear(3 * size * size, data_dim)

    def forward(self, z, return_intermediate=False):
        batch_size = z.size(0)
        out = self.fc(z)
        if self.bn:
            out = self.bn1d(out)
        out = self.relu(out)
        if return_intermediate:
            l_1 = out.detach().clone()
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(out)
        out = self.relu(out)
        out.flatten(1)
        out = self.fc_out(out.flatten(1))
        if return_intermediate:
            return out, l_1
        return out

    def sample_latent(self, n_samples, z_size, temperature):
        return torch.randn((n_samples, z_size)) * temperature


class ResNet_D(nn.Module):

    def __init__(self, data_dim, size, nfilter=64, nfilter_max=512, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.size = size
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)
        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio), ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)]
        self.fc_input = nn.Linear(data_dim, 3 * size * size)
        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [nn.AvgPool2d(3, stride=2, padding=1), ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)]
        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.fc_input(x)
        out = self.relu(out).view(batch_size, 3, self.size, self.size)
        out = self.relu(self.conv_img(out))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)
        return out


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module.
    This module is reference encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.
    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        idim (int, optional): Dimension of the input features.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the reference encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gst_layers (int, optional): The number of GRU layers in the reference encoder.
        gst_units (int, optional): The number of GRU units in the reference encoder.
    """

    def __init__(self, idim=80, conv_layers: 'int'=6, conv_chans_list=(32, 32, 64, 64, 128, 128), conv_kernel_size: 'int'=3, conv_stride: 'int'=2, gst_layers: 'int'=1, gst_units: 'int'=128):
        """Initialize reference encoder module."""
        super(ReferenceEncoder, self).__init__()
        assert conv_kernel_size % 2 == 1, 'kernel size must be odd.'
        assert len(conv_chans_list) == conv_layers, 'the number of conv layers and length of channels list must be the same.'
        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [torch.nn.Conv2d(conv_in_chans, conv_out_chans, kernel_size=conv_kernel_size, stride=conv_stride, padding=padding, bias=False), torch.nn.BatchNorm2d(conv_out_chans), torch.nn.ReLU(inplace=True)]
        self.convs = torch.nn.Sequential(*convs)
        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding
        gst_in_units = idim
        for i in range(conv_layers):
            gst_in_units = (gst_in_units - conv_kernel_size + 2 * padding) // conv_stride + 1
        gst_in_units *= conv_out_chans
        self.gst = torch.nn.GRU(gst_in_units, gst_units, gst_layers, batch_first=True)

    def forward(self, speech):
        """Calculate forward propagation.
        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).
        Returns:
            Tensor: Reference embedding (B, gst_units)
        """
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)
        hs = self.convs(xs).transpose(1, 2)
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)
        self.gst.flatten_parameters()
        _, ref_embs = self.gst(hs)
        ref_embs = ref_embs[-1]
        return ref_embs


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module.
    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.
    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.
    """

    def __init__(self, ref_embed_dim: 'int'=128, gst_tokens: 'int'=10, gst_token_dim: 'int'=128, gst_heads: 'int'=4, dropout_rate: 'float'=0.0):
        """Initialize style token layer module."""
        super(StyleTokenLayer, self).__init__()
        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter('gst_embs', torch.nn.Parameter(gst_embs))
        self.mha = MultiHeadedAttention(q_dim=ref_embed_dim, k_dim=gst_token_dim // gst_heads, v_dim=gst_token_dim // gst_heads, n_head=gst_heads, n_feat=gst_token_dim, dropout_rate=dropout_rate)

    def forward(self, ref_embs):
        """Calculate forward propagation.
        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim).
        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).
        """
        batch_size = ref_embs.size(0)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        ref_embs = ref_embs.unsqueeze(1)
        style_embs = self.mha(ref_embs, gst_embs, gst_embs, None)
        return style_embs.squeeze(1)


class GSTStyleEncoder(torch.nn.Module):
    """Style encoder.
    This module is style encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.
    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        idim (int, optional): Dimension of the input features.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the reference encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gst_layers (int, optional): The number of GRU layers in the reference encoder.
        gst_units (int, optional): The number of GRU units in the reference encoder.
    """

    def __init__(self, idim: 'int'=128, gst_tokens: 'int'=512, gst_token_dim: 'int'=64, gst_heads: 'int'=8, conv_layers: 'int'=8, conv_chans_list=(32, 32, 64, 64, 128, 128, 256, 256), conv_kernel_size: 'int'=3, conv_stride: 'int'=2, gst_layers: 'int'=2, gst_units: 'int'=256):
        """Initialize global style encoder module."""
        super(GSTStyleEncoder, self).__init__()
        self.num_tokens = gst_tokens
        self.ref_enc = ReferenceEncoder(idim=idim, conv_layers=conv_layers, conv_chans_list=conv_chans_list, conv_kernel_size=conv_kernel_size, conv_stride=conv_stride, gst_layers=gst_layers, gst_units=gst_units)
        self.stl = StyleTokenLayer(ref_embed_dim=gst_units, gst_tokens=gst_tokens, gst_token_dim=gst_token_dim, gst_heads=gst_heads)

    def forward(self, speech):
        """Calculate forward propagation.
        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
        Returns:
            Tensor: Style token embeddings (B, token_dim).
        """
        ref_embs = self.ref_enc(speech)
        style_embs = self.stl(ref_embs)
        return style_embs

    def calculate_ada4_regularization_loss(self):
        losses = list()
        for emb1_index in range(self.num_tokens):
            for emb2_index in range(emb1_index + 1, self.num_tokens):
                if emb1_index != emb2_index:
                    losses.append(torch.nn.functional.cosine_similarity(self.stl.gst_embs[emb1_index], self.stl.gst_embs[emb2_index], dim=0))
        return sum(losses)


class StyleEmbedding(torch.nn.Module):
    """
    The style embedding should provide information of the speaker and their speaking style

    The feedback signal for the module will come from the TTS objective, so it doesn't have a dedicated train loop.
    The train loop does however supply supervision in the form of a barlow twins objective.

    See the git history for some other approaches for style embedding, like the SWIN transformer
    and a simple LSTM baseline. GST turned out to be the best.
    """

    def __init__(self, embedding_dim=16, style_tts_encoder=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_gst = not style_tts_encoder
        if style_tts_encoder:
            self.style_encoder = StyleTTSEncoder(style_dim=embedding_dim)
        else:
            self.style_encoder = GSTStyleEncoder(gst_token_dim=embedding_dim)

    def forward(self, batch_of_feature_sequences, batch_of_feature_sequence_lengths):
        """
        Args:
            batch_of_feature_sequences: b is the batch axis, 128 features per timestep
                                   and l time-steps, which may include padding
                                   for most elements in the batch (b, l, 128)
            batch_of_feature_sequence_lengths: indicate for every element in the batch,
                                          what the true length is, since they are
                                          all padded to the length of the longest
                                          element in the batch (b, 1)
        Returns:
            batch of n dimensional embeddings (b,n)
        """
        minimum_sequence_length = 512
        specs = list()
        for index, spec_length in enumerate(batch_of_feature_sequence_lengths):
            spec = batch_of_feature_sequences[index][:spec_length]
            spec = spec.repeat((2, 1))
            current_spec_length = len(spec)
            while current_spec_length < minimum_sequence_length:
                spec = spec.repeat((2, 1))
                current_spec_length = len(spec)
            specs.append(spec[:minimum_sequence_length])
        spec_batch = torch.stack(specs, dim=0)
        return self.style_encoder(speech=spec_batch)


class DownSample(nn.Module):

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class LearnedDownSample(nn.Module):

    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)


class ResBlk(nn.Module):

    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class StyleEncoder(nn.Module):

    def __init__(self, dim_in=128, style_dim=64, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, speech):
        h = self.shared(speech.unsqueeze(1))
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        return s


class LearnedUpSample(nn.Module):

    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, output_padding=(1, 0), padding=(1, 0))
        elif self.layer_type == 'half':
            self.conv = nn.ConvTranspose2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, output_padding=1, padding=1)
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):

    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class GuidedAttentionLoss(torch.nn.Module):
    """
    Guided attention loss function module.

    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.

    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.4, alpha=1.0):
        """
        Initialize guided attention loss module.

        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """
        Calculate forward propagation.

        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).

        Returns:
            Tensor: Guided attention loss value.
        """
        self._reset_masks()
        self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens)
        self.masks = self._make_masks(ilens, olens)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen), device=ilens.device)
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma)
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """
        Make guided attention mask.
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen, device=olen.device).float(), torch.arange(ilen, device=ilen.device).float())
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * sigma ** 2))

    @staticmethod
    def _make_masks(ilens, olens):
        """
        Make masks indicating non-padded part.

        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        """
        in_masks = make_non_pad_mask(ilens, device=ilens.device)
        out_masks = make_non_pad_mask(olens, device=olens.device)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)


class GuidedMultiHeadAttentionLoss(GuidedAttentionLoss):
    """
    Guided attention loss function module for multi head attention.

    Args:
        sigma (float, optional): Standard deviation to control
        how close attention to a diagonal.
        alpha (float, optional): Scaling coefficient (lambda).
        reset_always (bool, optional): Whether to always reset masks.
    """

    def forward(self, att_ws, ilens, olens):
        """
        Calculate forward propagation.

        Args:
            att_ws (Tensor):
                Batch of multi head attention weights (B, H, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).

        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).unsqueeze(1)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).unsqueeze(1)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss


class SequentialWrappableConditionalLayerNorm(nn.Module):

    def __init__(self, hidden_dim, speaker_embedding_dim):
        super(SequentialWrappableConditionalLayerNorm, self).__init__()
        if isinstance(hidden_dim, int):
            self.normal_shape = hidden_dim
        self.speaker_embedding_dim = speaker_embedding_dim
        self.W_scale = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.normal_shape), nn.Tanh(), nn.Linear(self.normal_shape, self.normal_shape))
        self.W_bias = nn.Sequential(nn.Linear(self.speaker_embedding_dim, self.normal_shape), nn.Tanh(), nn.Linear(self.normal_shape, self.normal_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale[0].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[2].weight, 0.0)
        torch.nn.init.constant_(self.W_scale[0].bias, 1.0)
        torch.nn.init.constant_(self.W_scale[2].bias, 1.0)
        torch.nn.init.constant_(self.W_bias[0].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[2].weight, 0.0)
        torch.nn.init.constant_(self.W_bias[0].bias, 0.0)
        torch.nn.init.constant_(self.W_bias[2].bias, 0.0)

    def forward(self, packed_input):
        x, speaker_embedding = packed_input
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y = scale.unsqueeze(1) * ((x - mean) / var) + bias.unsqueeze(1)
        return y


class DurationPredictor(torch.nn.Module):
    """
    Duration predictor module.

    This is a module of duration predictor described
    in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different
        between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`,
        those are calculated in linear domain.

    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, utt_embed_dim=None, embedding_integration='AdaIN'):
        """
        Initialize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.embedding_projections = torch.nn.ModuleList()
        self.utt_embed_dim = utt_embed_dim
        self.use_conditional_layernorm_embedding_integration = embedding_integration in ['AdaIN', 'ConditionalLayerNorm']
        for idx in range(n_layers):
            if utt_embed_dim is not None:
                if embedding_integration == 'AdaIN':
                    self.embedding_projections += [AdaIN1d(style_dim=utt_embed_dim, num_features=idim)]
                elif embedding_integration == 'ConditionalLayerNorm':
                    self.embedding_projections += [ConditionalLayerNorm(speaker_embedding_dim=utt_embed_dim, hidden_dim=idim)]
                else:
                    self.embedding_projections += [torch.nn.Linear(utt_embed_dim + idim, idim)]
            else:
                self.embedding_projections += [lambda x: x]
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2), torch.nn.ReLU())]
            self.norms += [LayerNorm(n_chans, dim=1)]
            self.dropouts += [torch.nn.Dropout(dropout_rate)]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False, utt_embed=None):
        xs = xs.transpose(1, -1)
        for f, c, d, p in zip(self.conv, self.norms, self.dropouts, self.embedding_projections):
            xs = f(xs)
            if self.utt_embed_dim is not None:
                xs = integrate_with_utt_embed(hs=xs.transpose(1, 2), utt_embeddings=utt_embed, projection=p, embedding_training=self.use_conditional_layernorm_embedding_integration).transpose(1, 2)
            xs = c(xs)
            xs = d(xs)
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)
        if is_inference:
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()
        else:
            xs = xs.masked_fill(x_masks, 0.0)
        return xs

    def forward(self, xs, padding_mask=None, utt_embed=None):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            padding_mask (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        return self._forward(xs, padding_mask, False, utt_embed=utt_embed)

    def inference(self, xs, padding_mask=None, utt_embed=None):
        """
        Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            padding_mask (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).

        """
        return self._forward(xs, padding_mask, True, utt_embed=utt_embed)


class DurationPredictorLoss(torch.nn.Module):
    """
    Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction='mean'):
        """
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        """
        Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)
        return loss


class Conv1dLinear(torch.nn.Module):
    """
    Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate):
        """
        Initialize Conv1dLinear module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(Conv1dLinear, self).__init__()
        self.w_1 = torch.nn.Conv1d(in_chans, hidden_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.w_2 = torch.nn.Linear(hidden_chans, in_chans)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (torch.Tensor): Batch of input tensors (B, T, in_chans).

        Returns:
            torch.Tensor: Batch of output tensors (B, T, hidden_chans).
        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(self.dropout(x))


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """
        Construct an PositionalEncoding object.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0, device=d_model.device).expand(1, max_len))

    def extend_pe(self, x):
        """
        Reset the positional encodings.
        """
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(x.size(1) - 1, -1, -1.0, dtype=torch.float32).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        """
        Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """
    Scaled positional encoding module.

    See Sec. 3.2  https://arxiv.org/abs/1809.08895

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

    def reset_parameters(self):
        self.alpha.data = torch.tensor(1.0)

    def forward(self, x):
        """
        Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionwiseFeedForward(torch.nn.Module):
    """
    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Conv1d(torch.nn.Conv1d):
    """
    Conv1d module with customized initialization.
    """

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """
    1x1 Conv1d with customized initialization.
    """

    def __init__(self, in_channels, out_channels, bias):
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


class HiFiGANResidualBlock(torch.nn.Module):
    """Residual block module in HiFiGAN."""

    def __init__(self, kernel_size=3, channels=512, dilations=(1, 3, 5), bias=True, use_additional_convs=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}):
        """
        Initialize HiFiGANResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            dilations (List[int]): List of dilation factors.
            use_additional_convs (bool): Whether to use additional convolution layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super().__init__()
        self.use_additional_convs = use_additional_convs
        self.convs1 = torch.nn.ModuleList()
        if use_additional_convs:
            self.convs2 = torch.nn.ModuleList()
        assert kernel_size % 2 == 1, 'Kernel size must be odd number.'
        for dilation in dilations:
            self.convs1 += [torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation, bias=bias, padding=(kernel_size - 1) // 2 * dilation))]
            if use_additional_convs:
                self.convs2 += [torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, bias=bias, padding=(kernel_size - 1) // 2))]

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).
        """
        for idx in range(len(self.convs1)):
            xt = self.convs1[idx](x)
            if self.use_additional_convs:
                xt = self.convs2[idx](xt)
            x = xt + x
        return x


class ResidualStack(torch.nn.Module):

    def __init__(self, kernel_size=3, channels=32, dilation=1, bias=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.2}, pad='ReflectionPad1d', pad_params={}):
        """
        Initialize ResidualStack module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.

        """
        super(ResidualStack, self).__init__()
        assert (kernel_size - 1) % 2 == 0, 'Not support even number kernel size.'
        self.stack = torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params), torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.Conv1d(channels, channels, 1, bias=bias))
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        """
        Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, chennels, T).

        """
        return self.stack(c) + self.skip_layer(c)


class STFT(torch.nn.Module):

    def __init__(self, n_fft=512, win_length=None, hop_length=128, window='hann', center=True, normalized=False, onesided=True):
        super().__init__()
        self.n_fft = n_fft
        if win_length is None:
            self.win_length = n_fft
        else:
            self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.window = window

    def extra_repr(self):
        return f'n_fft={self.n_fft}, win_length={self.win_length}, hop_length={self.hop_length}, center={self.center}, normalized={self.normalized}, onesided={self.onesided}'

    def forward(self, input_wave, ilens=None):
        """
        STFT forward function.
        Args:
            input_wave: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq, 2) or (Batch, Frames, Channels, Freq, 2)
        """
        bs = input_wave.size(0)
        if input_wave.dim() == 3:
            multi_channel = True
            input_wave = input_wave.transpose(1, 2).reshape(-1, input_wave.size(1))
        else:
            multi_channel = False
        if self.window is not None:
            window_func = getattr(torch, f'{self.window}_window')
            window = window_func(self.win_length, dtype=input_wave.dtype, device=input_wave.device)
        else:
            window = None
        complex_output = torch_stft(input=input_wave, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, center=self.center, window=window, normalized=self.normalized, onesided=self.onesided, return_complex=True)
        output = torch.view_as_real(complex_output)
        output = output.transpose(1, 2)
        if multi_channel:
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(1, 2)
        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens = ilens + 2 * pad
            olens = torch.div(ilens - self.win_length, self.hop_length, rounding_mode='trunc') + 1
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None
        return output, olens

    def inverse(self, input, ilens=None):
        """
        Inverse STFT.
        Args:
            input: Tensor(batch, T, F, 2) or ComplexTensor(batch, T, F)
            ilens: (batch,)
        Returns:
            wavs: (batch, samples)
            ilens: (batch,)
        """
        istft = torch.functional.istft
        if self.window is not None:
            window_func = getattr(torch, f'{self.window}_window')
            window = window_func(self.win_length, dtype=input.dtype, device=input.device)
        else:
            window = None
        if isinstance(input, ComplexTensor):
            input = torch.stack([input.real, input.imag], dim=-1)
        assert input.shape[-1] == 2
        input = input.transpose(1, 2)
        wavs = istft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, center=self.center, normalized=self.normalized, onesided=self.onesided, length=ilens.max() if ilens is not None else ilens)
        return wavs, ilens


class VariancePredictor(torch.nn.Module, ABC):
    """
    Variance predictor module.

    This is a module of variance predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(self, idim, n_layers=2, n_chans=384, kernel_size=3, bias=True, dropout_rate=0.5, utt_embed_dim=None, embedding_integration='AdaIN'):
        """
        Initialize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.embedding_projections = torch.nn.ModuleList()
        self.utt_embed_dim = utt_embed_dim
        self.use_conditional_layernorm_embedding_integration = embedding_integration in ['AdaIN', 'ConditionalLayerNorm']
        for idx in range(n_layers):
            if utt_embed_dim is not None:
                if embedding_integration == 'AdaIN':
                    self.embedding_projections += [AdaIN1d(style_dim=utt_embed_dim, num_features=idim)]
                elif embedding_integration == 'ConditionalLayerNorm':
                    self.embedding_projections += [ConditionalLayerNorm(speaker_embedding_dim=utt_embed_dim, hidden_dim=idim)]
                else:
                    self.embedding_projections += [torch.nn.Linear(utt_embed_dim + idim, idim)]
            else:
                self.embedding_projections += [lambda x: x]
            in_chans = idim if idx == 0 else n_chans
            self.conv += [torch.nn.Sequential(torch.nn.Conv1d(in_chans, n_chans, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias), torch.nn.ReLU())]
            self.norms += [LayerNorm(n_chans, dim=1)]
            self.dropouts += [torch.nn.Dropout(dropout_rate)]
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs, padding_mask=None, utt_embed=None):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            padding_mask (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).
        """
        xs = xs.transpose(1, -1)
        for f, c, d, p in zip(self.conv, self.norms, self.dropouts, self.embedding_projections):
            xs = f(xs)
            if self.utt_embed_dim is not None:
                xs = integrate_with_utt_embed(hs=xs.transpose(1, 2), utt_embeddings=utt_embed, projection=p, embedding_training=self.use_conditional_layernorm_embedding_integration).transpose(1, 2)
            xs = c(xs)
            xs = d(xs)
        xs = self.linear(xs.transpose(1, 2))
        if padding_mask is not None:
            xs = xs.masked_fill(padding_mask, 0.0)
        return xs


class DiscriminatorNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.filters = nn.ModuleList([nn.utils.weight_norm(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))), nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))), nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))), nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))), nn.utils.weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))])
        self.out = nn.utils.weight_norm(nn.Conv2d(32, 1, 3, 1, 1))
        self.fc = nn.Linear(900, 1)

    def forward(self, y):
        feature_maps = list()
        feature_maps.append(y)
        for d in self.filters:
            y = d(y)
            feature_maps.append(y)
            y = nn.functional.leaky_relu(y, 0.1)
        y = self.out(y)
        feature_maps.append(y)
        y = torch.flatten(y, 1, -1)
        y = self.fc(y)
        return y, feature_maps


def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SpectrogramDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.D = DiscriminatorNet()
        self.D.apply(weights_init_D)

    def _generator_feedback(self, data_generated, data_real):
        for p in self.D.parameters():
            p.requires_grad = False
        score_fake, fmap_fake = self.D(data_generated)
        _, fmap_real = self.D(data_real)
        feature_matching_loss = 0.0
        for feat_fake, feat_real in zip(fmap_fake, fmap_real):
            feature_matching_loss += nn.functional.l1_loss(feat_fake, feat_real.detach())
        discr_loss = nn.functional.mse_loss(input=score_fake, target=torch.ones(score_fake.shape, device=score_fake.device), reduction='mean')
        return feature_matching_loss + discr_loss

    def _discriminator_feature_matching(self, data_generated, data_real):
        for p in self.D.parameters():
            p.requires_grad = True
        self.D.train()
        score_fake, _ = self.D(data_generated)
        score_real, _ = self.D(data_real)
        discr_loss = 0.0
        discr_loss = discr_loss + nn.functional.mse_loss(input=score_fake, target=torch.zeros(score_fake.shape, device=score_fake.device), reduction='mean')
        discr_loss = discr_loss + nn.functional.mse_loss(input=score_real, target=torch.ones(score_real.shape, device=score_real.device), reduction='mean')
        return discr_loss

    def calc_discriminator_loss(self, data_generated, data_real):
        return self._discriminator_feature_matching(data_generated.detach(), data_real)

    def calc_generator_feedback(self, data_generated, data_real):
        return self._generator_feedback(data_generated, data_real)


class MaskedRefinementObjective(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.classification_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.l1_loss = torch.nn.L1Loss(reduction='none')

    def forward(self, predicted_one_hot, gold_one_hot, non_pad_mask):
        ce = list()
        for one_hot_pred, one_hot_target in zip(predicted_one_hot, gold_one_hot.transpose(0, 1).transpose(2, 3)):
            ce.append(self.classification_loss(one_hot_pred, one_hot_target))
        classification_loss = torch.stack(ce).sum(0)
        out_masks = non_pad_mask.unsqueeze(-1)
        out_masks = torch.nn.functional.pad(out_masks.transpose(1, 2), [0, gold_one_hot.size(2) - out_masks.size(1), 0, 0, 0, 0], value=False).transpose(1, 2)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= gold_one_hot.size(0) * gold_one_hot.size(-1)
        classification_loss = classification_loss.mul(out_weights.squeeze()).masked_select(out_masks.squeeze()).sum()
        return classification_loss, classification_loss


class CodecRefinementTransformer(torch.nn.Module):

    def __init__(self, attention_dimension=128, num_codebooks=4, codebook_size=1024, backtranslation_dim=8, attention_heads=4, positionwise_conv_kernel_size=1, use_macaron_style_in_conformer=True, use_cnn_in_conformer=False, decoder_layers=6, decoder_units=1280, decoder_concat_after=False, conformer_decoder_kernel_size=31, decoder_normalize_before=True, transformer_dec_dropout_rate=0.2, transformer_dec_positional_dropout_rate=0.1, transformer_dec_attn_dropout_rate=0.1, utt_embed_dim=512, use_conditional_layernorm_embedding_integration=False):
        super().__init__()
        self.reconstruction_transformer = Conformer(conformer_type='decoder', attention_dim=num_codebooks * backtranslation_dim, attention_heads=attention_heads, linear_units=decoder_units, num_blocks=decoder_layers, input_layer=None, dropout_rate=transformer_dec_dropout_rate, positional_dropout_rate=transformer_dec_positional_dropout_rate, attention_dropout_rate=transformer_dec_attn_dropout_rate, normalize_before=decoder_normalize_before, concat_after=decoder_concat_after, positionwise_conv_kernel_size=positionwise_conv_kernel_size, macaron_style=use_macaron_style_in_conformer, use_cnn_module=use_cnn_in_conformer, cnn_module_kernel=conformer_decoder_kernel_size, use_output_norm=False, utt_embed=utt_embed_dim, use_conditional_layernorm_embedding_integration=use_conditional_layernorm_embedding_integration)
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.input_embeddings = torch.nn.ModuleList()
        self.backtranslation_heads = torch.nn.ModuleList()
        self.hierarchical_classifier = torch.nn.ModuleList()
        self.padding_id = codebook_size + 5
        for head in range(num_codebooks):
            self.input_embeddings.append(torch.nn.Embedding(num_embeddings=self.padding_id + 1, embedding_dim=backtranslation_dim, padding_idx=self.padding_id))
            self.backtranslation_heads.append(torch.nn.Embedding(num_embeddings=self.padding_id + 1, embedding_dim=backtranslation_dim, padding_idx=self.padding_id))
            self.hierarchical_classifier.append(torch.nn.Linear(num_codebooks * backtranslation_dim + head * backtranslation_dim, codebook_size))
        self.criterion = MaskedRefinementObjective()
        for backtranslation_head in self.backtranslation_heads:
            torch.nn.init.normal_(backtranslation_head.weight, mean=0, std=attention_dimension ** -0.5)
        for input_embedding in self.input_embeddings:
            torch.nn.init.normal_(input_embedding.weight, mean=0, std=attention_dimension ** -0.5)

    def forward(self, index_sequence, is_inference, speaker_embedding, padding_mask=None, gold_index_sequence=None):
        """
        index_sequence: [batch, codebook_index, time_steps] a sequence of indexes that come from an argmax of the previous prediction layer.
        is_inference: boolean flag that indicates whether to return the masked language modelling loss or the refined sequence
        speaker_embedding: [batch, speaker_embed_dim]
        padding_mask: [batch, time_steps] a mask that is True for all time steps that are padding and should not be considered and False everywhere else.

        return: loss if is_inference is false, otherwise [batch, codebook_index, time_steps] a sequence of indexes with the same shape and same interpretation, refined through iterative masked language modelling.
        """
        if not is_inference:
            index_sequence_padding_accounted = index_sequence.masked_fill(mask=padding_mask.unsqueeze(1), value=self.padding_id)
        else:
            index_sequence_padding_accounted = index_sequence
        sequence_of_continuous_tokens = self.indexes_per_codebook_to_stacked_embedding_vector(index_sequence_padding_accounted)
        contextualized_sequence = self.contextualize_sequence(sequence_of_continuous_tokens, speaker_embedding, non_padding_mask=~padding_mask if padding_mask is not None else None)
        predicted_indexes_one_hot = list()
        backtranslated_indexes = list()
        for head_index, classifier_head in enumerate(self.hierarchical_classifier):
            predicted_indexes_one_hot.append(classifier_head(torch.cat([contextualized_sequence] + backtranslated_indexes, dim=2)))
            predicted_lookup_index = torch.argmax(predicted_indexes_one_hot[-1], dim=-1)
            backtranslation = self.backtranslation_heads[head_index](predicted_lookup_index)
            if len(backtranslation.size()) == 1:
                backtranslation = backtranslation.unsqueeze(0)
            backtranslated_indexes.append(backtranslation)
        indexes = torch.cat(predicted_indexes_one_hot, dim=2)
        indexes = indexes.view(contextualized_sequence.size(0), contextualized_sequence.size(1), self.num_codebooks, self.codebook_size)
        indexes = indexes.transpose(1, 2)
        indexes = indexes.transpose(2, 3)
        indexes = indexes.transpose(0, 1)
        if is_inference:
            return indexes
        else:
            return self.criterion(predicted_one_hot=indexes, gold_one_hot=gold_index_sequence, non_pad_mask=~padding_mask)

    def contextualize_sequence(self, masked_sequence, utterance_embedding, non_padding_mask):
        decoded_speech, _ = self.reconstruction_transformer(masked_sequence, non_padding_mask.unsqueeze(2) if non_padding_mask is not None else None, utterance_embedding=utterance_embedding)
        return decoded_speech

    def indexes_per_codebook_to_stacked_embedding_vector(self, index_sequence_per_codebook):
        continuous_frame_sequences = list()
        for codebook_id, backtranslation_head in enumerate(self.backtranslation_heads):
            continuous_frame_sequences.append(backtranslation_head(index_sequence_per_codebook.transpose(0, 1)[codebook_id]))
        stacked_embedding_vector = torch.cat(continuous_frame_sequences, dim=-1)
        return stacked_embedding_vector


class DurationCalculator(torch.nn.Module):

    def __init__(self, reduction_factor=1.0):
        super().__init__()

    @torch.no_grad()
    def forward(self, att_ws, vis=None):
        """
        Convert alignment matrix to durations.
        """
        if vis is not None:
            plt.figure(figsize=(8, 4))
            plt.imshow(att_ws.cpu().numpy(), interpolation='nearest', aspect='auto', origin='lower')
            plt.xlabel('Inputs')
            plt.ylabel('Outputs')
            plt.tight_layout()
            plt.savefig(vis)
            plt.close()
        durations = torch.stack([att_ws.argmax(-1).eq(i).sum() for i in range(att_ws.shape[1])])
        return durations.view(-1)


class EnergyCalculator(torch.nn.Module):

    def __init__(self, fs=16000, n_fft=1024, win_length=None, hop_length=256, window='hann', center=True, normalized=False, onesided=True, use_token_averaged_energy=True, reduction_factor=1):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.use_token_averaged_energy = use_token_averaged_energy
        if use_token_averaged_energy:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor
        self.stft = STFT(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=window, center=center, normalized=normalized, onesided=onesided)

    def output_size(self):
        return 1

    def get_parameters(self):
        return dict(fs=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, win_length=self.win_length, center=self.stft.center, normalized=self.stft.normalized, use_token_averaged_energy=self.use_token_averaged_energy, reduction_factor=self.reduction_factor)

    def forward(self, input_waves, input_waves_lengths=None, feats_lengths=None, durations=None, durations_lengths=None, norm_by_average=True, text=None):
        if input_waves_lengths is None:
            input_waves_lengths = input_waves.new_ones(input_waves.shape[0], dtype=torch.long) * input_waves.shape[1]
        input_stft, energy_lengths = self.stft(input_waves, input_waves_lengths)
        assert input_stft.dim() >= 4, input_stft.shape
        assert input_stft.shape[-1] == 2, input_stft.shape
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        energy = torch.sqrt(torch.clamp(input_power.sum(dim=2), min=1e-10))
        if feats_lengths is not None:
            energy = [self._adjust_num_frames(e[:el].view(-1), fl) for e, el, fl in zip(energy, energy_lengths, feats_lengths)]
            energy_lengths = feats_lengths
        if self.use_token_averaged_energy:
            energy = [self._average_by_duration(e[:el].view(-1), d, text) for e, el, d in zip(energy, energy_lengths, durations)]
            energy_lengths = durations_lengths
        if isinstance(energy, list):
            energy = pad_list(energy, 0.0)
        if norm_by_average:
            average = energy[0][energy[0] != 0.0].mean()
            energy = energy / average
        return energy.unsqueeze(-1), energy_lengths

    def _average_by_duration(self, x, d, text=None):
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [(x[start:end].mean() if len(x[start:end]) != 0 else x.new_tensor(0.0)) for start, end in zip(d_cumsum[:-1], d_cumsum[1:])]
        return torch.stack(x_avg)

    @staticmethod
    def _adjust_num_frames(x, num_frames):
        if num_frames > len(x):
            x = F.pad(x, (0, num_frames - len(x)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x


class ActNorm(nn.Module):

    def __init__(self, channels, ddi=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi
        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2))
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True
        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = torch.sum(-self.logs) * x_len
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len
        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - m ** 2
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-06))
            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape)
            logs_init = (-logs).view(*self.logs.shape)
            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):

    def __init__(self, channels, n_split=4, no_jacobian=False, lu=True, n_sqz=2, **kwargs):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.no_jacobian = no_jacobian
        w_init = torch.linalg.qr(torch.FloatTensor(self.n_split, self.n_split).normal_(), 'complete')[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.lu = lu
        if lu:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_init.shape, dtype=float), -1)
            eye = np.eye(*w_init.shape, dtype=float)
            self.register_buffer('p', torch.Tensor(np_p.astype(float)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(float)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(float)), requires_grad=True)
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(float)), requires_grad=True)
            self.u = nn.Parameter(torch.Tensor(np_u.astype(float)), requires_grad=True)
            self.register_buffer('l_mask', torch.Tensor(l_mask))
            self.register_buffer('eye', torch.Tensor(eye))
        else:
            self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])
        x = x.view(b, self.n_sqz, c // self.n_split, self.n_split // self.n_sqz, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)
        if self.lu:
            self.weight, log_s = self._get_weight()
            logdet = log_s.sum()
            logdet = logdet * (c / self.n_split) * x_len
        else:
            logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len
        if reverse:
            if hasattr(self, 'weight_inv'):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float())
            logdet = -logdet
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)
        z = z.view(b, self.n_sqz, self.n_split // self.n_sqz, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def _get_weight(self):
        l, log_s, u = self.l, self.log_s, self.u
        l = l * self.l_mask + self.eye
        u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
        weight = torch.matmul(self.p, torch.matmul(l, u))
        return weight, log_s

    def store_inverse(self):
        weight, _ = self._get_weight()
        self.weight_inv = torch.inverse(weight.float())


class InvConv(nn.Module):

    def __init__(self, channels, no_jacobian=False, lu=True, **kwargs):
        super().__init__()
        w_shape = [channels, channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(float)
        LU_decomposed = lu
        if not LU_decomposed:
            self.register_parameter('weight', nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=float), -1)
            eye = np.eye(*w_shape, dtype=float)
            self.register_buffer('p', torch.Tensor(np_p.astype(float)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(float)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(float)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(float)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(float)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed
        self.weight = None

    def get_weight(self, device, reverse):
        w_shape = self.w_shape
        self.p = self.p
        self.sign_s = self.sign_s
        self.l_mask = self.l_mask
        self.eye = self.eye
        l = self.l * self.l_mask + self.eye
        u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
        dlogdet = self.log_s.sum()
        if not reverse:
            w = torch.matmul(self.p, torch.matmul(l, u))
        else:
            l = torch.inverse(l.double()).float()
            u = torch.inverse(u.double()).float()
            w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
        return w.view(w_shape[0], w_shape[1], 1), dlogdet

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        """
        log-det = log|abs(|W|)| * pixels
        """
        b, c, t = x.size()
        if x_mask is None:
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])
        logdet = 0
        if not reverse:
            weight, dlogdet = self.get_weight(x.device, reverse)
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet + dlogdet * x_len
            return z, logdet
        else:
            if self.weight is None:
                weight, dlogdet = self.get_weight(x.device, reverse)
            else:
                weight, dlogdet = self.weight, self.dlogdet
            z = F.conv1d(x, weight)
            if logdet is not None:
                logdet = logdet - dlogdet * x_len
            return z, logdet

    def store_inverse(self):
        self.weight, self.dlogdet = self.get_weight('cuda', reverse=True)


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):

    def __init__(self, hidden_size, kernel_size, dilation_rate, n_layers, c_cond=0, p_dropout=0, share_cond_layers=False, is_BTC=False, use_weightnorm=True):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert hidden_size % 2 == 0
        self.is_BTC = is_BTC
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = c_cond
        self.p_dropout = p_dropout
        self.share_cond_layers = share_cond_layers
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        if c_cond != 0 and not share_cond_layers:
            cond_layer = torch.nn.Conv1d(c_cond, 2 * hidden_size * n_layers, 1)
            if use_weightnorm:
                self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            else:
                self.cond_layer = cond_layer
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size, dilation=dilation, padding=padding)
            if use_weightnorm:
                in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_size
            else:
                res_skip_channels = hidden_size
            res_skip_layer = torch.nn.Conv1d(hidden_size, res_skip_channels, 1)
            if use_weightnorm:
                res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, nonpadding=None, cond=None):
        if self.is_BTC:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2) if cond is not None else None
            nonpadding = nonpadding.transpose(1, 2) if nonpadding is not None else None
        if nonpadding is None:
            nonpadding = 1
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_size])
        if cond is not None and not self.share_cond_layers:
            cond = self.cond_layer(cond)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if cond is not None:
                cond_offset = i * 2 * self.hidden_size
                cond_l = cond[:, cond_offset:cond_offset + 2 * self.hidden_size, :]
            else:
                cond_l = torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, cond_l, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_size, :]) * nonpadding
                output = output + res_skip_acts[:, self.hidden_size:, :]
            else:
                output = output + res_skip_acts
        output = output * nonpadding
        if self.is_BTC:
            output = output.transpose(1, 2)
        return output

    def remove_weight_norm(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)


class CouplingBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0.0, sigmoid_scale=False, wn=None, use_weightnorm=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale
        start = torch.nn.Conv1d(in_channels // 2, hidden_channels, 1)
        if use_weightnorm:
            start = torch.nn.utils.weight_norm(start)
        self.start = start
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        self.wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout, use_weightnorm=use_weightnorm)
        if wn is not None:
            self.wn.in_layers = wn.in_layers
            self.wn.res_skip_layers = wn.res_skip_layers

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels // 2], x[:, self.in_channels // 2:]
        x = self.start(x_0) * x_mask
        x = self.wn(x, x_mask, g)
        out = self.end(x)
        z_0 = x_0
        m = out[:, :self.in_channels // 2, :]
        logs = out[:, self.in_channels // 2:, :]
        if self.sigmoid_scale:
            logs = torch.log(1e-06 + torch.sigmoid(logs + 2))
        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = torch.sum(-logs * x_mask, [1, 2])
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])
        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()


class Glow(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_blocks, n_layers, condition_integration_projection, p_dropout=0.0, n_split=4, n_sqz=2, sigmoid_scale=False, text_condition_channels=0, inv_conv_type='near', share_cond_layers=False, share_wn_layers=0, use_weightnorm=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.text_condition_channels = text_condition_channels
        self.share_cond_layers = share_cond_layers
        self.prior_dist = dist.Normal(0, 1)
        self.g_proj = condition_integration_projection
        if text_condition_channels != 0 and share_cond_layers:
            cond_layer = torch.nn.Conv1d(text_condition_channels * n_sqz, 2 * hidden_channels * n_layers, 1)
            if use_weightnorm:
                self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
            else:
                self.cond_layer = cond_layer
        wn = None
        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * n_sqz))
            if inv_conv_type == 'near':
                self.flows.append(InvConvNear(channels=in_channels * n_sqz, n_split=n_split, n_sqz=n_sqz))
            if inv_conv_type == 'invconv':
                self.flows.append(InvConv(channels=in_channels * n_sqz))
            if share_wn_layers > 0:
                if b % share_wn_layers == 0:
                    wn = WN(hidden_channels, kernel_size, dilation_rate, n_layers, text_condition_channels * n_sqz, p_dropout, share_cond_layers, use_weightnorm=use_weightnorm)
            self.flows.append(CouplingBlock(in_channels * n_sqz, hidden_channels, kernel_size=kernel_size, dilation_rate=dilation_rate, n_layers=n_layers, gin_channels=text_condition_channels * n_sqz, p_dropout=p_dropout, sigmoid_scale=sigmoid_scale, wn=wn, use_weightnorm=use_weightnorm))

    def forward(self, tgt_mels, infer, mel_out, encoded_texts, tgt_nonpadding, glow_sampling_temperature=0.7):
        x_recon = mel_out.transpose(1, 2)
        g = x_recon
        B, _, T = g.shape
        if encoded_texts is not None and self.text_condition_channels != 0:
            g = torch.cat([g, encoded_texts.transpose(1, 2)], 1)
            g = self.g_proj(g)
        prior_dist = self.prior_dist
        if not infer:
            y_lengths = tgt_nonpadding.sum(-1)
            tgt_mels = tgt_mels.transpose(1, 2)
            z_postflow, ldj = self._forward(tgt_mels, tgt_nonpadding, g=g)
            ldj = ldj / y_lengths / 80
            try:
                postflow_loss = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
            except ValueError:
                None
                postflow_loss = None
            return postflow_loss
        else:
            nonpadding = torch.ones_like(x_recon[:, :1, :]) if tgt_nonpadding is None else tgt_nonpadding
            z_post = torch.randn(x_recon.shape) * glow_sampling_temperature
            x_recon, _ = self._forward(z_post, nonpadding, g, reverse=True)
            return x_recon.transpose(1, 2)

    def _forward(self, x, x_mask=None, g=None, reverse=False, return_hiddens=False):
        logdet_tot = 0
        if not reverse:
            flows = self.flows
        else:
            flows = reversed(self.flows)
        if return_hiddens:
            hs = []
        if self.n_sqz > 1:
            x, x_mask_ = glow_utils.squeeze(x, x_mask, self.n_sqz)
            if g is not None:
                g, _ = glow_utils.squeeze(g, x_mask, self.n_sqz)
            x_mask = x_mask_
        if self.share_cond_layers and g is not None:
            g = self.cond_layer(g)
        for f in flows:
            x, logdet = f(x, x_mask, g=g, reverse=reverse)
            if return_hiddens:
                hs.append(x)
            logdet_tot += logdet
        if self.n_sqz > 1:
            x, x_mask = glow_utils.unsqueeze(x, x_mask, self.n_sqz)
        if return_hiddens:
            return x, logdet_tot, hs
        return x, logdet_tot

    def store_inverse(self):

        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(remove_weight_norm)
        for f in self.flows:
            f.store_inverse()


class CacheCreator:

    def __init__(self, cache_root='.'):
        self.iso_codes = list(load_json_from_path(hf_hub_download(cache_dir=MODEL_DIR, repo_id='Flux9665/ToucanTTS', filename='iso_to_fullname.json')).keys())
        self.iso_lookup = load_json_from_path(hf_hub_download(cache_dir=MODEL_DIR, repo_id='Flux9665/ToucanTTS', filename='iso_lookup.json'))
        self.cache_root = cache_root
        self.pairs = list()
        for index_1 in tqdm(range(len(self.iso_codes)), desc='Collecting language pairs'):
            for index_2 in range(index_1, len(self.iso_codes)):
                self.pairs.append((self.iso_codes[index_1], self.iso_codes[index_2]))

    def create_tree_cache(self, cache_root='.'):
        iso_to_family_memberships = load_json_from_path(hf_hub_download(cache_dir=MODEL_DIR, repo_id='Flux9665/ToucanTTS', filename='iso_to_fullname.json'))
        self.pair_to_tree_distance = dict()
        for pair in tqdm(self.pairs, desc='Generating tree pairs'):
            lang_1 = pair[0]
            lang_2 = pair[1]
            depth_of_l1 = len(iso_to_family_memberships[lang_1])
            depth_of_l2 = len(iso_to_family_memberships[lang_2])
            depth_of_lca = len(set(iso_to_family_memberships[pair[0]]).intersection(set(iso_to_family_memberships[pair[1]])))
            self.pair_to_tree_distance[pair] = depth_of_l1 + depth_of_l2 - (2 * (depth_of_lca + 1) if depth_of_lca > 1 else depth_of_lca)
        min_dist = min(self.pair_to_tree_distance.values())
        max_dist = max(self.pair_to_tree_distance.values())
        for pair in self.pair_to_tree_distance:
            if pair[0] == pair[1]:
                self.pair_to_tree_distance[pair] = 0.0
            else:
                self.pair_to_tree_distance[pair] = (self.pair_to_tree_distance[pair] + abs(min_dist)) / (max_dist + abs(min_dist))
        lang_1_to_lang_2_to_tree_dist = dict()
        for pair in tqdm(self.pair_to_tree_distance):
            lang_1 = pair[0]
            lang_2 = pair[1]
            dist = self.pair_to_tree_distance[pair]
            if lang_1 not in lang_1_to_lang_2_to_tree_dist.keys():
                lang_1_to_lang_2_to_tree_dist[lang_1] = dict()
            lang_1_to_lang_2_to_tree_dist[lang_1][lang_2] = dist
        with open(os.path.join(cache_root, 'lang_1_to_lang_2_to_tree_dist.json'), 'w', encoding='utf-8') as f:
            json.dump(lang_1_to_lang_2_to_tree_dist, f, ensure_ascii=False, indent=4)

    def create_map_cache(self, cache_root='.'):
        self.pair_to_map_dist = dict()
        iso_to_long_lat = load_json_from_path(hf_hub_download(cache_dir=MODEL_DIR, repo_id='Flux9665/ToucanTTS', filename='iso_to_long_lat.json'))
        for pair in tqdm(self.pairs, desc='Generating map pairs'):
            try:
                long_1, lat_1 = iso_to_long_lat[pair[0]]
                long_2, lat_2 = iso_to_long_lat[pair[1]]
                geodesic((lat_1, long_1), (lat_2, long_2))
                self.pair_to_map_dist[pair] = geodesic((lat_1, long_1), (lat_2, long_2)).miles
            except KeyError:
                pass
        lang_1_to_lang_2_to_map_dist = dict()
        for pair in self.pair_to_map_dist:
            lang_1 = pair[0]
            lang_2 = pair[1]
            dist = self.pair_to_map_dist[pair]
            if lang_1 not in lang_1_to_lang_2_to_map_dist.keys():
                lang_1_to_lang_2_to_map_dist[lang_1] = dict()
            lang_1_to_lang_2_to_map_dist[lang_1][lang_2] = dist
        with open(os.path.join(cache_root, 'lang_1_to_lang_2_to_map_dist.json'), 'w', encoding='utf-8') as f:
            json.dump(lang_1_to_lang_2_to_map_dist, f, ensure_ascii=False, indent=4)

    def create_oracle_cache(self, model_path, cache_root='.'):
        """Oracle language-embedding distance of supervised languages is only used for evaluation, not usable for zero-shot.

        Note: The generated oracle cache is only valid for the given `model_path`!"""
        loss_fn = torch.nn.MSELoss(reduction='mean')
        self.pair_to_oracle_dist = dict()
        lang_embs = torch.load(model_path)['model']['encoder.language_embedding.weight']
        lang_embs.requires_grad_(False)
        for pair in tqdm(self.pairs, desc='Generating oracle pairs'):
            try:
                dist = loss_fn(lang_embs[self.iso_lookup[-1][pair[0]]], lang_embs[self.iso_lookup[-1][pair[1]]]).item()
                self.pair_to_oracle_dist[pair] = dist
            except KeyError:
                pass
        lang_1_to_lang_2_oracle_dist = dict()
        for pair in self.pair_to_oracle_dist:
            lang_1 = pair[0]
            lang_2 = pair[1]
            dist = self.pair_to_oracle_dist[pair]
            if lang_1 not in lang_1_to_lang_2_oracle_dist.keys():
                lang_1_to_lang_2_oracle_dist[lang_1] = dict()
            lang_1_to_lang_2_oracle_dist[lang_1][lang_2] = dist
        with open(os.path.join(cache_root, 'lang_1_to_lang_2_to_oracle_dist.json'), 'w', encoding='utf-8') as f:
            json.dump(lang_1_to_lang_2_oracle_dist, f, ensure_ascii=False, indent=4)

    def create_learned_cache(self, model_path, cache_root='.'):
        """Note: The generated learned distance cache is only valid for the given `model_path`!"""
        create_learned_cache(model_path, cache_root=cache_root)

    def create_required_files(self, model_path, create_oracle=False):
        if not os.path.exists(os.path.join(self.cache_root, 'lang_1_to_lang_2_to_tree_dist.json')) or os.path.exists(hf_hub_download(cache_dir=MODEL_DIR, repo_id='Flux9665/ToucanTTS', filename='lang_1_to_lang_2_to_tree_dist.json')):
            self.create_tree_cache(cache_root='Preprocessing/multilinguality')
        if not os.path.exists(os.path.join(self.cache_root, 'lang_1_to_lang_2_to_map_dist.json')) or os.path.exists(hf_hub_download(cache_dir=MODEL_DIR, repo_id='Flux9665/ToucanTTS', filename='lang_1_to_lang_2_to_map_dist.json')):
            self.create_map_cache(cache_root='Preprocessing/multilinguality')
        if not os.path.exists(os.path.join(self.cache_root, 'asp_dict.pkl')) or os.path.exists(hf_hub_download(cache_dir=MODEL_DIR, repo_id='Flux9665/ToucanTTS', filename='asp_dict.pkl')):
            raise FileNotFoundError('asp_dict.pkl must be downloaded separately.')
        if not os.path.exists(os.path.join(self.cache_root, 'lang_1_to_lang_2_to_learned_dist.json')) or os.path.exists(hf_hub_download(cache_dir=MODEL_DIR, repo_id='Flux9665/ToucanTTS', filename='lang_1_to_lang_2_to_learned_dist.json')):
            self.create_learned_cache(model_path=model_path, cache_root='Preprocessing/multilinguality')
        if create_oracle:
            if not os.path.exists(os.path.join(self.cache_root, 'lang_1_to_lang_2_to_oracle_dist.json')):
                if not model_path:
                    raise ValueError('model_path is required for creating oracle cache.')
                self.create_oracle_cache(model_path=args.model_path, cache_root='Preprocessing/multilinguality')
        None


class LanguageEmbeddingSpaceStructureLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        cc = CacheCreator(cache_root='Preprocessing/multilinguality')
        if not os.path.exists('Preprocessing/multilinguality/lang_1_to_lang_2_to_tree_dist.json'):
            cc.create_tree_cache(cache_root='Preprocessing/multilinguality')
        if not os.path.exists('Preprocessing/multilinguality/lang_1_to_lang_2_to_tree_dist.json'):
            cc.create_map_cache(cache_root='Preprocessing/multilinguality')
        self.tree_dist = load_json_from_path('Preprocessing/multilinguality/lang_1_to_lang_2_to_tree_dist.json')
        self.map_dist = load_json_from_path('Preprocessing/multilinguality/lang_1_to_lang_2_to_map_dist.json')
        self.largest_value_map_dist = 0.0
        for _, values in self.map_dist.items():
            for _, value in values.items():
                self.largest_value_map_dist = max(self.largest_value_map_dist, value)
        self.iso_codes_to_ids = load_json_from_path('Preprocessing/multilinguality/iso_lookup.json')[-1]
        self.ids_to_iso_codes = {v: k for k, v in self.iso_codes_to_ids.items()}

    def forward(self, language_ids, language_embeddings):
        """
        Args:
            language_ids (Tensor): IDs of languages in the same order as the embeddings to calculate the distances according to the metrics.
            language_embeddings (Tensor): Batch of language embeddings, of which the distances will be compared to the distances according to the metrics.

        Returns:
            Tensor: Language Embedding Structure Loss Value
        """
        losses = list()
        for language_id_1, language_embedding_1 in zip(language_ids, language_embeddings):
            for language_id_2, language_embedding_2 in zip(language_ids, language_embeddings):
                if language_id_1 != language_id_2:
                    embed_dist = torch.nn.functional.l1_loss(language_embedding_1, language_embedding_2)
                    lang_1 = self.ids_to_iso_codes[language_id_1]
                    lang_2 = self.ids_to_iso_codes[language_id_2]
                    try:
                        tree_dist = self.tree_dist[lang_1][lang_2]
                    except KeyError:
                        tree_dist = self.tree_dist[lang_2][lang_1]
                    try:
                        map_dist = self.map_dist[lang_1][lang_2] / self.largest_value_map_dist
                    except KeyError:
                        map_dist = self.map_dist[lang_2][lang_1] / self.largest_value_map_dist
                    metric_distance = (torch.tensor(tree_dist) + torch.tensor(map_dist)) / 2
                    losses.append(torch.nn.functional.l1_loss(embed_dist, metric_distance))
        return sum(losses) / len(losses)


class Parselmouth(torch.nn.Module):
    """
    F0 estimation with Parselmouth https://parselmouth.readthedocs.io/en/stable/index.html
    """

    def __init__(self, fs=16000, n_fft=1024, hop_length=256, f0min=40, f0max=600, use_token_averaged_f0=True, use_continuous_f0=True, use_log_f0=False, reduction_factor=1):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_period = 1000 * hop_length / fs
        self.f0min = f0min
        self.f0max = f0max
        self.use_token_averaged_f0 = use_token_averaged_f0
        self.use_continuous_f0 = use_continuous_f0
        self.use_log_f0 = use_log_f0
        if use_token_averaged_f0:
            assert reduction_factor >= 1
        self.reduction_factor = reduction_factor

    def output_size(self):
        return 1

    def get_parameters(self):
        return dict(fs=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, f0min=self.f0min, f0max=self.f0max, use_token_averaged_f0=self.use_token_averaged_f0, use_continuous_f0=self.use_continuous_f0, use_log_f0=self.use_log_f0, reduction_factor=self.reduction_factor)

    def forward(self, input_waves, input_waves_lengths=None, feats_lengths=None, durations=None, durations_lengths=None, norm_by_average=True, text=None):
        pitch = self._calculate_f0(input_waves[0])
        pitch = self._adjust_num_frames(pitch, feats_lengths[0]).view(-1)
        pitch = self._average_by_duration(pitch, durations[0], text).view(-1)
        pitch_lengths = durations_lengths
        if norm_by_average:
            average = pitch[pitch != 0.0].mean()
            pitch = pitch / average
        return pitch.unsqueeze(-1), pitch_lengths

    def _calculate_f0(self, input):
        x = input.cpu().numpy().astype(np.double)
        snd = parselmouth.Sound(values=x, sampling_frequency=self.fs)
        f0 = snd.to_pitch(time_step=self.hop_length / self.fs, pitch_floor=self.f0min, pitch_ceiling=self.f0max).selected_array['frequency']
        if self.use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)
        if self.use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return input.new_tensor(f0.reshape(-1), dtype=torch.float)

    @staticmethod
    def _adjust_num_frames(x, num_frames):
        if num_frames > len(x):
            x = F.pad(x, (math.ceil((num_frames - len(x)) / 2), math.floor((num_frames - len(x)) / 2)))
        elif num_frames < len(x):
            x = x[:num_frames]
        return x

    @staticmethod
    def _convert_to_continuous_f0(f0: 'np.array'):
        if (f0 == 0).all():
            return f0
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0
        nonzero_idxs = np.where(f0 != 0)[0]
        interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
        f0 = interp_fn(np.arange(0, f0.shape[0]))
        return f0

    def _average_by_duration(self, x, d, text=None):
        d_cumsum = F.pad(d.cumsum(dim=0), (1, 0))
        x_avg = [(x[start:end].masked_select(x[start:end].gt(0.0)).mean(dim=0) if len(x[start:end].masked_select(x[start:end].gt(0.0))) != 0 else x.new_tensor(0.0)) for start, end in zip(d_cumsum[:-1], d_cumsum[1:])]
        return torch.stack(x_avg)


class ToucanTTSLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_criterion = torch.nn.L1Loss(reduction='none')
        self.l2_criterion = torch.nn.MSELoss(reduction='none')
        self.duration_criterion = DurationPredictorLoss(reduction='none')

    def forward(self, predicted_features, gold_features, features_lengths, text_lengths, gold_durations, predicted_durations, predicted_pitch, predicted_energy, gold_pitch, gold_energy):
        """
        Args:
            predicted_features (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            gold_features (Tensor): Batch of target features (B, Lmax, odim).
            features_lengths (LongTensor): Batch of the lengths of each target (B,).
            gold_durations (LongTensor): Batch of durations (B, Tmax).
            gold_pitch (LongTensor): Batch of pitch (B, Tmax).
            gold_energy (LongTensor): Batch of energy (B, Tmax).
            predicted_durations (LongTensor): Batch of outputs of duration predictor (B, Tmax).
            predicted_pitch (LongTensor): Batch of outputs of pitch predictor (B, Tmax).
            predicted_energy (LongTensor): Batch of outputs of energy predictor (B, Tmax).
            text_lengths (LongTensor): Batch of the lengths of each input (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration loss value
        """
        distance_loss = self.l1_criterion(predicted_features, gold_features)
        duration_loss = self.duration_criterion(predicted_durations, gold_durations)
        pitch_loss = self.l2_criterion(predicted_pitch, gold_pitch)
        energy_loss = self.l2_criterion(predicted_energy, gold_energy)
        out_masks = make_non_pad_mask(features_lengths).unsqueeze(-1)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= gold_features.size(0) * gold_features.size(-1)
        duration_masks = make_non_pad_mask(text_lengths)
        duration_weights = duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
        variance_masks = duration_masks.unsqueeze(-1)
        variance_weights = duration_weights.unsqueeze(-1)
        distance_loss = distance_loss.mul(out_weights).masked_select(out_masks).sum()
        duration_loss = duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
        pitch_loss = pitch_loss.mul(variance_weights).masked_select(variance_masks).sum()
        energy_loss = energy_loss.mul(variance_weights).masked_select(variance_masks).sum()
        return distance_loss, duration_loss, pitch_loss, energy_loss


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        return input.transpose(1, 2)


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-09

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + 1.0 / (beta + self.no_div_by_zero) * pow(sin(x * alpha), 2)
        return x


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


class AMPBlock1(torch.nn.Module):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(AMPBlock1, self).__init__()
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2])))])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))), weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList([Activation1d(activation=SnakeBeta(channels, alpha_logscale=True)) for _ in range(self.num_layers)])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class GeneratorAdversarialLoss(torch.nn.Module):

    def __init__(self, average_by_discriminators=True, loss_type='mse'):
        """Initialize GeneratorAversarialLoss module."""
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ['mse', 'hinge'], f'{loss_type} is not supported.'
        if loss_type == 'mse':
            self.criterion = self._mse_loss
        else:
            self.criterion = self._hinge_loss

    def forward(self, outputs):
        """
        Calcualate generator adversarial loss.

        Args:
            outputs (Tensor or list): Discriminator outputs or list of
                discriminator outputs.

        Returns:
            Tensor: Generator adversarial loss value.
        """
        if isinstance(outputs, (tuple, list)):
            adv_loss = 0.0
            for i, outputs_ in enumerate(outputs):
                if isinstance(outputs_, (tuple, list)):
                    outputs_ = outputs_[-1]
                adv_loss = adv_loss + self.criterion(outputs_)
            if self.average_by_discriminators:
                adv_loss /= i + 1
        else:
            adv_loss = self.criterion(outputs)
        return adv_loss

    def _mse_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _hinge_loss(self, x):
        return -x.mean()


class DiscriminatorAdversarialLoss(torch.nn.Module):

    def __init__(self, average_by_discriminators=True, loss_type='mse'):
        super().__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ['mse', 'hinge'], f'{loss_type} is not supported.'
        if loss_type == 'mse':
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(self, outputs_hat, outputs):
        """
        Calcualate discriminator adversarial loss.

        Args:
            outputs_hat (Tensor or list): Discriminator outputs or list of
                discriminator outputs calculated from generator outputs.
            outputs (Tensor or list): Discriminator outputs or list of
                discriminator outputs calculated from groundtruth.

        Returns:
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.
        """
        if isinstance(outputs, (tuple, list)):
            real_loss = 0.0
            fake_loss = 0.0
            for i, (outputs_hat_, outputs_) in enumerate(zip(outputs_hat, outputs)):
                if isinstance(outputs_hat_, (tuple, list)):
                    outputs_hat_ = outputs_hat_[-1]
                    outputs_ = outputs_[-1]
                real_loss = real_loss + self.real_criterion(outputs_)
                fake_loss = fake_loss + self.fake_criterion(outputs_hat_)
            if self.average_by_discriminators:
                fake_loss /= i + 1
                real_loss /= i + 1
        else:
            real_loss = self.real_criterion(outputs)
            fake_loss = self.fake_criterion(outputs_hat)
        return real_loss + fake_loss

    def _mse_real_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x):
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_real_loss(self, x):
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x):
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


def _normalize(tensor, dim):
    denom = tensor.norm(p=2.0, dim=dim, keepdim=True).clamp_min(1e-12)
    return tensor / denom


class SANConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(SANConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=1, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        scale = self.weight.norm(p=2.0, dim=[1, 2], keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(in_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.normalize_weight()

    def forward(self, input, flg_train=False):
        if self.bias is not None:
            input = input + self.bias.view(self.in_channels, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1)
        if flg_train:
            out_fun = F.conv1d(input, normalized_weight.detach(), None, self.stride, self.padding, self.dilation, self.groups)
            out_dir = F.conv1d(input.detach(), normalized_weight, None, self.stride, self.padding, self.dilation, self.groups)
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = F.conv1d(input, normalized_weight, None, self.stride, self.padding, self.dilation, self.groups)
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=[1, 2])


class CoMBD(torch.nn.Module):

    def __init__(self, filters, kernels, groups, strides, use_spectral_norm=False):
        super(CoMBD, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList()
        init_channel = 1
        for i, (f, k, g, s) in enumerate(zip(filters, kernels, groups, strides)):
            self.convs.append(norm_f(Conv1d(init_channel, f, k, s, padding=get_padding(k, 1), groups=g)))
            init_channel = f
        self.conv_post = norm_f(SANConv1d(filters[-1], 1, 3, 1, padding=get_padding(3, 1)))

    def forward(self, x, discriminator_train_flag):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x, discriminator_train_flag)
        return x, fmap


class PQMF(torch.nn.Module):

    def __init__(self, N=4, taps=62, cutoff=0.15, beta=9.0):
        super(PQMF, self).__init__()
        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta
        QMF = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        H = np.zeros((N, len(QMF)))
        G = np.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (2 * k + 1) * (np.pi / (2 * N)) * (np.arange(taps + 1) - (taps - 1) / 2)
            phase = (-1) ** k * np.pi / 4
            H[k] = 2 * QMF * np.cos(constant_factor + phase)
            G[k] = 2 * QMF * np.cos(constant_factor - phase)
        H = torch.from_numpy(H[:, None, :]).float()
        G = torch.from_numpy(G[None, :, :]).float()
        self.register_buffer('H', H)
        self.register_buffer('G', G)
        updown_filter = torch.zeros((N, N, N)).float()
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer('updown_filter', updown_filter)
        self.N = N
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def forward(self, x):
        return self.analysis(x)

    def analysis(self, x):
        return F.conv1d(x, self.H, padding=self.taps // 2, stride=self.N)

    def synthesis(self, x):
        x = F.conv_transpose1d(x, self.updown_filter * self.N, stride=self.N)
        x = F.conv1d(x, self.G, padding=self.taps // 2)
        return x


class MultiCoMBDiscriminator(torch.nn.Module):

    def __init__(self, kernels, channels, groups, strides):
        super(MultiCoMBDiscriminator, self).__init__()
        self.combd_1 = CoMBD(filters=channels, kernels=kernels[0], groups=groups, strides=strides)
        self.combd_2 = CoMBD(filters=channels, kernels=kernels[1], groups=groups, strides=strides)
        self.combd_3 = CoMBD(filters=channels, kernels=kernels[2], groups=groups, strides=strides)
        self.pqmf_2 = PQMF(N=2, taps=256, cutoff=0.25, beta=10.0)
        self.pqmf_4 = PQMF(N=8, taps=192, cutoff=0.13, beta=10.0)

    def forward(self, wave_final, intermediate_wave_upsampled_twice=None, intermediate_wave_upsampled_once=None, discriminator_train_flag=False):
        if intermediate_wave_upsampled_twice is not None and intermediate_wave_upsampled_once is not None:
            features_of_predicted = []
            out3, p3_fmap_hat = self.combd_3(wave_final, discriminator_train_flag)
            features_of_predicted = features_of_predicted + p3_fmap_hat
            x2_hat_ = self.pqmf_2(wave_final)[:, :1, :]
            x1_hat_ = self.pqmf_4(wave_final)[:, :1, :]
            out2, p2_fmap_hat_ = self.combd_2(intermediate_wave_upsampled_twice, discriminator_train_flag)
            features_of_predicted = features_of_predicted + p2_fmap_hat_
            out1, p1_fmap_hat_ = self.combd_1(intermediate_wave_upsampled_once, discriminator_train_flag)
            features_of_predicted = features_of_predicted + p1_fmap_hat_
            out22, p2_fmap_hat = self.combd_2(x2_hat_, discriminator_train_flag)
            features_of_predicted = features_of_predicted + p2_fmap_hat
            out12, p1_fmap_hat = self.combd_1(x1_hat_, discriminator_train_flag)
            features_of_predicted = features_of_predicted + p1_fmap_hat
            return [out1, out12, out2, out22, out3], features_of_predicted
        else:
            features_of_gold = []
            out3, p3_fmap = self.combd_3(wave_final, discriminator_train_flag)
            features_of_gold = features_of_gold + p3_fmap
            x2_ = self.pqmf_2(wave_final)[:, :1, :]
            x1_ = self.pqmf_4(wave_final)[:, :1, :]
            out2, p2_fmap_ = self.combd_2(x2_, discriminator_train_flag)
            features_of_gold = features_of_gold + p2_fmap_
            out1, p1_fmap_ = self.combd_1(x1_, discriminator_train_flag)
            features_of_gold = features_of_gold + p1_fmap_
            out22, p2_fmap = self.combd_2(x2_, discriminator_train_flag)
            features_of_gold = features_of_gold + p2_fmap
            out12, p1_fmap = self.combd_1(x1_, discriminator_train_flag)
            features_of_gold = features_of_gold + p1_fmap
            return [out1, out12, out2, out22, out3], features_of_gold


class MDC(torch.nn.Module):

    def __init__(self, in_channel, channel, kernel, stride, dilations, use_spectral_norm=False):
        super(MDC, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch.nn.ModuleList()
        self.num_dilations = len(dilations)
        for d in dilations:
            self.convs.append(norm_f(Conv1d(in_channel, channel, kernel, stride=1, padding=get_padding(kernel, d), dilation=d)))
        self.conv_out = norm_f(SANConv1d(channel, channel, 3, stride=stride, padding=get_padding(3, 1)))

    def forward(self, x):
        xs = None
        for l in self.convs:
            if xs is None:
                xs = l(x)
            else:
                xs += l(x)
        x = xs / self.num_dilations
        x = self.conv_out(x)
        x = F.leaky_relu(x, 0.1)
        return x


class SubBandDiscriminator(torch.nn.Module):

    def __init__(self, init_channel, channels, kernel, strides, dilations, use_spectral_norm=False):
        super(SubBandDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.mdcs = torch.nn.ModuleList()
        for channel, stride, dilation in zip(channels, strides, dilations):
            self.mdcs.append(MDC(init_channel, channel, kernel, stride, dilation))
            init_channel = channel
        self.conv_post = norm_f(SANConv1d(init_channel, 1, 3, padding=get_padding(3, 1)))

    def forward(self, x, discriminator_train_flag):
        fmap = []
        for l in self.mdcs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x, discriminator_train_flag)
        return x, fmap


class MultiSubBandDiscriminator(torch.nn.Module):

    def __init__(self, tkernels, fkernel, tchannels, fchannels, tstrides, fstride, tdilations, fdilations, tsubband, n, m, freq_init_ch):
        super(MultiSubBandDiscriminator, self).__init__()
        self.fsbd = SubBandDiscriminator(init_channel=freq_init_ch, channels=fchannels, kernel=fkernel, strides=fstride, dilations=fdilations)
        self.tsubband1 = tsubband[0]
        self.tsbd1 = SubBandDiscriminator(init_channel=self.tsubband1, channels=tchannels, kernel=tkernels[0], strides=tstrides[0], dilations=tdilations[0])
        self.tsubband2 = tsubband[1]
        self.tsbd2 = SubBandDiscriminator(init_channel=self.tsubband2, channels=tchannels, kernel=tkernels[1], strides=tstrides[1], dilations=tdilations[1])
        self.tsubband3 = tsubband[2]
        self.tsbd3 = SubBandDiscriminator(init_channel=self.tsubband3, channels=tchannels, kernel=tkernels[2], strides=tstrides[2], dilations=tdilations[2])
        self.pqmf_n = PQMF(N=n, taps=256, cutoff=0.03, beta=10.0)
        self.pqmf_m = PQMF(N=m, taps=256, cutoff=0.1, beta=9.0)

    def forward(self, wave, discriminator_train_flag):
        fmap_hat = []
        xn_hat = self.pqmf_n(wave)
        q3_hat, feat_q3_hat = self.tsbd3(xn_hat[:, :self.tsubband3, :], discriminator_train_flag)
        fmap_hat = fmap_hat + feat_q3_hat
        q2_hat, feat_q2_hat = self.tsbd2(xn_hat[:, :self.tsubband2, :], discriminator_train_flag)
        fmap_hat = fmap_hat + feat_q2_hat
        q1_hat, feat_q1_hat = self.tsbd1(xn_hat[:, :self.tsubband1, :], discriminator_train_flag)
        fmap_hat = fmap_hat + feat_q1_hat
        xm_hat = self.pqmf_m(wave)
        xm_hat = xm_hat.transpose(-2, -1)
        q4_hat, feat_q4_hat = self.fsbd(xm_hat, discriminator_train_flag)
        fmap_hat = fmap_hat + feat_q4_hat
        return [q1_hat, q2_hat, q3_hat, q4_hat], fmap_hat


class BigVGAN(torch.nn.Module):

    def __init__(self, num_mels=128, upsample_initial_channel=1024, upsample_rates=(8, 6, 2, 2, 2), upsample_kernel_sizes=(16, 12, 4, 4, 4), resblock_kernel_sizes=(3, 7, 11), resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)), weights=None):
        super(BigVGAN, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(num_mels, upsample_initial_channel, 7, 1, padding=3))
        self.ups = ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(ModuleList([weight_norm(ConvTranspose1d(upsample_initial_channel // 2 ** i, upsample_initial_channel // 2 ** (i + 1), k, u, padding=(k - u) // 2))]))
        self.resblocks = ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(ch, k, d))
        activation_post = SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)
        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        None
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class FeatureMatchLoss(torch.nn.Module):

    def __init__(self, average_by_layers=True, average_by_discriminators=False, include_final_outputs=False):
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(self, feats_hat, feats):
        """
        Calculate feature matching loss.

        Args:
            feats_hat (list): List of lists of discriminator outputs
                calculated from generator outputs.
            feats (list): List of lists of discriminator outputs
                calculated from ground-truth.

        Returns:
            Tensor: Feature matching loss value.
        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            feat_match_loss_ = 0.0
            if not self.include_final_outputs:
                feats_hat_ = feats_hat_[:-1]
                feats_ = feats_[:-1]
            for j, (feat_hat_, feat_) in enumerate(zip(feats_hat_, feats_)):
                feat_match_loss_ += F.l1_loss(feat_hat_, feat_.detach())
            if self.average_by_layers:
                feat_match_loss_ /= j + 1
            feat_match_loss += feat_match_loss_
        if self.average_by_discriminators:
            feat_match_loss /= i + 1
        return feat_match_loss


class CodecSimulator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = torchaudio.transforms.MuLawEncoding(quantization_channels=64)
        self.decoder = torchaudio.transforms.MuLawDecoding(quantization_channels=64)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SANConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(SANConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=1, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
        scale = self.weight.norm(p=2.0, dim=[1, 2, 3], keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale.view(out_channels))
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(in_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.normalize_weight()

    def forward(self, input, flg_train=False):
        if self.bias is not None:
            input = input + self.bias.view(self.in_channels, 1, 1)
        normalized_weight = self._get_normalized_weight()
        scale = self.scale.view(self.out_channels, 1, 1)
        if flg_train:
            out_fun = F.conv2d(input, normalized_weight.detach(), None, self.stride, self.padding, self.dilation, self.groups)
            out_dir = F.conv2d(input.detach(), normalized_weight, None, self.stride, self.padding, self.dilation, self.groups)
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = F.conv2d(input, normalized_weight, None, self.stride, self.padding, self.dilation, self.groups)
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = self._get_normalized_weight()

    def _get_normalized_weight(self):
        return _normalize(self.weight, dim=[1, 2, 3])


class HiFiGANPeriodDiscriminator(torch.nn.Module):

    def __init__(self, in_channels=1, out_channels=1, period=3, kernel_sizes=(5, 3), channels=32, downsample_scales=(3, 3, 3, 3, 1), max_downsample_channels=1024, bias=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}, use_weight_norm=True, use_spectral_norm=False):
        """
        Initialize HiFiGANPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, 'Kernel size must be odd number.'
        assert kernel_sizes[1] % 2 == 1, 'Kernel size must be odd number.'
        self.period = period
        self.convs = torch.nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [torch.nn.Sequential(torch.nn.Conv2d(in_chs, out_chs, (kernel_sizes[0], 1), (downsample_scale, 1), padding=((kernel_sizes[0] - 1) // 2, 0)), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
            in_chs = out_chs
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = SANConv2d(out_chs, out_channels, (kernel_sizes[1] - 1, 1), 1, padding=((kernel_sizes[1] - 1) // 2, 0))
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Either use use_weight_norm or use_spectral_norm.')
        if use_weight_norm:
            self.apply_weight_norm()
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, discriminator_train_flag):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).

        Returns:
            list: List of each layer's tensors.
        """
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        outs = []
        for layer in self.convs:
            x = layer(x)
            outs = outs + [x]
        x = self.output_conv(x, discriminator_train_flag)
        return x, outs

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.
        """

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """
        Apply spectral normalization module from all of the layers.
        """

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, periods=(2, 3, 5, 7, 11), discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [5, 3], 'channels': 32, 'downsample_scales': [3, 3, 3, 3, 1], 'max_downsample_channels': 1024, 'bias': True, 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}, 'use_weight_norm': True, 'use_spectral_norm': False}):
        """
        Initialize HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params['period'] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x, discriminator_train_flag):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        feats = []
        for f in self.discriminators:
            d_out, d_feats = f(x, discriminator_train_flag)
            outs = outs + [d_out]
            feats = feats + d_feats
        return outs, feats


class HiFiGANScaleDiscriminator(torch.nn.Module):

    def __init__(self, in_channels=1, out_channels=1, kernel_sizes=(15, 41, 5, 3), channels=128, max_downsample_channels=1024, max_groups=16, bias=True, downsample_scales=(2, 2, 4, 4, 1), nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}, use_weight_norm=True, use_spectral_norm=False):
        """
        Initialize HiFiGAN scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of four kernel sizes. The first will be used for the first conv layer,
                and the second is for downsampling part, and the remaining two are for output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1
        self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_channels, channels, kernel_sizes[0], bias=bias, padding=(kernel_sizes[0] - 1) // 2), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
        in_chs = channels
        out_chs = channels
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs, out_chs, kernel_size=kernel_sizes[1], stride=downsample_scale, padding=(kernel_sizes[1] - 1) // 2, groups=groups, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
            in_chs = out_chs
            out_chs = min(in_chs * 2, max_downsample_channels)
            groups = min(groups * 4, max_groups)
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [torch.nn.Sequential(torch.nn.Conv1d(in_chs, out_chs, kernel_size=kernel_sizes[2], stride=1, padding=(kernel_sizes[2] - 1) // 2, bias=bias), getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params))]
        self.post_conv = SANConv1d(out_chs, out_channels, kernel_sizes[3], padding=(kernel_sizes[3] - 1) // 2)
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Either use use_weight_norm or use_spectral_norm.')
        if use_weight_norm:
            self.apply_weight_norm()
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, discriminator_train_flag):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of output tensors of each layer.
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs = outs + [x]
        x = self.post_conv(x, discriminator_train_flag)
        return x, outs

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.
        """

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """
        Apply spectral normalization module from all of the layers.
        """

        def _apply_spectral_norm(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.spectral_norm(m)
        self.apply(_apply_spectral_norm)


class HiFiGANMultiScaleDiscriminator(torch.nn.Module):

    def __init__(self, scales=3, downsample_pooling='AvgPool1d', downsample_pooling_params={'kernel_size': 4, 'stride': 2, 'padding': 2}, discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [15, 41, 5, 3], 'channels': 128, 'max_downsample_channels': 1024, 'max_groups': 16, 'bias': True, 'downsample_scales': [2, 2, 4, 4, 1], 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}}, follow_official_norm=False):
        """
        Initialize HiFiGAN multi-scale discriminator module.

        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
        """
        super().__init__()
        self.discriminators = torch.nn.ModuleList()
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params['use_weight_norm'] = False
                    params['use_spectral_norm'] = True
                else:
                    params['use_weight_norm'] = True
                    params['use_spectral_norm'] = False
            self.discriminators += [HiFiGANScaleDiscriminator(**params)]
        self.pooling = getattr(torch.nn, downsample_pooling)(**downsample_pooling_params)

    def forward(self, x, discriminator_train_flag):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        feats = []
        for f in self.discriminators:
            out, d_feats = f(x, discriminator_train_flag)
            feats = feats + d_feats
            outs = outs + [out]
            x = self.pooling(x)
        return outs, feats


class HiFiGANMultiScaleMultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self, scales=3, scale_downsample_pooling='AvgPool1d', scale_downsample_pooling_params={'kernel_size': 4, 'stride': 2, 'padding': 2}, scale_discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [15, 41, 5, 3], 'channels': 128, 'max_downsample_channels': 1024, 'max_groups': 16, 'bias': True, 'downsample_scales': [4, 4, 4, 4, 1], 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}}, follow_official_norm=True, periods=[2, 3, 5, 7, 11], period_discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [5, 3], 'channels': 32, 'downsample_scales': [3, 3, 3, 3, 1], 'max_downsample_channels': 1024, 'bias': True, 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}, 'use_weight_norm': True, 'use_spectral_norm': False}):
        """
        Initialize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
        """
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(scales=scales, downsample_pooling=scale_downsample_pooling, downsample_pooling_params=scale_downsample_pooling_params, discriminator_params=scale_discriminator_params, follow_official_norm=follow_official_norm)
        self.mpd = HiFiGANMultiPeriodDiscriminator(periods=periods, discriminator_params=period_discriminator_params)

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.
        """
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        return msd_outs + mpd_outs


class AvocodoHiFiGANJointDiscriminator(torch.nn.Module):
    """
    Contradicting the legacy name, the Avocodo parts were removed again for stability
    """

    def __init__(self, scales=4, scale_downsample_pooling='AvgPool1d', scale_downsample_pooling_params={'kernel_size': 4, 'stride': 2, 'padding': 2}, scale_discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [15, 41, 5, 3], 'channels': 128, 'max_downsample_channels': 1024, 'max_groups': 16, 'bias': True, 'downsample_scales': [4, 4, 4, 1], 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}}, follow_official_norm=True, periods=(2, 3, 5, 7, 11), period_discriminator_params={'in_channels': 1, 'out_channels': 1, 'kernel_sizes': [5, 3], 'channels': 32, 'downsample_scales': [3, 3, 3, 1], 'max_downsample_channels': 1024, 'bias': True, 'nonlinear_activation': 'LeakyReLU', 'nonlinear_activation_params': {'negative_slope': 0.1}, 'use_weight_norm': True, 'use_spectral_norm': False}):
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(scales=scales, downsample_pooling=scale_downsample_pooling, downsample_pooling_params=scale_downsample_pooling_params, discriminator_params=scale_discriminator_params, follow_official_norm=follow_official_norm)
        self.mpd = HiFiGANMultiPeriodDiscriminator(periods=periods, discriminator_params=period_discriminator_params)

    def forward(self, wave, discriminator_train_flag=False):
        """
        Calculate forward propagation.

        Args:
            wave: The predicted or gold waveform
            intermediate_wave_upsampled_twice: the wave before the final upsampling in the generator
            intermediate_wave_upsampled_once: the wave before the second final upsampling in the generator

        Returns:
            List: List of lists of each discriminator outputs,
                which consists of each layer's output tensors.
        """
        msd_outs, msd_feats = self.msd(wave, discriminator_train_flag)
        mpd_outs, mpd_feats = self.mpd(wave, discriminator_train_flag)
        return msd_outs + mpd_outs, msd_feats + mpd_feats


class HiFiGAN(torch.nn.Module):

    def __init__(self, in_channels=128, out_channels=1, channels=768, kernel_size=7, upsample_scales=(8, 6, 2, 2, 2), upsample_kernel_sizes=(16, 12, 4, 4, 4), resblock_kernel_sizes=(3, 7, 11), resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)), use_additional_convs=True, bias=True, nonlinear_activation='LeakyReLU', nonlinear_activation_params={'negative_slope': 0.1}, weights=None):
        """
        Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()
        assert kernel_size % 2 == 1, 'Kernel size must be odd number.'
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_kernel_sizes)
        self.num_blocks = len(resblock_kernel_sizes)
        self.input_conv = torch.nn.Conv1d(in_channels, channels, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.upsamples = torch.nn.ModuleList()
        self.blocks = torch.nn.ModuleList()
        for i in range(len(upsample_kernel_sizes)):
            self.upsamples += [torch.nn.Sequential(getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params), torch.nn.ConvTranspose1d(channels // 2 ** i, channels // 2 ** (i + 1), upsample_kernel_sizes[i], upsample_scales[i], padding=(upsample_kernel_sizes[i] - upsample_scales[i]) // 2))]
            for j in range(len(resblock_kernel_sizes)):
                self.blocks += [ResidualBlock(kernel_size=resblock_kernel_sizes[j], channels=channels // 2 ** (i + 1), dilations=resblock_dilations[j], bias=bias, use_additional_convs=use_additional_convs, nonlinear_activation=nonlinear_activation, nonlinear_activation_params=nonlinear_activation_params)]
        self.output_conv = torch.nn.Sequential(torch.nn.LeakyReLU(), torch.nn.Conv1d(channels // 2 ** (i + 1), out_channels, kernel_size, 1, padding=(kernel_size - 1) // 2), torch.nn.Tanh())
        self.apply_weight_norm()
        self.reset_parameters()
        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, c):
        """
        Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).
            Tensor: intermediate result
            Tensor: another intermediate result
        """
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](c)
            cs = 0.0
            for j in range(self.num_blocks):
                cs += self.blocks[i * self.num_blocks + j](c)
            c = cs / self.num_blocks
        c = self.output_conv(c)
        return c

    def reset_parameters(self):
        """
        Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py
        """

        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
        self.apply(_reset_parameters)

    def remove_weight_norm(self):
        """
        Remove weight normalization module from all of the layers.
        """

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return
        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """
        Apply weight normalization module from all of the layers.
        """

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
        self.apply(_apply_weight_norm)

    def inference(self, c, normalize_before=False):
        """
        Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
            normalize_before (bool): Whether to perform normalization.

        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).
        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float)
        if normalize_before:
            c = (c - self.mean) / self.scale
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)


class MelSpectrogramLoss(torch.nn.Module):

    def __init__(self, fs=24000, fft_size=1024, hop_size=256, win_length=None, window='hann', num_mels=128, fmin=20, fmax=None, center=True, normalized=False, onesided=True, eps=1e-10, log_base=10.0):
        super().__init__()
        self.mel_spectrogram = MelSpectrogram(fs=fs, fft_size=fft_size, hop_size=hop_size, win_length=win_length, window=window, num_mels=num_mels, fmin=fmin, fmax=fmax, center=center, normalized=normalized, onesided=onesided, eps=eps, log_base=log_base)

    def forward(self, y_hat, y):
        """
        Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: Mel-spectrogram loss value.
        """
        mel_hat = self.mel_spectrogram(y_hat)
        mel = self.mel_spectrogram(y)
        mel_loss = F.l1_loss(mel_hat, mel)
        return mel_loss


class SANEmbedding(nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None):
        super(SANEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight, device=device, dtype=dtype)
        scale = self.weight.norm(p=2.0, dim=1, keepdim=True).clamp_min(1e-12)
        self.weight = nn.parameter.Parameter(self.weight / scale.expand_as(self.weight))
        self.scale = nn.parameter.Parameter(scale)

    def forward(self, input, flg_train=False):
        out = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        out = _normalize(out, dim=-1)
        scale = F.embedding(input, self.scale, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        if flg_train:
            out_fun = out.detach()
            out_dir = out
            out = [out_fun * scale, out_dir * scale.detach()]
        else:
            out = out * scale
        return out

    @torch.no_grad()
    def normalize_weight(self):
        self.weight.data = _normalize(self.weight, dim=1)


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_distributed():
    return world_size() > 1


def all_reduce(tensor: 'torch.Tensor', op=torch.distributed.ReduceOp.SUM):
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)


def _check_number_of_params(params: 'tp.List[torch.Tensor]'):
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        raise RuntimeError(f'Mismatch in number of params: ours is {len(params)}, at least one worker has a different one.')


def _is_complex_or_float(tensor):
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def broadcast_tensors(tensors: 'tp.Iterable[torch.Tensor]', src: 'int'=0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def ema_inplace(moving_avg, new, decay: 'float'):
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)


def sample_vectors(samples, num: 'int'):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def kmeans(samples, num_clusters: 'int', num_iters: 'int'=10):
    dim, dtype = samples.shape[-1], samples.dtype
    means = sample_vectors(samples, num_clusters)
    for _ in range(num_iters):
        diffs = rearrange(samples, 'n d -> n () d') - rearrange(means, 'c d -> () c d')
        dists = -(diffs ** 2).sum(dim=-1)
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


def laplace_smoothing(x, n_categories: 'int', epsilon: 'float'=1e-05):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """

    def __init__(self, dim: 'int', codebook_size: 'int', kmeans_init: 'int'=False, kmeans_iters: 'int'=10, decay: 'float'=0.99, epsilon: 'float'=1e-05, threshold_ema_dead_code: 'int'=2):
        super().__init__()
        self.decay = decay
        init_fn: 'tp.Union[tp.Callable[..., torch.Tensor], tp.Any]' = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.register_buffer('inited', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(mask[..., None], sample_vectors(samples, self.codebook_size), self.embed)
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace_(batch_samples, mask=expired_codes)
        broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, '... d -> (...) d')
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)
        self.init_embed_(x)
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)
        if self.training:
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
        return quantize, embed_ind


def default(val: 'tp.Any', d: 'tp.Any') ->tp.Any:
    return val if val is not None else d


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """

    def __init__(self, dim: 'int', codebook_size: 'int', codebook_dim: 'tp.Optional[int]'=None, decay: 'float'=0.99, epsilon: 'float'=1e-05, kmeans_init: 'bool'=True, kmeans_iters: 'int'=50, threshold_ema_dead_code: 'int'=2, commitment_weight: 'float'=1.0):
        super().__init__()
        _codebook_dim: 'int' = default(codebook_dim, dim)
        requires_projection = _codebook_dim != dim
        self.project_in = nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters, decay=decay, epsilon=epsilon, threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, 'b d n -> b n d')
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        if len(quantize.size()) < 3:
            quantize = quantize.unsqueeze(0)
        quantize = rearrange(quantize, 'b n d -> b d n')
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, 'b d n -> b n d')
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            quantize = x + (quantize - x).detach()
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        if self.training:
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, 'b n d -> b d n')
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x, n_q: 'tp.Optional[int]'=None):
        quantized_out = 0.0
        residual = x
        all_losses = []
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)
        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: 'torch.Tensor', n_q: 'tp.Optional[int]'=None, st: 'tp.Optional[int]'=None) ->torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        st = st or 0
        for layer in self.layers[st:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: 'torch.Tensor') ->torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm', 'time_layer_norm', 'layer_norm', 'time_group_norm'])


def apply_parametrization_norm(module: 'nn.Module', norm: 'str'='none') ->nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        return module


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """

    def __init__(self, normalized_shape: 'tp.Union[int, tp.List[int], torch.Size]', **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return


def get_norm_module(module: 'nn.Module', causal: 'bool'=False, norm: 'str'='none', **norm_kwargs) ->nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, causal: bool=False, norm: str='none', norm_kwargs: tp.Dict[str, tp.Any]={}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


def get_extra_padding_for_conv1d(x: 'torch.Tensor', kernel_size: 'int', stride: 'int', padding_total: 'int'=0) ->int:
    """See `pad_for_conv1d`.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: 'torch.Tensor', paddings: 'tp.Tuple[int, int]', mode: 'str'='zero', value: 'float'=0.0):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, dilation: 'int'=1, groups: 'int'=1, bias: 'bool'=True, causal: 'bool'=False, norm: 'str'='none', norm_kwargs: 'tp.Dict[str, tp.Any]'={}, pad_mode: 'str'='reflect'):
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(f'SConv1d has been initialized with stride > 1 and dilation > 1 (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, groups=groups, bias=bias, causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        padding_total = (kernel_size - 1) * dilation - (stride - 1)
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, causal: bool=False, norm: str='none', norm_kwargs: tp.Dict[str, tp.Any]={}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


def unpad1d(x: 'torch.Tensor', paddings: 'tp.Tuple[int, int]'):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert padding_left + padding_right <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, causal: 'bool'=False, norm: 'str'='none', trim_right_ratio: 'float'=1.0, norm_kwargs: 'tp.Dict[str, tp.Any]'={}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride, causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, '`trim_right_ratio` != 1.0 only makes sense for causal convolutions'
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        y = self.convtr(x)
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    Args:
        dim (int): Dimension of the input/output
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3)
        true_skip (bool): Whether to use true skip connection or a simple convolution as the skip connection.
    """

    def __init__(self, dim: 'int', kernel_sizes: 'tp.List[int]'=[3, 1], dilations: 'tp.List[int]'=[1, 1], activation: 'str'='ELU', activation_params: 'dict'={'alpha': 1.0}, norm: 'str'='weight_norm', norm_params: 'tp.Dict[str, tp.Any]'={}, causal: 'bool'=False, pad_mode: 'str'='reflect', compress: 'int'=2, true_skip: 'bool'=True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [act(**activation_params), SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation, norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
        self.block = nn.Sequential(*block)
        self.shortcut: 'nn.Module'
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(self, dimension: 'int', num_layers: 'int'=2, skip: 'bool'=True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class SEANetDecoder(nn.Module):
    """SEANet decoder.
    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """

    def __init__(self, channels: 'int'=1, dimension: 'int'=128, n_filters: 'int'=32, n_residual_layers: 'int'=1, ratios: 'tp.List[int]'=[8, 5, 4, 2], activation: 'str'='ELU', activation_params: 'dict'={'alpha': 1.0}, final_activation: 'tp.Optional[str]'=None, final_activation_params: 'tp.Optional[dict]'=None, norm: 'str'='weight_norm', norm_params: 'tp.Dict[str, tp.Any]'={}, kernel_size: 'int'=7, last_kernel_size: 'int'=7, residual_kernel_size: 'int'=3, dilation_base: 'int'=2, causal: 'bool'=False, pad_mode: 'str'='reflect', true_skip: 'bool'=False, compress: 'int'=2, lstm: 'int'=2, trim_right_ratio: 'float'=1.0):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: 'tp.List[nn.Module]' = [SConv1d(dimension, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]
        for i, ratio in enumerate(self.ratios):
            model += [act(**activation_params), SConvTranspose1d(mult * n_filters, mult * n_filters // 2, kernel_size=ratio * 2, stride=ratio, norm=norm, norm_kwargs=norm_params, causal=causal, trim_right_ratio=trim_right_ratio)]
            for j in range(n_residual_layers):
                model += [SEANetResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernel_size, 1], dilations=[dilation_base ** j, 1], activation=activation, activation_params=activation_params, norm=norm, norm_params=norm_params, causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]
            mult //= 2
        model += [act(**activation_params), SConv1d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        y = self.model(z)
        return y


class SEANetEncoder(nn.Module):
    """SEANet encoder.
    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
    """

    def __init__(self, channels: 'int'=1, dimension: 'int'=128, n_filters: 'int'=32, n_residual_layers: 'int'=1, ratios: 'tp.List[int]'=[8, 5, 4, 2], activation: 'str'='ELU', activation_params: 'dict'={'alpha': 1.0}, norm: 'str'='weight_norm', norm_params: 'tp.Dict[str, tp.Any]'={}, kernel_size: 'int'=7, last_kernel_size: 'int'=7, residual_kernel_size: 'int'=3, dilation_base: 'int'=2, causal: 'bool'=False, pad_mode: 'str'='reflect', true_skip: 'bool'=False, compress: 'int'=2, lstm: 'int'=2):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        act = getattr(nn, activation)
        mult = 1
        model: 'tp.List[nn.Module]' = [SConv1d(channels, mult * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
        for i, ratio in enumerate(self.ratios):
            for j in range(n_residual_layers):
                model += [SEANetResnetBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1], dilations=[dilation_base ** j, 1], norm=norm, norm_params=norm_params, activation=activation, activation_params=activation_params, causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip)]
            model += [act(**activation_params), SConv1d(mult * n_filters, mult * n_filters * 2, kernel_size=ratio * 2, stride=ratio, norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
            mult *= 2
        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]
        model += [act(**activation_params), SConv1d(mult * n_filters, dimension, last_kernel_size, norm=norm, norm_kwargs=norm_params, causal=causal, pad_mode=pad_mode)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class EnCodec(nn.Module):

    def __init__(self, n_filters, D, target_bandwidths=[1, 1.5, 2, 4, 6, 12], ratios=[8, 5, 4, 2], sample_rate=16000, bins=1024, normalize=False):
        super().__init__()
        self.hop_length = np.prod(ratios)
        self.encoder = SEANetEncoder(n_filters=n_filters, dimension=D, ratios=ratios)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))
        self.bits_per_codebook = int(math.log2(bins))
        self.target_bandwidths = target_bandwidths
        self.quantizer = ResidualVectorQuantizer(dimension=D, n_q=n_q, bins=bins)
        self.decoder = SEANetDecoder(n_filters=n_filters, dimension=D, ratios=ratios)

    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def forward(self, x):
        e = self.encoder(x)
        max_idx = len(self.target_bandwidths) - 1
        bw = self.target_bandwidths[random.randint(0, max_idx)]
        quantized, codes, bandwidth, commit_loss = self.quantizer(e, self.frame_rate, bw)
        o = self.decoder(quantized)
        return o, commit_loss, None

    def encode(self, x, target_bw=None, st=None):
        e = self.encoder(x)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        if st is None:
            st = 0
        codes = self.quantizer.encode(e, self.frame_rate, bw, st)
        return codes

    def decode(self, codes):
        quantized = self.quantizer.decode(codes)
        o = self.decoder(quantized)
        return o


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, norm: str='none', norm_kwargs: tp.Dict[str, tp.Any]={}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Module):
    """Wrapper around ConvTranspose2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self, *args, norm: str='none', norm_kwargs: tp.Dict[str, tp.Any]={}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MetricsCombiner(torch.nn.Module):

    def __init__(self, m):
        super().__init__()
        self.scoring_function = kan.KAN(width=[3, 5, 1], grid=5, k=5, seed=m, auto_save=False, device=DEVICE)
        self.scoring_function.speed(compile=True)

    def forward(self, x):
        return self.scoring_function(x)


class EnsembleModel(torch.nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x):
        x = x
        distances = list()
        for model in self.models:
            distances.append(model(x))
        return sum(distances) / len(distances)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class RedundancyReduction(torch.nn.Module):

    def __init__(self, lambd=1e-05, vector_dimensions=256):
        super().__init__()
        self.lambd = lambd
        self.bn = torch.nn.BatchNorm1d(vector_dimensions, affine=False)

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))
        off_diag = off_diagonal(c).pow_(2).sum()
        return self.lambd * off_diag


class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambd=1e-05, vector_dimensions=256):
        super().__init__()
        self.lambd = lambd
        self.bn = torch.nn.BatchNorm1d(vector_dimensions, affine=False)

    def forward(self, z1, z2):
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.size(0))
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class TripletLoss(torch.nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.cosine_similarity = torch.nn.CosineSimilarity()
        self.margin = margin

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        positive_distance = 1 - self.cosine_similarity(anchor_embeddings, positive_embeddings)
        negative_distance = 1 - self.cosine_similarity(anchor_embeddings, negative_embeddings)
        losses = torch.max(positive_distance - negative_distance + self.margin, torch.full_like(positive_distance, 0))
        return torch.mean(losses)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM(torch.nn.Module):
    """
    Adapted from https://github.com/Po-Hsun-Su/pytorch-ssim
    """

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (ActNorm,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (AdaIN1d,
     lambda: ([], {'style_dim': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {})),
    (AvocodoHiFiGANJointDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {})),
    (BatchNormConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ConditionalLayerNorm,
     lambda: ([], {'hidden_dim': 4, 'speaker_embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Conv1d1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'bias': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (Conv1dLinear,
     lambda: ([], {'in_chans': 4, 'hidden_chans': 4, 'kernel_size': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (ConvolutionModule,
     lambda: ([], {'channels': 4, 'kernel_size': 1}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (DiscriminatorAdversarialLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (DurationCalculator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (DurationPredictorLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FeatureMatchLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (GeneratorAdversarialLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (HiFiGANPeriodDiscriminator,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4]), 0], {})),
    (InvConv,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (InvConvNear,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (LayerNorm,
     lambda: ([], {'nout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (MDC,
     lambda: ([], {'in_channel': 4, 'channel': 4, 'kernel': 4, 'stride': 1, 'dilations': [4, 4]}),
     lambda: ([torch.rand([4, 4])], {})),
    (MultiHeadAttention,
     lambda: ([], {'channels': 4, 'out_channels': 4, 'n_heads': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (MultiHeadedAttention,
     lambda: ([], {'n_head': 4, 'n_feat': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (MultiLayeredConv1d,
     lambda: ([], {'in_chans': 4, 'hidden_chans': 4, 'kernel_size': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4])], {})),
    (MultiSequential,
     lambda: ([], {}),
     lambda: ([], {})),
    (NormConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (NormConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (NormConvTranspose1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (NormConvTranspose2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (PQMF,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {})),
    (PositionwiseFeedForward,
     lambda: ([], {'idim': 4, 'hidden_units': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RelPositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResBlk,
     lambda: ([], {'dim_in': 4, 'dim_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNetBlock,
     lambda: ([], {'fin': 4, 'fout': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (ResNet_G,
     lambda: ([], {'data_dim': 4, 'z_dim': 4, 'size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (RotaryPositionalEmbeddings,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SANConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SANConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (SConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SConvTranspose1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (SEANetDecoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 4])], {})),
    (SEANetEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 4])], {})),
    (SEANetResnetBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SLSTM,
     lambda: ([], {'dimension': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SequentialWrappableConditionalLayerNorm,
     lambda: ([], {'hidden_dim': 4, 'speaker_embedding_dim': 4}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {})),
    (SnakeBeta,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (StyleAdaptiveLayerNorm,
     lambda: ([], {'in_channels': 4, 'cond_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TimestepEmbedding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'filter_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Transpose,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (TripletLoss,
     lambda: ([], {'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
]

