
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


from typing import Tuple


import numpy as np


from torch import nn


from torch import Tensor


from torch.nn import Module


import torch.nn.functional as F


from collections import namedtuple


from functools import wraps


from torch import einsum


import math


import copy


from random import random


from functools import partial


from torch.optim import Adam


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from typing import Callable


from typing import List


from typing import Optional


from torch.nn.utils.rnn import pad_sequence


def exists(val):
    return val is not None


class ForwardSumLoss(Module):

    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.blank_logprob = blank_logprob
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, attn_logprob, key_lens, query_lens):
        device, blank_logprob = attn_logprob.device, self.blank_logprob
        max_key_len = attn_logprob.size(-1)
        attn_logprob = rearrange(attn_logprob, 'b 1 c t -> c b t')
        attn_logprob = F.pad(attn_logprob, (1, 0, 0, 0, 0, 0), value=blank_logprob)
        mask_value = -torch.finfo(attn_logprob.dtype).max
        attn_logprob.masked_fill_(torch.arange(max_key_len + 1, device=device, dtype=torch.long).view(1, 1, -1) > key_lens.view(1, -1, 1), mask_value)
        attn_logprob = attn_logprob.log_softmax(dim=-1)
        target_seqs = torch.arange(1, max_key_len + 1, device=device, dtype=torch.long)
        target_seqs = repeat(target_seqs, 'n -> b n', b=key_lens.numel())
        cost = self.ctc_loss(attn_logprob, target_seqs, query_lens, key_lens)
        return cost


class BinLoss(Module):

    def forward(self, attn_hard, attn_logprob, key_lens):
        batch, device = attn_logprob.shape[0], attn_logprob.device
        max_key_len = attn_logprob.size(-1)
        attn_logprob = rearrange(attn_logprob, 'b 1 c t -> c b t')
        attn_hard = rearrange(attn_hard, 'b t c -> c b t')
        mask_value = -torch.finfo(attn_logprob.dtype).max
        attn_logprob.masked_fill_(torch.arange(max_key_len, device=device, dtype=torch.long).view(1, 1, -1) > key_lens.view(1, -1, 1), mask_value)
        attn_logprob = attn_logprob.log_softmax(dim=-1)
        return (attn_hard * attn_logprob).sum() / batch


def pad_tensor(input, pad, value=0):
    pad = [item for sublist in reversed(pad) for item in sublist]
    assert len(pad) // 2 == len(input.shape), 'Padding dimensions do not match input dimensions'
    return F.pad(input, pad, mode='constant', value=value)


def maximum_path(value, mask, const=None):
    device = value.device
    dtype = value.dtype
    if not exists(const):
        const = torch.tensor(float('-inf'))
    value = value * mask
    b, t_x, t_y = value.shape
    direction = torch.zeros(value.shape, dtype=torch.int64, device=device)
    v = torch.zeros((b, t_x), dtype=torch.float32, device=device)
    x_range = torch.arange(t_x, dtype=torch.float32, device=device).view(1, -1)
    for j in range(t_y):
        v0 = pad_tensor(v, ((0, 0), (1, 0)), value=const)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = torch.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask
        index_mask = x_range <= j
        v = torch.where(index_mask.view(1, -1), v_max + value[:, :, j], const)
    direction = torch.where(mask.bool(), direction, 1)
    path = torch.zeros(value.shape, dtype=torch.float32, device=device)
    index = mask[:, :, 0].sum(1).long() - 1
    index_range = torch.arange(b, device=device)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.float()
    path = path
    return path


class Aligner(Module):

    def __init__(self, dim_in, dim_hidden, attn_channels=80, temperature=0.0005):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.attn_channels = attn_channels
        self.temperature = temperature
        self.aligner = AlignerNet(dim_in=self.dim_in, dim_hidden=self.dim_hidden, attn_channels=self.attn_channels, temperature=self.temperature)

    def forward(self, x, x_mask, y, y_mask):
        alignment_soft, alignment_logprob = self.aligner(y, rearrange(x, 'b d t -> b t d'), x_mask)
        x_mask = rearrange(x_mask, '... i -> ... i 1')
        y_mask = rearrange(y_mask, '... j -> ... 1 j')
        attn_mask = x_mask * y_mask
        attn_mask = rearrange(attn_mask, 'b 1 i j -> b i j')
        alignment_soft = rearrange(alignment_soft, 'b 1 c t -> b t c')
        alignment_mask = maximum_path(alignment_soft, attn_mask)
        alignment_hard = torch.sum(alignment_mask, -1).int()
        return alignment_hard, alignment_soft, alignment_logprob, alignment_mask


Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner


print_once = once(print)


class Attend(nn.Module):

    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.register_buffer('mask', None, persistent=False)
        self.use_flash = use_flash
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'
        self.cpu_config = Config(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not use_flash:
            return
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer('mask', mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda
        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)
        config = self.cuda_config if is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal)
        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        n, device = q.shape[-2], q.device
        scale = q.shape[-1] ** -0.5
        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'
        sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        return out


def divisible_by(num, den):
    return num % den == 0


class LearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class AudioToMel(nn.Module):

    def __init__(self, *, n_mels=100, sampling_rate=24000, f_max=8000, n_fft=1024, win_length=640, hop_length=160, log=True):
        super().__init__()
        self.log = log
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_max = f_max
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

    def forward(self, audio):
        stft_transform = T.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, window_fn=torch.hann_window)
        spectrogram = stft_transform(audio)
        mel_transform = T.MelScale(n_mels=self.n_mels, sample_rate=self.sampling_rate, n_stft=self.n_fft // 2 + 1, f_max=self.f_max)
        mel = mel_transform(spectrogram)
        if self.log:
            mel = T.AmplitudeToDB()(mel)
        return mel


class CausalConv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kernel_size, = self.kernel_size
        dilation, = self.dilation
        stride, = self.stride
        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


_DEF_PUNCS = ';:,.!?¡¿—…"«»“”'


_PUNC_IDX = collections.namedtuple('_punc_index', ['punc', 'position'])


class Punctuation:
    """Handle punctuations in text.

    Just strip punctuations from text or strip and restore them later.

    Args:
        puncs (str): The punctuations to be processed. Defaults to `_DEF_PUNCS`.

    Example:
        >>> punc = Punctuation()
        >>> punc.strip("This is. example !")
        'This is example'

        >>> text_striped, punc_map = punc.strip_to_restore("This is. example !")
        >>> ' '.join(text_striped)
        'This is example'

        >>> text_restored = punc.restore(text_striped, punc_map)
        >>> text_restored[0]
        'This is. example !'
    """

    def __init__(self, puncs: 'str'=_DEF_PUNCS):
        self.puncs = puncs

    @staticmethod
    def default_puncs():
        """Return default set of punctuations."""
        return _DEF_PUNCS

    @property
    def puncs(self):
        return self._puncs

    @puncs.setter
    def puncs(self, value):
        if not isinstance(value, six.string_types):
            raise ValueError('[!] Punctuations must be of type str.')
        self._puncs = ''.join(list(dict.fromkeys(list(value))))
        self.puncs_regular_exp = re.compile(f'(\\s*[{re.escape(self._puncs)}]+\\s*)+')

    def strip(self, text):
        """Remove all the punctuations by replacing with `space`.

        Args:
            text (str): The text to be processed.

        Example::

            "This is. example !" -> "This is example "
        """
        return re.sub(self.puncs_regular_exp, ' ', text).rstrip().lstrip()

    def strip_to_restore(self, text):
        """Remove punctuations from text to restore them later.

        Args:
            text (str): The text to be processed.

        Examples ::

            "This is. example !" -> [["This is", "example"], [".", "!"]]

        """
        text, puncs = self._strip_to_restore(text)
        return text, puncs

    def _strip_to_restore(self, text):
        """Auxiliary method for Punctuation.preserve()"""
        matches = list(re.finditer(self.puncs_regular_exp, text))
        if not matches:
            return [text], []
        if len(matches) == 1 and matches[0].group() == text:
            return [], [_PUNC_IDX(text, PuncPosition.ALONE)]
        puncs = []
        for match in matches:
            position = PuncPosition.MIDDLE
            if match == matches[0] and text.startswith(match.group()):
                position = PuncPosition.BEGIN
            elif match == matches[-1] and text.endswith(match.group()):
                position = PuncPosition.END
            puncs.append(_PUNC_IDX(match.group(), position))
        splitted_text = []
        for idx, punc in enumerate(puncs):
            split = text.split(punc.punc)
            prefix, suffix = split[0], punc.punc.join(split[1:])
            splitted_text.append(prefix)
            if idx == len(puncs) - 1 and len(suffix) > 0:
                splitted_text.append(suffix)
            text = suffix
        return splitted_text, puncs

    @classmethod
    def restore(cls, text, puncs):
        """Restore punctuation in a text.

        Args:
            text (str): The text to be processed.
            puncs (List[str]): The list of punctuations map to be used for restoring.

        Examples ::

            ['This is', 'example'], ['.', '!'] -> "This is. example!"

        """
        return cls._restore(text, puncs, 0)

    @classmethod
    def _restore(cls, text, puncs, num):
        """Auxiliary method for Punctuation.restore()"""
        if not puncs:
            return text
        if not text:
            return [''.join(m.punc for m in puncs)]
        current = puncs[0]
        if current.position == PuncPosition.BEGIN:
            return cls._restore([current.punc + text[0]] + text[1:], puncs[1:], num)
        if current.position == PuncPosition.END:
            return [text[0] + current.punc] + cls._restore(text[1:], puncs[1:], num + 1)
        if current.position == PuncPosition.ALONE:
            return [current.mark] + cls._restore(text, puncs[1:], num + 1)
        if len(text) == 1:
            return cls._restore([text[0] + current.punc], puncs[1:], num)
        return cls._restore([text[0] + current.punc + text[1]] + text[2:], puncs[1:], num)


class BasePhonemizer(abc.ABC):
    """Base phonemizer class

    Phonemization follows the following steps:
        1. Preprocessing:
            - remove empty lines
            - remove punctuation
            - keep track of punctuation marks

        2. Phonemization:
            - convert text to phonemes

        3. Postprocessing:
            - join phonemes
            - restore punctuation marks

    Args:
        language (str):
            Language used by the phonemizer.

        punctuations (List[str]):
            List of punctuation marks to be preserved.

        keep_puncs (bool):
            Whether to preserve punctuation marks or not.
    """

    def __init__(self, language, punctuations=Punctuation.default_puncs(), keep_puncs=False):
        if not self.is_available():
            raise RuntimeError('{} not installed on your system'.format(self.name()))
        self._language = self._init_language(language)
        self._keep_puncs = keep_puncs
        self._punctuator = Punctuation(punctuations)

    def _init_language(self, language):
        """Language initialization

        This method may be overloaded in child classes (see Segments backend)

        """
        if not self.is_supported_language(language):
            raise RuntimeError(f'language "{language}" is not supported by the {self.name()} backend')
        return language

    @property
    def language(self):
        """The language code configured to be used for phonemization"""
        return self._language

    @staticmethod
    @abc.abstractmethod
    def name():
        """The name of the backend"""
        ...

    @classmethod
    @abc.abstractmethod
    def is_available(cls):
        """Returns True if the backend is installed, False otherwise"""
        ...

    @classmethod
    @abc.abstractmethod
    def version(cls):
        """Return the backend version as a tuple (major, minor, patch)"""
        ...

    @staticmethod
    @abc.abstractmethod
    def supported_languages():
        """Return a dict of language codes -> name supported by the backend"""
        ...

    def is_supported_language(self, language):
        """Returns True if `language` is supported by the backend"""
        return language in self.supported_languages()

    @abc.abstractmethod
    def _phonemize(self, text, separator):
        """The main phonemization method"""

    def _phonemize_preprocess(self, text) ->Tuple[List[str], List]:
        """Preprocess the text before phonemization

        1. remove spaces
        2. remove punctuation

        Override this if you need a different behaviour
        """
        text = text.strip()
        if self._keep_puncs:
            return self._punctuator.strip_to_restore(text)
        return [self._punctuator.strip(text)], []

    def _phonemize_postprocess(self, phonemized, punctuations) ->str:
        """Postprocess the raw phonemized output

        Override this if you need a different behaviour
        """
        if self._keep_puncs:
            return self._punctuator.restore(phonemized, punctuations)[0]
        return phonemized[0]

    def phonemize(self, text: 'str', separator='|', language: 'str'=None) ->str:
        """Returns the `text` phonemized for the given language

        Args:
            text (str):
                Text to be phonemized.

            separator (str):
                string separator used between phonemes. Default to '_'.

        Returns:
            (str): Phonemized text
        """
        text, punctuations = self._phonemize_preprocess(text)
        phonemized = []
        for t in text:
            p = self._phonemize(t, separator)
            phonemized.append(p)
        phonemized = self._phonemize_postprocess(phonemized, punctuations)
        return phonemized

    def print_logs(self, level: 'int'=0):
        indent = '\t' * level
        None
        None


def _espeak_exe(espeak_lib: 'str', args: 'List', sync=False) ->List[str]:
    """Run espeak with the given arguments."""
    cmd = [espeak_lib, '-q', '-b', '1']
    cmd.extend(args)
    logging.debug('espeakng: executing %s', repr(cmd))
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
        res = iter(p.stdout.readline, b'')
        if not sync:
            p.stdout.close()
            if p.stderr:
                p.stderr.close()
            if p.stdin:
                p.stdin.close()
            return res
        res2 = []
        for line in res:
            res2.append(line)
        p.stdout.close()
        if p.stderr:
            p.stderr.close()
        if p.stdin:
            p.stdin.close()
        p.wait()
    return res2


espeak_version_pattern = re.compile('text-to-speech:\\s(?P<version>\\d+\\.\\d+(\\.\\d+)?)')


def get_espeak_version():
    output = subprocess.getoutput('espeak --version')
    match = espeak_version_pattern.search(output)
    return match.group('version')


def get_espeakng_version():
    output = subprocess.getoutput('espeak-ng --version')
    return output.split()[3]


def is_tool(name):
    return which(name) is not None


LANGUAGE_MAP = {'en-us': 'en', 'fr-fr': 'es', 'hi': 'hi'}


class AbbreviationExpander:

    def __init__(self, abbreviations_file):
        self.abbreviations = {}
        self.patterns = {}
        self.load_abbreviations(abbreviations_file)

    def load_abbreviations(self, abbreviations_file):
        with open(abbreviations_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                abbreviation = row['abbreviation']
                expansion = row['expansion']
                language = row['language'].lower()
                self.abbreviations.setdefault(language, {})[abbreviation] = expansion
                if language not in self.patterns:
                    self.patterns[language] = re.compile('\\b(' + '|'.join(re.escape(key) for key in self.abbreviations[language].keys()) + ')\\b', re.IGNORECASE)

    def replace_abbreviations(self, match, language):
        return self.abbreviations[language][match.group(0).lower()]

    def replace_text_abbreviations(self, text, language):
        if language.lower() in self.patterns:
            return self.patterns[language.lower()].sub(lambda match: self.replace_abbreviations(match, language.lower()), text)
        else:
            return text


class NumberNormalizer:

    def __init__(self):
        self._inflect = inflect.engine()
        self._number_re = re.compile('-?[0-9]+')
        self._currency_re = re.compile('([$€£¥₹])([0-9\\,\\.]*[0-9]+)')
        self._currencies = {}

    def add_currency(self, symbol, conversion_rates):
        self._currencies[symbol] = conversion_rates

    def normalize_numbers(self, text, language='en'):
        self._inflect = inflect.engine()
        self._set_language(language)
        text = re.sub(self._currency_re, self._expand_currency, text)
        text = re.sub(self._number_re, lambda match: self._expand_number(match, language), text)
        return text

    def _set_language(self, language):
        if language == 'en':
            self._inflect = inflect.engine()
        else:
            self._inflect = inflect.engine()

    def _expand_currency(self, match):
        unit = match.group(1)
        currency = self._currencies.get(unit)
        if currency:
            value = match.group(2)
            return self._expand_currency_value(value, currency)
        return match.group(0)

    def _expand_currency_value(self, value, inflection):
        parts = value.replace(',', '').split('.')
        if len(parts) > 2:
            return f'{value} {inflection[2]}'
        text = []
        integer = int(parts[0]) if parts[0] else 0
        if integer > 0:
            integer_unit = inflection.get(integer, inflection[2])
            text.append(f'{integer} {integer_unit}')
        fraction = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if fraction > 0:
            fraction_unit = inflection.get(fraction / 100, inflection[0.02])
            text.append(f'{fraction} {fraction_unit}')
        if not text:
            return f'zero {inflection[2]}'
        return ' '.join(text)

    def _expand_number(self, match, language: 'str') ->str:
        num = int(match.group(0))
        if 1000 < num < 3000:
            if num == 2000:
                return self._number_to_words(num, language)
            if 2000 < num < 2010:
                return f'{self._number_to_words(2000, language)} {self._number_to_words(num % 100, language)}'
            if num % 100 == 0:
                return f"{self._number_to_words(num // 100, language)} {self._get_word('hundred')}"
            return self._number_to_words(num, language)
        return self._number_to_words(num, language)

    def _number_to_words(self, n: 'int', language: 'str') ->str:
        try:
            if language == 'en':
                return self._inflect.number_to_words(n)
            else:
                return num2words(n, lang=language)
        except:
            try:
                return num_to_word(n, lang=language)
            except:
                raise NotImplementedError('language not implemented')

    def _get_word(self, word):
        return word


class TimeExpander:

    def __init__(self):
        self._inflect = inflect.engine()
        self._time_re = self._get_time_regex()

    def _get_time_regex(self):
        return re.compile("""\\b
            ((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))  # hours
            :
            ([0-5][0-9])                            # minutes
            \\s*(a\\\\.m\\\\.|am|pm|p\\\\.m\\\\.|a\\\\.m|p\\\\.m)? # am/pm
            \\b""", re.IGNORECASE | re.X)

    def _expand_num(self, n: 'int', language: 'str') ->str:
        try:
            if language == 'en':
                return self._inflect.number_to_words(n)
            else:
                return num2words(n, lang=language)
        except:
            try:
                return num_to_word(n, lang=language)
            except:
                raise NotImplementedError('language not implemented')

    def _expand_time(self, match: "'re.Match'", language: 'str') ->str:
        hour = int(match.group(1))
        past_noon = hour >= 12
        time = []
        if hour > 12:
            hour -= 12
        elif hour == 0:
            hour = 12
            past_noon = True
        time.append(self._expand_num(hour, language))
        minute = int(match.group(6))
        if minute > 0:
            if minute < 10:
                time.append('oh')
            time.append(self._expand_num(minute, language))
        am_pm = match.group(7)
        if am_pm is not None:
            time.extend(list(am_pm.replace('.', '')))
        return ' '.join(time)

    def expand_time(self, text: 'str', language: 'str') ->str:
        return re.sub(self._time_re, lambda match: self._expand_time(match, language), text)


class TextProcessor:

    def __init__(self, lang='en'):
        self.lang = lang
        self._whitespace_re = re.compile('\\s+')
        self.ab_expander = AbbreviationExpander(str(CURRENT_DIR / 'expand/abbreviations.csv'))
        self.time_expander = TimeExpander()
        self.num_normalizer = NumberNormalizer()
        symbol = '$'
        conversion_rates = {(0.01): 'cent', (0.02): 'cents', (1): 'dollar', (2): 'dollars'}
        self.num_normalizer.add_currency(symbol, conversion_rates)

    def lowercase(self, text):
        return text.lower()

    def collapse_whitespace(self, text):
        return re.sub(self._whitespace_re, ' ', text).strip()

    def remove_aux_symbols(self, text):
        text = re.sub('[\\<\\>\\(\\)\\[\\]\\"]+', '', text)
        return text

    def phoneme_cleaners(self, text, language='en'):
        text = self.time_expander.expand_time(text, language=language)
        text = self.num_normalizer.normalize_numbers(text, language=language)
        text = self.ab_expander.replace_text_abbreviations(text, language=language)
        text = self.remove_aux_symbols(text)
        text = self.collapse_whitespace(text)
        return text


_diacrilics = 'ɚ˞ɫ'


_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'


_other_symbols = 'ʍwɥʜʢʡɕʑɺɧʲ'


_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'


_suprasegmentals = "'̃ˈˌːˑ. ,-"


_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'


_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics


def default(val, d):
    return val if exists(val) else d


class Tokenizer:

    def __init__(self, vocab=_phonemes, text_cleaner: 'Optional[Callable]'=None, phonemizer: 'Optional[Callable]'=None, default_lang='en-us', add_blank: 'bool'=False, use_eos_bos=False, pad_id=-1):
        self.text_cleaner = default(text_cleaner, TextProcessor().phoneme_cleaners)
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.pad_id = pad_id
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.phonemizer = phonemizer
        if not exists(self.phonemizer):
            self.phonemizer = ESpeak(language=default_lang)
        self.language = self.phonemizer.language
        self.not_found_characters = []

    @property
    def espeak_language(self):
        return LANGUAGE_MAP.get(self.language, None)

    def encode(self, text: 'str') ->List[int]:
        """Encodes a string of text as a sequence of IDs."""
        token_ids = []
        for char in text:
            try:
                idx = self.char_to_id[char]
                token_ids.append(idx)
            except KeyError:
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    None
                    None
        return token_ids

    def decode(self, token_ids: 'List[int]') ->str:
        """Decodes a sequence of IDs to a string of text."""
        text = ''
        for token_id in token_ids:
            text += self.id_to_char[token_id]
        return text

    def text_to_ids(self, text: 'str', language: 'str'=None) ->Tuple[List[int], str, str]:
        """Converts a string of text to a sequence of token IDs.

        Args:
            text(str):
                The text to convert to token IDs.

            language(str):
                The language code of the text. Defaults to None.

        TODO:
            - Add support for language-specific processing.

        1. Text normalizatin
        2. Phonemization (if use_phonemes is True)
        3. Add blank char between characters
        4. Add BOS and EOS characters
        5. Text to token IDs
        """
        language = default(language, self.espeak_language)
        cleaned_text = None
        if self.text_cleaner is not None:
            text = self.text_cleaner(text, language=language)
            cleaned_text = text
        phonemized = self.phonemizer.phonemize(text, separator='', language=language)
        if self.add_blank:
            phonemized = self.intersperse_blank_char(phonemized, True)
        if self.use_eos_bos:
            phonemized = self.pad_with_bos_eos(phonemized)
        return self.encode(phonemized), cleaned_text, phonemized

    def texts_to_tensor_ids(self, texts: 'List[str]', language: 'str'=None) ->Tensor:
        all_ids = []
        for text in texts:
            ids, *_ = self.text_to_ids(text, language=language)
            all_ids.append(torch.tensor(ids))
        return pad_sequence(all_ids, batch_first=True, padding_value=self.pad_id)

    def ids_to_text(self, id_sequence: 'List[int]') ->str:
        """Converts a sequence of token IDs to a string of text."""
        return self.decode(id_sequence)

    def pad_with_bos_eos(self, char_sequence: 'List[str]'):
        """Pads a sequence with the special BOS and EOS characters."""
        return [self.characters.bos] + list(char_sequence) + [self.characters.eos]

    def intersperse_blank_char(self, char_sequence: 'List[str]', use_blank_char: 'bool'=False):
        """Intersperses the blank character between characters in a sequence.

        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        char_to_use = self.characters.blank if use_blank_char else self.characters.pad
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result


class Attention(nn.Module):

    def __init__(self, dim, *, dim_context=None, causal=False, dim_head=64, heads=8, dropout=0.0, use_flash=False, cross_attn_include_queries=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries
        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)
        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)
        context = default(context, x)
        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)
        q, k, v = self.to_q(x), *self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        out = self.attend(q, k, v, mask=mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GEGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3)
    conv = None
    if causal_conv:
        conv = nn.Sequential(Rearrange('b n d -> b d n'), CausalConv1d(dim_inner, dim_inner, 3), Rearrange('b d n -> b n d'))
    return Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), conv, nn.Linear(dim_inner, dim))


class RMSNorm(nn.Module):

    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma
        if not self.cond:
            return out
        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))
        return out * gamma + beta


class Transformer(nn.Module):

    def __init__(self, dim, *, depth, causal=False, dim_head=64, heads=8, use_flash=False, dropout=0.0, ff_mult=4, final_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([RMSNorm(dim), Attention(dim, causal=causal, dim_head=dim_head, heads=heads, dropout=dropout, use_flash=use_flash), RMSNorm(dim), FeedForward(dim, mult=ff_mult)]))
        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x, mask=None):
        for attn_norm, attn, ff_norm, ff in self.layers:
            x = attn(attn_norm(x), mask=mask) + x
            x = ff(ff_norm(x)) + x
        return self.norm(x)


class Block(nn.Module):

    def __init__(self, dim, dim_out, kernel=3, groups=8, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, dim, dim_out, kernel, *, dropout=0.0, groups=8, num_convs=2):
        super().__init__()
        blocks = []
        for ind in range(num_convs):
            is_first = ind == 0
            dim_in = dim if is_first else dim_out
            block = Block(dim_in, dim_out, kernel, groups=groups, dropout=dropout)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        h = self.blocks(x)
        out = h + self.res_conv(x)
        return rearrange(out, 'b c n -> b n c')


def ConvBlock(dim, dim_out, kernel, dropout=0.0):
    return nn.Sequential(Rearrange('b n c -> b c n'), nn.Conv1d(dim, dim_out, kernel, padding=kernel // 2), nn.SiLU(), nn.Dropout(dropout), Rearrange('b c n -> b n c'))


class DurationPitchPredictorTrunk(nn.Module):

    def __init__(self, dim=512, depth=10, kernel_size=3, dim_context=None, heads=8, dim_head=64, dropout=0.2, use_resnet_block=True, num_convs_per_resnet_block=2, num_convolutions_per_block=3, use_flash_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        conv_klass = ConvBlock if not use_resnet_block else partial(ResnetBlock, num_convs=num_convs_per_resnet_block)
        for _ in range(depth):
            layer = nn.ModuleList([nn.Sequential(*[conv_klass(dim, dim, kernel_size) for _ in range(num_convolutions_per_block)]), RMSNorm(dim), Attention(dim, dim_context=dim_context, heads=heads, dim_head=dim_head, dropout=dropout, use_flash=use_flash_attn, cross_attn_include_queries=True)])
            self.layers.append(layer)
        self.to_pred = nn.Sequential(nn.Linear(dim, 1), Rearrange('... 1 -> ...'), nn.ReLU())

    def forward(self, x, encoded_prompts, prompt_mask=None):
        for conv, norm, attn in self.layers:
            x = conv(x)
            x = attn(norm(x), encoded_prompts, mask=prompt_mask) + x
        return self.to_pred(x)


class PerceiverResampler(nn.Module):

    def __init__(self, *, dim, depth, dim_context=None, num_latents=64, dim_head=64, heads=8, ff_mult=4, use_flash_attn=False):
        super().__init__()
        dim_context = default(dim_context, dim)
        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash_attn, cross_attn_include_queries=True), FeedForward(dim=dim, mult=ff_mult)]))
        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0]
        x = self.proj_context(x)
        latents = repeat(self.latents, 'n d -> b n d', b=batch)
        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class WavenetResBlock(nn.Module):

    def __init__(self, dim, *, dilation, kernel_size=3, skip_conv=False, dim_cond_mult=None):
        super().__init__()
        self.cond = exists(dim_cond_mult)
        self.to_time_cond = None
        if self.cond:
            self.to_time_cond = nn.Linear(dim * dim_cond_mult, dim * 2)
        self.conv = CausalConv1d(dim, dim, kernel_size, dilation=dilation)
        self.res_conv = CausalConv1d(dim, dim, 1)
        self.skip_conv = CausalConv1d(dim, dim, 1) if skip_conv else None

    def forward(self, x, t=None):
        if self.cond:
            assert exists(t)
            t = self.to_time_cond(t)
            t = rearrange(t, 'b c -> b c 1')
            t_gamma, t_beta = t.chunk(2, dim=-2)
        res = self.res_conv(x)
        x = self.conv(x)
        if self.cond:
            x = x * t_gamma + t_beta
        x = x.tanh() * x.sigmoid()
        x = x + res
        skip = None
        if exists(self.skip_conv):
            skip = self.skip_conv(x)
        return x, skip


mlist = nn.ModuleList


class WavenetStack(nn.Module):

    def __init__(self, dim, *, layers, kernel_size=3, has_skip=False, dim_cond_mult=None):
        super().__init__()
        dilations = 2 ** torch.arange(layers)
        self.has_skip = has_skip
        self.blocks = mlist([])
        for dilation in dilations.tolist():
            block = WavenetResBlock(dim=dim, kernel_size=kernel_size, dilation=dilation, skip_conv=has_skip, dim_cond_mult=dim_cond_mult)
            self.blocks.append(block)

    def forward(self, x, t):
        residuals = []
        skips = []
        if isinstance(x, Tensor):
            x = (x,) * len(self.blocks)
        for block_input, block in zip(x, self.blocks):
            residual, skip = block(block_input, t)
            residuals.append(residual)
            skips.append(skip)
        if self.has_skip:
            return torch.stack(skips)
        return residuals


class Wavenet(nn.Module):

    def __init__(self, dim, *, stacks, layers, init_conv_kernel=3, dim_cond_mult=None):
        super().__init__()
        self.init_conv = CausalConv1d(dim, dim, init_conv_kernel)
        self.stacks = mlist([])
        for ind in range(stacks):
            is_last = ind == stacks - 1
            stack = WavenetStack(dim, layers=layers, dim_cond_mult=dim_cond_mult, has_skip=is_last)
            self.stacks.append(stack)
        self.final_conv = CausalConv1d(dim, dim, 1)

    def forward(self, x, t=None):
        x = self.init_conv(x)
        for stack in self.stacks:
            x = stack(x, t)
        return self.final_conv(x.sum(dim=0))


class ConditionableTransformer(nn.Module):

    def __init__(self, dim, *, depth, dim_head=64, heads=8, ff_mult=4, ff_causal_conv=False, dim_cond_mult=None, cross_attn=False, use_flash=False):
        super().__init__()
        self.dim = dim
        self.layers = mlist([])
        cond = exists(dim_cond_mult)
        maybe_adaptive_norm_kwargs = dict(scale=not cond, dim_cond=dim * dim_cond_mult) if cond else dict()
        rmsnorm = partial(RMSNorm, **maybe_adaptive_norm_kwargs)
        for _ in range(depth):
            self.layers.append(mlist([rmsnorm(dim), Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash), rmsnorm(dim) if cross_attn else None, Attention(dim=dim, dim_head=dim_head, heads=heads, use_flash=use_flash) if cross_attn else None, rmsnorm(dim), FeedForward(dim=dim, mult=ff_mult, causal_conv=ff_causal_conv)]))
        self.to_pred = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim, bias=False))

    def forward(self, x, times=None, context=None):
        t = times
        for attn_norm, attn, cross_attn_norm, cross_attn, ff_norm, ff in self.layers:
            res = x
            x = attn_norm(x, cond=t)
            x = attn(x) + res
            if exists(cross_attn):
                assert exists(context)
                res = x
                x = cross_attn_norm(x, cond=t)
                x = cross_attn(x, context=context) + res
            res = x
            x = ff_norm(x, cond=t)
            x = ff(x) + res
        return self.to_pred(x)


def pad_or_curtail_to_length(t, length):
    if t.shape[-1] == length:
        return t
    if t.shape[-1] > length:
        return t[..., :length]
    return F.pad(t, (0, length - t.shape[-1]))


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def average_over_durations(values, durs):
    """
        - in:
            - values: B, 1, T_de
            - durs: B, T_en
        - out:
            - avg: B, 1, T_en
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))
    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = repeat(durs_cums_starts, 'bs l -> bs n l', n=n_formants)
    dce = repeat(durs_cums_ends, 'bs l -> bs n l', n=n_formants)
    values_sums = torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
    values_nelems = torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)
    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems)
    return avg


def compute_pitch_pytorch(wav, sample_rate):
    pitch_feature = torchaudio.functional.compute_kaldi_pitch(wav, sample_rate)
    pitch, nfcc = pitch_feature.unbind(dim=-1)
    return pitch


def compute_pitch_pyworld(wav, sample_rate, hop_length, pitch_fmax=640.0):
    is_tensor_input = torch.is_tensor(wav)
    if is_tensor_input:
        device = wav.device
        wav = wav.contiguous().cpu().numpy()
    if divisible_by(len(wav), hop_length):
        wav = np.pad(wav, (0, hop_length // 2), mode='reflect')
    wav = wav.astype(np.double)
    outs = []
    for sample in wav:
        f0, t = pw.dio(sample, fs=sample_rate, f0_ceil=pitch_fmax, frame_period=1000 * hop_length / sample_rate)
        f0 = pw.stonemask(sample, f0, t, sample_rate)
        outs.append(f0)
    outs = np.stack(outs)
    if is_tensor_input:
        outs = torch.from_numpy(outs)
    return outs


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-09):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)


def create_mask(sequence_length, max_len):
    dtype, device = sequence_length.dtype, sequence_length.device
    seq_range = torch.arange(max_len, dtype=dtype, device=device)
    sequence_length = rearrange(sequence_length, 'b -> b 1')
    seq_range = rearrange(seq_range, 't -> 1 t')
    return seq_range < sequence_length


def f0_to_coarse(f0, f0_bin=256, f0_max=1100.0, f0_min=50.0):
    f0_mel_max = 1127 * torch.log(1 + torch.tensor(f0_max) / 700)
    f0_mel_min = 1127 * torch.log(1 + torch.tensor(f0_min) / 700)
    f0_mel = 1127 * (1 + f0 / 700).log()
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).int()
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def gamma_to_alpha_sigma(gamma, scale=1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gamma_to_log_snr(gamma, scale=1, eps=1e-05):
    return log(gamma * scale ** 2 / (1 - gamma), eps=eps)


def generate_mask_from_repeats(repeats):
    repeats = repeats.int()
    device = repeats.device
    lengths = repeats.sum(dim=-1)
    max_length = lengths.amax().item()
    cumsum = repeats.cumsum(dim=-1)
    cumsum_exclusive = F.pad(cumsum, (1, -1), value=0.0)
    seq = torch.arange(max_length, device=device)
    seq = repeat(seq, '... j -> ... i j', i=repeats.shape[-1])
    cumsum = rearrange(cumsum, '... i -> ... i 1')
    cumsum_exclusive = rearrange(cumsum_exclusive, '... i -> ... i 1')
    lengths = rearrange(lengths, 'b -> b 1 1')
    mask = (seq < cumsum) & (seq >= cumsum_exclusive) & (seq < lengths)
    return mask


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def safe_div(numer, denom):
    return numer / denom.clamp(min=1e-10)


def sigmoid_schedule(t, start=-3, end=3, tau=1, clamp_min=1e-09):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min, max=1.0)


def simple_linear_schedule(t, clip_min=1e-09):
    return (1 - t).clamp(min=clip_min)


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (Attend,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (CausalConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4])], {})),
    (GEGLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (RMSNorm,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {})),
    (Wavenet,
     lambda: ([], {'dim': 4, 'stacks': 4, 'layers': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (WavenetResBlock,
     lambda: ([], {'dim': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4])], {})),
    (WavenetStack,
     lambda: ([], {'dim': 4, 'layers': 1}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {})),
]

