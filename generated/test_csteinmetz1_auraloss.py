
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


import numpy as np


from typing import List


from typing import Any


from torch import Tensor as T


import scipy.signal


import math


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')


class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 0.0
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance='L1', reduction='mean'):
        super(STFTMagnitudeLoss, self).__init__()
        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac
        if distance == 'L1':
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == 'L2':
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)


def compare_filters(iir_b, iir_a, fir_b, fs=1):
    w_iir, h_iir = scipy.signal.freqz(iir_b, iir_a, fs=fs, worN=2048)
    w_fir, h_fir = scipy.signal.freqz(fir_b, fs=fs)
    h_iir_db = 20 * np.log10(np.abs(h_iir) + 1e-08)
    h_fir_db = 20 * np.log10(np.abs(h_fir) + 1e-08)
    plt.plot(w_iir, h_iir_db, label='IIR filter')
    plt.plot(w_fir, h_fir_db, label='FIR approx. filter')
    plt.xscale('log')
    plt.ylim([-50, 10])
    plt.xlim([10, 22050.0])
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('Mag. (dB)')
    plt.legend()
    plt.grid()
    plt.show()


class FIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module.

    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap (only applicable for "hp" and "fd"). Default: 0.85
        ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
        plot (bool): Plot the magnitude respond of the filter. Default: False

    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    A-weighting filter - "aw"
    First-order highpass - "hp"
    Folded differentiator - "fd"

    Note that the default coefficeint value of 0.85 is optimized for
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type='hp', coef=0.85, fs=44100, ntaps=101, plot=False):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot
        import scipy.signal
        if ntaps % 2 == 0:
            raise ValueError(f'ntaps must be odd (ntaps={ntaps}).')
        if filter_type == 'hp':
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == 'fd':
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == 'aw':
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997
            NUMs = [(2 * np.pi * f4) ** 2 * 10 ** (A1000 / 20), 0, 0, 0, 0]
            DENs = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
            DENs = np.polymul(np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype('float32')).view(1, 1, -1)
            if plot:
                compare_filters(b, a, taps, fs=fs)

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        input = torch.nn.functional.conv1d(input, self.fir.weight.data, padding=self.ntaps // 2)
        target = torch.nn.functional.conv1d(target, self.fir.weight.data, padding=self.ntaps // 2)
        return input, target


def apply_reduction(losses, reduction='none'):
    """Apply reduction to collection of losses."""
    if reduction == 'mean':
        losses = losses.mean()
    elif reduction == 'sum':
        losses = losses.sum()
    return losses


def get_window(win_type: 'str', win_length: 'int'):
    """Return a window function.

    Args:
        win_type (str): Window type. Can either be one of the window function provided in PyTorch
            ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            or any of the windows provided by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.get_window.html).
        win_length (int): Window length

    Returns:
        win: The window as a 1D torch tensor
    """
    try:
        win = getattr(torch, win_type)(win_length)
    except:
        win = torch.from_numpy(scipy.signal.windows.get_window(win_type, win_length))
    return win


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module.

    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)

    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """

    def __init__(self, fft_sizes: 'List[int]'=[1024, 2048, 512], hop_sizes: 'List[int]'=[120, 240, 50], win_lengths: 'List[int]'=[600, 1200, 240], window: 'str'='hann_window', w_sc: 'float'=1.0, w_log_mag: 'float'=1.0, w_lin_mag: 'float'=0.0, w_phs: 'float'=0.0, sample_rate: 'float'=None, scale: 'str'=None, n_bins: 'int'=None, perceptual_weighting: 'bool'=False, scale_invariance: 'bool'=False, **kwargs):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, w_sc, w_log_mag, w_lin_mag, w_phs, sample_rate, scale, n_bins, perceptual_weighting, scale_invariance, **kwargs)]

    def forward(self, x, y):
        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []
        for f in self.stft_losses:
            if f.output == 'full':
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1])
                log_mag_loss.append(tmp_loss[2])
                lin_mag_loss.append(tmp_loss[3])
                phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)
        mrstft_loss /= len(self.stft_losses)
        if f.output == 'loss':
            return mrstft_loss
        else:
            return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class RandomResolutionSTFTLoss(torch.nn.Module):
    """Random resolution STFT loss module.

    See [Steinmetz & Reiss, 2020](https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf)

    Args:
        resolutions (int): Total number of STFT resolutions.
        min_fft_size (int): Smallest FFT size.
        max_fft_size (int): Largest FFT size.
        min_hop_size (int): Smallest hop size as porportion of window size.
        min_hop_size (int): Largest hop size as porportion of window size.
        window (str): Window function type.
        randomize_rate (int): Number of forwards before STFTs are randomized.
    """

    def __init__(self, resolutions=3, min_fft_size=16, max_fft_size=32768, min_hop_size=0.1, max_hop_size=1.0, windows=['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window'], w_sc=1.0, w_log_mag=1.0, w_lin_mag=0.0, w_phs=0.0, sample_rate=None, scale=None, n_mels=None, randomize_rate=1, **kwargs):
        super().__init__()
        self.resolutions = resolutions
        self.min_fft_size = min_fft_size
        self.max_fft_size = max_fft_size
        self.min_hop_size = min_hop_size
        self.max_hop_size = max_hop_size
        self.windows = windows
        self.randomize_rate = randomize_rate
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_mels = n_mels
        self.nforwards = 0
        self.randomize_losses()

    def randomize_losses(self):
        self.stft_losses = torch.nn.ModuleList()
        for n in range(self.resolutions):
            frame_size = 2 ** np.random.randint(np.log2(self.min_fft_size), np.log2(self.max_fft_size))
            hop_size = int(frame_size * (self.min_hop_size + np.random.rand() * (self.max_hop_size - self.min_hop_size)))
            window_length = int(frame_size * np.random.choice([1.0, 0.5, 0.25]))
            window = np.random.choice(self.windows)
            self.stft_losses += [STFTLoss(frame_size, hop_size, window_length, window, self.w_sc, self.w_log_mag, self.w_lin_mag, self.w_phs, self.sample_rate, self.scale, self.n_mels)]

    def forward(self, input, target):
        if input.size(-1) <= self.max_fft_size:
            raise ValueError(f'Input length ({input.size(-1)}) must be larger than largest FFT size ({self.max_fft_size}).')
        elif target.size(-1) <= self.max_fft_size:
            raise ValueError(f'Target length ({target.size(-1)}) must be larger than largest FFT size ({self.max_fft_size}).')
        if self.nforwards % self.randomize_rate == 0:
            self.randomize_losses()
        loss = 0.0
        for f in self.stft_losses:
            loss += f(input, target)
        loss /= len(self.stft_losses)
        self.nforwards += 1
        return loss


class SumAndDifference(torch.nn.Module):
    """Sum and difference signal extraction module."""

    def __init__(self):
        """Initialize sum and difference extraction module."""
        super(SumAndDifference, self).__init__()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, #channels, #samples).
        Returns:
            Tensor: Sum signal.
            Tensor: Difference signal.
        """
        if not x.size(1) == 2:
            raise ValueError(f'Input must be stereo: {x.size(1)} channel(s).')
        sum_sig = self.sum(x).unsqueeze(1)
        diff_sig = self.diff(x).unsqueeze(1)
        return sum_sig, diff_sig

    @staticmethod
    def sum(x):
        return x[:, 0, :] + x[:, 1, :]

    @staticmethod
    def diff(x):
        return x[:, 0, :] - x[:, 1, :]


class SumAndDifferenceSTFTLoss(torch.nn.Module):
    """Sum and difference sttereo STFT loss module.

    See [Steinmetz et al., 2020](https://arxiv.org/abs/2010.10291)

    Args:
        fft_sizes (List[int]): List of FFT sizes.
        hop_sizes (List[int]): List of hop sizes.
        win_lengths (List[int]): List of window lengths.
        window (str, optional): Window function type.
        w_sum (float, optional): Weight of the sum loss component. Default: 1.0
        w_diff (float, optional): Weight of the difference loss component. Default: 1.0
        perceptual_weighting (bool, optional): Apply perceptual A-weighting (Sample rate must be supplied). Default: False
        mel_stft (bool, optional): Use Multi-resoltuion mel spectrograms. Default: False
        n_mel_bins (int, optional): Number of mel bins to use when mel_stft = True. Default: 128
        sample_rate (float, optional): Audio sample rate. Default: None
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
    """

    def __init__(self, fft_sizes: 'List[int]', hop_sizes: 'List[int]', win_lengths: 'List[int]', window: 'str'='hann_window', w_sum: 'float'=1.0, w_diff: 'float'=1.0, output: 'str'='loss', **kwargs):
        super().__init__()
        self.sd = SumAndDifference()
        self.w_sum = w_sum
        self.w_diff = w_diff
        self.output = output
        self.mrstft = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, window, **kwargs)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'):
        """This loss function assumes batched input of stereo audio in the time domain.

        Args:
            input (torch.Tensor): Input tensor with shape (batch size, 2, seq_len).
            target (torch.Tensor): Target tensor with shape (batch size, 2, seq_len).

        Returns:
            loss (torch.Tensor): Aggreate loss term. Only returned if output='loss'.
            loss (torch.Tensor), sum_loss (torch.Tensor), diff_loss (torch.Tensor):
                Aggregate and intermediate loss terms. Only returned if output='full'.
        """
        assert input.shape == target.shape
        bs, chs, seq_len = input.size()
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)
        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = (self.w_sum * sum_loss + self.w_diff * diff_loss) / 2
        if self.output == 'loss':
            return loss
        elif self.output == 'full':
            return loss, sum_loss, diff_loss


class ESRLoss(torch.nn.Module):
    """Error-to-signal ratio loss function module.

    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, eps: 'float'=1e-08, reduction: 'str'='mean') ->None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: 'T', target: 'T') ->T:
        num = ((target - input) ** 2).sum(dim=-1)
        denom = (target ** 2).sum(dim=-1) + self.eps
        losses = num / denom
        losses = apply_reduction(losses, reduction=self.reduction)
        return losses


class DCLoss(torch.nn.Module):
    """DC loss function module.

    See [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of elements in the output,
        'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, eps: 'float'=1e-08, reduction: 'str'='mean') ->None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: 'T', target: 'T') ->T:
        num = (target - input).mean(dim=-1) ** 2
        denom = (target ** 2).mean(dim=-1) + self.eps
        losses = num / denom
        losses = apply_reduction(losses, self.reduction)
        return losses


class LogCoshLoss(torch.nn.Module):
    """Log-cosh loss function module.

    See [Chen et al., 2019](https://openreview.net/forum?id=rkglvsC9Ym).

    Args:
        a (float, optional): Smoothness hyperparameter. Smaller is smoother. Default: 1.0
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, a=1.0, eps=1e-08, reduction='mean'):
        super(LogCoshLoss, self).__init__()
        self.a = a
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        losses = (1 / self.a * torch.log(torch.cosh(self.a * (input - target)) + self.eps)).mean(-1)
        losses = apply_reduction(losses, self.reduction)
        return losses


class SNRLoss(torch.nn.Module):
    """Signal-to-noise ratio loss module.

    Note that this does NOT implement the SDR from
    [Vincent et al., 2006](https://ieeexplore.ieee.org/document/1643671),
    which includes the application of a 512-tap FIR filter.
    """

    def __init__(self, zero_mean=True, eps=1e-08, reduction='mean'):
        super(SNRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean
        res = input - target
        losses = 10 * torch.log10((target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps)
        losses = apply_reduction(losses, self.reduction)
        return -losses


class SISDRLoss(torch.nn.Module):
    """Scale-invariant signal-to-distortion ratio loss module.

    Note that this returns the negative of the SI-SDR loss.

    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=True, eps=1e-08, reduction='mean'):
        super(SISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean
        alpha = (input * target).sum(-1) / ((target ** 2).sum(-1) + self.eps)
        target = target * alpha.unsqueeze(-1)
        res = input - target
        losses = 10 * torch.log10((target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps)
        losses = apply_reduction(losses, self.reduction)
        return -losses


class SDSDRLoss(torch.nn.Module):
    """Scale-dependent signal-to-distortion ratio loss module.

    Note that this returns the negative of the SD-SDR loss.

    See [Le Roux et al., 2018](https://arxiv.org/abs/1811.02508)

    Args:
        zero_mean (bool, optional) Remove any DC offset in the inputs. Default: ``True``
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'
    Shape:
        - input : :math:`(batch, nchs, ...)`.
        - target: :math:`(batch, nchs, ...)`.
    """

    def __init__(self, zero_mean=True, eps=1e-08, reduction='mean'):
        super(SDSDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if self.zero_mean:
            input_mean = torch.mean(input, dim=-1, keepdim=True)
            target_mean = torch.mean(target, dim=-1, keepdim=True)
            input = input - input_mean
            target = target - target_mean
        alpha = (input * target).sum(-1) / ((target ** 2).sum(-1) + self.eps)
        scaled_target = target * alpha.unsqueeze(-1)
        res = input - target
        losses = 10 * torch.log10((scaled_target ** 2).sum(-1) / ((res ** 2).sum(-1) + self.eps) + self.eps)
        losses = apply_reduction(losses, self.reduction)
        return -losses


class FiLM(torch.nn.Module):

    def __init__(self, num_features, cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)
        x = self.bn(x)
        x = x * g + b
        return x


def causal_crop(x, length: 'int'):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[..., start:stop]


def center_crop(x, length: 'int'):
    start = (x.shape[-1] - length) // 2
    stop = start + length
    return x[..., start:stop]


class TCNBlock(torch.nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, padding='same', dilation=1, grouped=False, causal=False, **kwargs):
        super(TCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal
        groups = out_ch if grouped and in_ch % out_ch == 0 else 1
        if padding == 'same':
            pad_value = kernel_size - 1 + (kernel_size - 1) * (dilation - 1)
        elif padding in ['none', 'valid']:
            pad_value = 0
        self.conv1 = torch.nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0, dilation=dilation, groups=groups, bias=False)
        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)
        else:
            self.bn = torch.nn.BatchNorm1d(out_ch)
        self.relu = torch.nn.PReLU(out_ch)
        self.res = torch.nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False)

    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.relu(x)
        x_res = self.res(x_in)
        if self.causal:
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])
        return x


class TCNModel(torch.nn.Module):
    """Temporal convolutional network.
    Args:
        nparams (int): Number of conditioning parameters.
        ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
        noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
        nblocks (int): Number of total TCN blocks. Default: 10
        kernel_size (int): Width of the convolutional kernels. Default: 3
        dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
        channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
        channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
        stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
        grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
        causal (bool): Causal TCN configuration does not consider future input values. Default: False
    """

    def __init__(self, ninputs=1, noutputs=1, nblocks=10, kernel_size=3, dilation_growth=1, channel_growth=1, channel_width=32, stack_size=10, grouped=False, causal=False):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            if channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth
            else:
                out_ch = channel_width
            dilation = dilation_growth ** (n % stack_size)
            self.blocks.append(TCNBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding='same' if causal else 'valid', causal=causal, grouped=grouped))
        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return torch.tanh(self.output(x))

    def compute_receptive_field(self):
        """Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1, self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + (self.hparams.kernel_size - 1) * dilation
        return


import torch
from torch.nn import MSELoss, ReLU
from types import SimpleNamespace


TESTCASES = [
    # (nn.Module, init_args, forward_args)
    (DCLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (ESRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (FIRFilter,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64]), torch.rand([4, 1, 64])], {})),
    (FiLM,
     lambda: ([], {'num_features': 4, 'cond_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})),
    (LogCoshLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SDSDRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SISDRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SNRLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (STFTMagnitudeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (SpectralConvergenceLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})),
    (TCNBlock,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4])], {})),
    (TCNModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64])], {})),
]

