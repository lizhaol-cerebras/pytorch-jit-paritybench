
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


import collections


import numpy


import functools


import copy


import math


import numpy as np


import itertools


import inspect


import warnings


class ELU(Module):
    """Apply exponential linear unit.
    `[Clevert et.al, 2015] <https://arxiv.org/abs/1511.07289>`_.

    The **ELU** function is defined as:

    .. math::
        \\text{ELU}(x) =
            \\begin{cases}
                x, & \\text{ if } x \\geq 0 \\\\
                \\alpha * (\\exp(x) - 1), & \\text{ otherwise }
            \\end{cases}

    Examples:

    ```python
    m = torch.nn.ELU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.elu(...)`_

    """

    def __init__(self, alpha=1.0, inplace=False):
        """Create a ``ELU`` module.

        Parameters
        ----------
        alpha : float, optional, default=1.
            The value to :math:`\\alpha`.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'alpha={}{}'.format(self.alpha, inplace_str)

    def forward(self, input):
        return functional.elu(input, self.alpha, self.inplace)


class GELU(Module):
    """Apply gaussian error linear unit.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **GELU** function is defined as:

    .. math:: \\text{GELU}(x) = x\\cdot\\frac{1}{2}[1 + \\text{erf}(x / \\sqrt{2})]

    Examples:

    ```python
    m = torch.nn.GELU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.gelu(...)`_

    """

    def __init__(self, approximate='none'):
        """Create a ``GELU`` module.

        Parameters
        ----------
        approximate : str, optional, default='none'
            The approximate algorithm.

        """
        super(GELU, self).__init__()
        self.approximate = approximate

    def extra_repr(self):
        return 'approximate={}'.format(self.approximate)

    def forward(self, input):
        return functional.gelu(input, approximate=self.approximate)


class GumbelSoftmax(Module):
    """Apply gumbel softmax function.
    `[Jang et.al, 2016] <https://arxiv.org/abs/1611.01144>`_.

    The **GumbelSoftmax** function is defined as:

    .. math::
        \\text{GumbelSoftmax}(x) =
            \\frac{exp((\\log(\\pi_{i}) + g_{i}) / \\tau)}
            {\\sum exp((\\log(\\pi_{j}) + g_{i}) / \\tau)} \\\\
        \\quad \\\\ \\text{where}\\quad g_{i} \\sim \\text{Gumbel}(0, 1)

    Examples:

    ```python
    m = torch.nn.GumbelSoftmax(tau=0.5, dim=1)
    x = torch.randn(2, 3)
    y = m(x)
    ```

    """

    def __init__(self, tau=1, dim=None, inplace=False):
        """Create a ``GumbelSoftmax`` module.

        Parameters
        ----------
        tau : Union[number, dragon.vm.torch.Tensor], default=1
            The temperature to use.
        dim : int, required
            The dimension to reduce.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(GumbelSoftmax, self).__init__()
        self.tau = tau
        self.dim = dim
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'dim={}{}'.format(self.dim, inplace_str)

    def forward(self, input):
        u_dist = random_ops.rand(input.shape, dtype=input.dtype, device=input.device)
        gumbel = -(-u_dist.log()).log()
        gumbel = (input + gumbel) / self.tau
        return functional.softmax(gumbel, self.dim, self.inplace)


class Hardsigmoid(Module):
    """Apply hard sigmoid function.

    The **HardSigmoid** function is defined as:

    .. math::
        \\text{Hardsigmoid}(x) = \\begin{cases}
            0 & \\text{if~} x \\le -3, \\\\
            1 & \\text{if~} x \\ge +3, \\\\
            x / 6 + 1 / 2 & \\text{otherwise}
        \\end{cases}

    Examples:

    ```python
    m = torch.nn.Hardsigmoid()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.hardsigmoid(...)`_

    """

    def __init__(self, inplace=False):
        """Create a ``Hardsigmoid`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Hardsigmoid, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return functional.hardsigmoid(input, self.inplace)


class Hardswish(Module):
    """Apply hard swish function.
    `[Howard et.al, 2019] <https://arxiv.org/abs/1905.02244>`_.

    The **HardSwish** function is defined as:

    .. math::
        \\text{Hardsigmoid}(x) = \\begin{cases}
            0 & \\text{if~} x \\le -3, \\\\
            x & \\text{if~} x \\ge +3, \\\\
            x \\cdot (x + 3) /6 & \\text{otherwise}
        \\end{cases}

    Examples:

    ```python
    m = torch.nn.Hardswish()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.hardswish(...)`_

    """

    def __init__(self):
        """Create a ``Hardswish`` module."""
        super(Hardswish, self).__init__()

    def forward(self, input):
        return functional.hardswish(input)


class LeakyReLU(Module):
    """Apply leaky rectified linear unit.

    The **LeakyReLU** function is defined as:

    .. math::
        \\text{LeakyReLU}(x) =
            \\begin{cases}
                x, & \\text{ if } x \\geq 0 \\\\
                slope * x, & \\text{ otherwise }
            \\end{cases}

    Examples:

    ```python
    m = torch.nn.LeakyReLU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.leaky_relu(...)`_

    """

    def __init__(self, negative_slope=0.01, inplace=False):
        """Create a ``LeakyReLU`` module.

        Parameters
        ----------
        negative_slope : float, optional, default=0.01
            The slope of negative side.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)

    def forward(self, input):
        return functional.leaky_relu(input, self.negative_slope, self.inplace)


class LogSoftmax(Module):
    """Apply logarithm softmax function.

    The **LogSoftmax** function is defined as:

    .. math:: \\text{LogSoftmax}(x) = \\log(\\frac{\\exp(x_{i})}{\\sum \\exp(x_{j})})

    Examples:

    ```python
    m = torch.nn.LogSoftmax(dim=1)
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.log_softmax(...)`_

    """

    def __init__(self, dim, inplace=False):
        """Create a ``LogSoftmax`` module.

        Parameters
        ----------
        dim : int
            The dimension to reduce.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(LogSoftmax, self).__init__()
        self.dim = dim
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'dim={}{}'.format(self.dim, inplace_str)

    def forward(self, input):
        return functional.log_softmax(input, self.dim, self.inplace)


class Parameter(Tensor):
    """A wrapped tensor considered to be a module parameter.

    Use this class to wrap a leaf tensor to be a parameter,
    that can be identified by ``torch.nn.Module``:

    ```python
    param = torch.nn.Parameter(torch.ones(2, 3))
    ```

    Typically, the gradient of a parameter should be computed,
    while you can set ``requires_grad`` to ``False`` to ignore.
    Froze a parameter from updating can be directly implemented
    by ignoring it's gradient:

    ```python
    param = torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
    ```

    """

    def __init__(self, tensor, requires_grad=True):
        """Create a ``Parameter``.

        Parameters
        ----------
        tensor : dragon.vm.torch.Tensor
            The tensor to be wrapped.
        requires_grad : bool, optional, default=True
            Whether to compute gradient if necessary.

        """
        super(Parameter, self).__init__(device=tensor.device, impl=tensor._impl, requires_grad=requires_grad)
        self._is_leaf = True
        self._wrapped_tensor = tensor

    def __repr__(self):
        """Return the representation string.

        Returns
        -------
        str
            The representation string.

        """
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()

    def __setstate__(self, state):
        """Set the serialization state.

        Parameters
        ----------
        state : Dict
            The state dict.

        """
        self._is_leaf = True
        self._wrapped_tensor = Tensor()
        self._wrapped_tensor.__setstate__(state)
        state.pop('array', None)
        super(Parameter, self).__setstate__(state)
        self._impl = self._wrapped_tensor._impl
        self._deleter = None


class Linear(Module):
    """Apply linear transformation.

    .. math:: \\text{out} = \\text{input} \\times \\text{weight}^{T} + \\text{bias}

    Examples:

    ```python
    m = torch.nn.Linear(2, 3)
    x = torch.ones(2, 2)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.linear(...)`_

    """

    def __init__(self, in_features, out_features, bias=True):
        """Create a ``Linear`` module.

        Parameters
        ----------
        in_features : int
            The number of input features.
        out_features : int
            The number of output features.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.

        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(Tensor(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)

    def forward(self, input):
        return functional.linear(input, self.weight, self.bias)

    def reset_parameters(self):
        stddev = 1.0 / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-stddev, stddev)
        if self.bias is not None:
            self.bias.data.uniform_(-stddev, stddev)


def _calculate_fan_in_and_fan_out(tensor):
    """Return the fan value according to tensor size."""
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError('Excepted 2 or higher tensor dimensions.')
    if dimensions == 2:
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input = tensor.size(1)
        num_output = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = math.prod(tensor.shape[2:])
        fan_in = num_input * receptive_field_size
        fan_out = num_output * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(tensor, gain=1):
    """Fill tensor from a xavier uniform distribution.

    .. math::
        \\text{tensor} \\sim \\mathcal{U}(-\\alpha, \\alpha) \\\\ \\, \\\\ \\,
            \\text{where} \\quad \\alpha = \\text{gain} \\times
                \\sqrt{\\frac{6}{\\text{fan\\_in} + \\text{fan\\_out}}}

    Parameters
    ----------
    tensor : dragon.vm.torch.Tensor
        The input tensor.
    gain : number, optional, default=1
        The gain value.

    Returns
    -------
    dragon.vm.torch.Tensor
        The input tensor.

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    a = 3.0 ** 0.5 * std
    with grad_mode.no_grad():
        return tensor.uniform_(-a, a)


class MultiheadAttention(Module):
    """Apply multihead attention.
    `[Vaswani et.al, 2017] <https://arxiv.org/abs/1706.03762>`_.

    See Also
    --------
    `torch.nn.functional.multi_head_attention_forward(...)`_

    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None):
        """Create a ``MultiheadAttention`` module.

        Parameters
        ----------
        embed_dim : int
            The dimension of input embeddings.
        num_heads : int
            The number of parallel heads.
        dropout: float, optional, default=0.
            The probability to set the attention to zero.
        bias : bool, optional, default=True
            Add a bias tensor to output or not.
        kdim : int, optional
            The dimension of key embedding.
        vdim : int, optional
            The dimension of value embedding.

        """
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError('<embed_dim> must be divisible by <num_heads>.')
        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(Tensor(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        if bias:
            self.in_proj_bias = Parameter(Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if self.in_proj_bias is not None:
            self.in_proj_bias.zero_()
            self.out_proj.bias.zero_()

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        return functional.multi_head_attention_forward(query, key, value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads, in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias, out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias, dropout_p=self.dropout, training=self.training, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, use_separate_proj_weight=not self._qkv_same_embed_dim, q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight, v_proj_weight=self.v_proj_weight)


class PReLU(Module):
    """Apply parametric rectified linear unit.
    `[He et.al, 2015] <https://arxiv.org/abs/1502.01852>`_.

    The **PReLU** function is defined as:

    .. math::
        \\text{PReLU}(x) =
            \\begin{cases}
                x, & \\text{ if } x \\geq 0 \\\\
                weight * x, & \\text{ otherwise }
            \\end{cases}

    Examples:

    ```python
    # Use a single parameter to scale all channels
    # Typically known as the ``channel-shared`` style
    m = torch.nn.PReLU(num_parameters=1)
    x = torch.randn(2, 3)
    y = m(x)

    # Use different parameter for each channel
    mm =  torch.nn.PReLU(num_parameters=3)
    z = mm(x)
    ```

    See Also
    --------
    `torch.nn.functional.prelu(...)`_

    """

    def __init__(self, num_parameters=1, init=0.25):
        """Create a ``PReLU`` module.

        Parameters
        ----------
        num_parameters : int, optional, default=1
            The number of parameters.
        init : float, optional, default=0.25
            The default value of parameters.

        """
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(Tensor(num_parameters).fill_(init))

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)

    def forward(self, input):
        return functional.prelu(input, self.weight)


class ReLU(Module):
    """Apply rectified linear unit.
    `[Nair & Hinton, 2010] <http://www.csri.utoronto.ca/~hinton/absps/reluICML.pdf>`_.

    The **ReLU** function is defined as:

    .. math::
        \\text{ReLU}(x) =
            \\begin{cases}
                x, & \\text{ if } x \\geq 0 \\\\
                0, & \\text{ otherwise }
            \\end{cases}

    Examples:

    ```python
    m = torch.nn.ReLU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.relu(...)`_

    """

    def __init__(self, inplace=False):
        """Create a ``ReLU`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(ReLU, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return functional.relu(input, inplace=self.inplace)


class ReLU6(Module):
    """Apply clipped-6 rectified linear unit.
    `[Krizhevsky, 2010] <http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf>`_.

    The **ReLU-6** function is defined as:

    .. math::
        \\text{ReLU-6}(x) =
            \\begin{cases}
                \\min(x, 6), & \\text{ if } x \\geq 0 \\\\
                0, & \\text{ otherwise }
            \\end{cases}

    Examples:

    ```python
    m = torch.nn.ReLU6()
    x = torch.tensor([-2, 0, 2, 4, 6, 8], 'float32')
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.relu6(...)`_

    """

    def __init__(self, inplace=False):
        """Create a ``ReLU6`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(ReLU6, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return functional.relu6(input, inplace=self.inplace)


class SELU(Module):
    """Apply scaled exponential linear unit.
    `[Klambauer et.al, 2017] <https://arxiv.org/abs/1706.02515>`_.

    The **SELU** function is defined as:

    .. math::
        \\text{SELU}(x) = 1.0507 *
            \\begin{cases}
                x, & \\text{ if } x \\geq 0 \\\\
                1.67326 * (\\exp(x) - 1), & \\text{ otherwise }
            \\end{cases}

    Examples:

    ```python
    m = torch.nn.SELU()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.selu(...)`_

    """

    def __init__(self, inplace=False):
        """Create a ``SELU`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(SELU, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return functional.selu(input, self.inplace)


class Sigmoid(Module):
    """Apply sigmoid function.

    The **Sigmoid** function is defined as:

    .. math:: \\text{Sigmoid}(x) = \\frac{1}{1 + \\exp(-x)}

    Examples:

    ```python
    m = torch.nn.Sigmoid()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.sigmoid(...)`_

    """

    def __init__(self, inplace=False):
        """Create a ``Sigmoid`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return functional.sigmoid(input, self.inplace)


class SiLU(Module):
    """Apply sigmoid linear unit.
    `[Hendrycks & Gimpel, 2016] <https://arxiv.org/abs/1606.08415>`_.

    The **SiLU** function is defined as:

    .. math:: \\text{SiLU}(x) = x \\cdot \\frac{1}{1 + \\exp(-x)}

    Examples:

    ```python
    m = torch.nn.So()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.silu(...)`_

    """

    def __init__(self):
        """Create a ``SiLU`` module."""
        super(SiLU, self).__init__()

    def forward(self, input):
        return functional.silu(input)


class Softmax(Module):
    """Apply softmax function.

    The **Softmax** function is defined as:

    .. math:: \\text{Softmax}(x_{i}) = \\frac{\\exp(x_{i})}{\\sum_{j} \\exp(x_{j})}

    Examples:

    ```python
    m = torch.nn.Softmax(dim=1)
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.softmax(...)`_

    """

    def __init__(self, dim, inplace=False):
        """Create a ``Softmax`` module.

        Parameters
        ----------
        dim : int
            The dimension to reduce.
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Softmax, self).__init__()
        self.dim = dim
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'dim={}{}'.format(self.dim, inplace_str)

    def forward(self, input):
        return functional.softmax(input, self.dim, self.inplace)


class Tanh(Module):
    """Apply tanh function.

    The **Tanh** function is defined as:

    .. math:: \\text{Tanh}(x) = \\frac{\\exp(x) - \\exp(-x)}{\\exp(x) + \\exp(-x)}

    Examples:

    ```python
    m = torch.nn.Tanh()
    x = torch.randn(2, 3)
    y = m(x)
    ```

    See Also
    --------
    `torch.nn.functional.tanh(...)`_

    """

    def __init__(self, inplace=False):
        """Create a ``Tanh`` module.

        Parameters
        ----------
        inplace : bool, optional, default=False
            Whether to do the operation in-place.

        """
        super(Tanh, self).__init__()
        self.inplace = inplace

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str

    def forward(self, input):
        return functional.tanh(input, inplace=self.inplace)


class dtype(str):
    """The basic data type.

    Following data types are defined:

    * ``torch.float16`` or ``torch.half``: 16-bit half-precision floating-point.

    * ``torch.bfloat16``: 16-bit brain floating-point.

    * ``torch.float32`` or ``torch.float``: 32-bit single-precision floating-point.

    * ``torch.float64`` or ``torch.double``: 64-bit double-precision floating-point.

    * ``torch.bfloat16``: 16-bit truncated floating-point.

    * ``torch.complex32``: 32-bit single-precision complex.

    * ``torch.complex64``: 64-bit single-precision complex.

    * ``torch.complex128``: 128-bit double-precision complex.

    * ``torch.int8``: 8-bit signed integer.

    * ``torch.uint8``: 8-bit unsigned integer.

    * ``torch.int16`` or ``torch.short``: 16-bit signed integer.

    * ``torch.int32`` or ``torch.int``: 32-bit signed integer.

    * ``torch.int64`` or ``torch.long``: 64-bit signed integer.

    * ``torch.bool``: Boolean.

    * ``torch.qint8``: Quantized 8-bit signed integer.

    * ``torch.quint8``: Quantized 8-bit unsigned integer.

    * ``torch.qint32``: Quantized 32-bit signed integer.

    """

    def __init__(self, s):
        """Create a ``dtype``.

        Parameters
        ----------
        s : str
            The data type descriptor.

        """
        super(dtype, self).__init__()


float32 = float = dtype('float32')


class _BatchNorm(Module):
    """BatchNorm base module."""

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(Tensor(num_features))
            self.bias = Parameter(Tensor(num_features))
        else:
            self.register_buffer('weight', constant_ops.ones(num_features))
            self.register_buffer('bias', constant_ops.zeros(num_features))
        if self.track_running_stats:
            self.num_batches_tracked = 0
        else:
            self.num_batches_tracked = None
        self.register_buffer('running_mean', constant_ops.zeros(num_features))
        self.register_buffer('running_var', constant_ops.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked = 0

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.one_()
            self.bias.data.zero_()

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)

    def forward(self, input):
        return functional.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, training=self.training, momentum=self._get_momentum(), eps=self.eps)

    def _apply(self, fn):
        lambda_source = inspect.getsource(fn)
        if 'half_()' in lambda_source:
            return self
        if 'bfloat16_()' in lambda_source:
            return self
        return super(_BatchNorm, self)._apply(fn)

    def _get_momentum(self):
        """Return the current momentum value."""
        momentum = 0.0 if self.momentum is None else self.momentum
        if self.track_running_stats:
            if self.training:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                if self.momentum is None:
                    momentum = 1.0 / float(self.num_batches_tracked)
        else:
            momentum = 0.0
        return momentum


class BatchNorm1d(_BatchNorm):
    """Apply batch normalization over 2d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \\frac{x - \\mathrm{E}[x]}
                       {\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
                  * \\gamma + \\beta

    The running average of statistics are calculated as:

    .. math:: x_{\\text{running}} = (1 - \\text{momentum}) * x_{\\text{running}}
                                   + \\text{momentum} * x_{\\text{batch}}

    See Also
    --------
    `torch.nn.functional.batch_norm(...)`_

    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        """Create a ``BatchNorm1d`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\\epsilon`.
        momentum : float, optional, default=0.1
            The value to :math:`\\text{momentum}`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.
        track_running_stats : bool, optional, default=True
            ``True`` to using stats when switching to ``eval``.

        """
        super(BatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)


class BatchNorm2d(_BatchNorm):
    """Apply batch normalization over 3d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \\frac{x - \\mathrm{E}[x]}
                       {\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
                  * \\gamma + \\beta

    The running average of statistics are calculated as:

    .. math:: x_{\\text{running}} = (1 - \\text{momentum}) * x_{\\text{running}}
                                   + \\text{momentum} * x_{\\text{batch}}

    See Also
    --------
    `torch.nn.functional.batch_norm(...)`_

    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        """Create a ``BatchNorm2d`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\\epsilon`.
        momentum : float, optional, default=0.1
            The value to :math:`\\text{momentum}`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.
        track_running_stats : bool, optional, default=True
            ``True`` to using stats when switching to ``eval``.

        """
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)


class BatchNorm3d(_BatchNorm):
    """Apply batch normalization over 4d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \\frac{x - \\mathrm{E}[x]}
                       {\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
                  * \\gamma + \\beta

    The running average of statistics are calculated as:

    .. math:: x_{\\text{running}} = (1 - \\text{momentum}) * x_{\\text{running}}
                                   + \\text{momentum} * x_{\\text{batch}}

    See Also
    --------
    `torch.nn.functional.batch_norm(...)`_

    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        """Create a ``BatchNorm3d`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\\epsilon`.
        momentum : float, optional, default=0.1
            The value to :math:`\\text{momentum}`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.
        track_running_stats : bool, optional, default=True
            ``True`` to using stats when switching to ``eval``.

        """
        super(BatchNorm3d, self).__init__(num_features, eps, momentum, affine, track_running_stats)


class SyncBatchNorm(_BatchNorm):
    """Apply sync batch normalization over input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \\frac{x - \\mathrm{E}[x]}
                       {\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
                  * \\gamma + \\beta

    The running average of statistics are calculated as:

    .. math:: x_{\\text{running}} = (1 - \\text{momentum}) * x_{\\text{running}}
                                   + \\text{momentum} * x_{\\text{batch}}

    If :attr:`process_group` is ``None``,
    use the value of ``dragon.distributed.get_group(...)``.

    See Also
    --------
    `torch.nn.functional.sync_batch_norm(...)`_

    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None):
        """Create a ``SyncBatchNorm`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\\epsilon`.
        momentum : float, optional, default=0.1
            The value to :math:`\\text{momentum}`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.
        track_running_stats : bool, optional, default=True
            ``True`` to using stats when switching to ``eval``.
        process_group : ProcessGroup, optional
            The group for communication.

        """
        super(SyncBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if process_group is None:
            process_group = dist_backend.get_group()
        self.process_group = process_group

    def forward(self, input):
        if self.training:
            return functional.sync_batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, training=self.training, momentum=self._get_momentum(), eps=self.eps, process_group=self.process_group)
        else:
            return functional.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, training=self.training, momentum=self._get_momentum(), eps=self.eps)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        """Convert to sync batch normalization recursively.

        Parameters
        ----------
        module : dragon.vm.torch.nn.Module
            The module containing batch normalization.
        process_group : ProcessGroup, optional
            The group for communication.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The output module.

        """
        module_output = module
        if isinstance(module, _BatchNorm):
            module_output = SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats, process_group)
            if module.affine:
                module_output.weight = module.weight
                module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output


class ChannelShuffle(Module):
    """Apply group shuffle to each channel.
    `[Zhang et.al, 2017] <https://arxiv.org/abs/1707.01083>`_.

    Examples:

    ```python
    m = torch.nn.ChannelShuffle(2)
    x = torch.tensor([1, 2, 3, 4])
    print(m(x))  # [1, 3, 2, 4]
    ```

    See Also
    --------
    `torch.nn.functional.channel_shuffle(...)`_

    """

    def __init__(self, groups, dim=1):
        """Create a ``ChannelShuffle`` module.

        Parameters
        ----------
        groups : int
            The number of shuffle groups.
        dim : int, optional, default=1
            The channel dimension.

        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
        self.dim = dim

    def extra_repr(self):
        return 'groups={}, dim={}'.format(self.groups, self.dim)

    def forward(self, input):
        return functional.channel_shuffle(input, self.groups, self.dim)


class Container(Module):
    """The base container."""

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        warnings.warn("nn.Container is deprecated. All of it's functionality is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


_context = None


def _initialize_context():
    global _context
    with _context_lock:
        if _context is None:
            _context = Context()


def context():
    """Return a singleton context object."""
    if _context is None:
        _initialize_context()
    return _context


class OpExec(object):
    """The executable operator."""
    _created_instances = {}

    def __init__(self, op_type):
        self._op_type = op_type
        self._ignore_keys = {'outputs'}
        def_args = {}
        def_args_getter = OpSchema.get_args(op_type)
        if def_args_getter is not None:
            def_args = def_args_getter()
        for k, v in def_args.items():
            if k.endswith('_desc'):
                self._ignore_keys.add(k.split('_desc')[0])
        self._config_cache = {}

    @classmethod
    def get_instance(cls, op_type):
        """Return the executable operator."""
        try:
            instance = cls._created_instances[op_type]
        except KeyError:
            instance = OpExec(op_type)
            cls._created_instances[op_type] = instance
        return instance

    def get_config(self, device, **kwargs):
        """Return the execution config."""
        cache_key = self._op_type + '/' + str(device)
        for k, v in kwargs.items():
            if k not in self._ignore_keys:
                cache_key += '/' + str(v)
        try:
            return self._config_cache[cache_key]
        except KeyError:
            def_args, feed_dict = {}, {}
            def_args_getter = OpSchema.get_args(self._op_type)
            if def_args_getter is not None:
                def_args = def_args_getter(**kwargs)
            device = def_args.pop('device', device)
            check_device = def_args.pop('check_device', True)
            no_grad = def_args.pop('no_grad', False)
            for k, v in def_args.items():
                if k.endswith('_desc') and v:
                    name = k.split('_desc')[0]
                    feed_dict[name] = v
                    def_args[k] = '$NAME/' + name
            op_def = proto_util.make_operator_def(op_type=self._op_type, name=kwargs.get('name', ''), device_option=device.to_proto(False), cache_key=cache_key, to_impl=True, **def_args)
            config = {'def': op_def, 'device': device, 'check_device': check_device, 'no_grad': no_grad, 'feed_dict': feed_dict}
            self._config_cache[cache_key] = config
            return config

