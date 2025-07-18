import math
import paddle
from paddle import Tensor
import paddle.nn as nn


class PositionalEncoding(nn.Layer):
    r"""The positional encoding scheme from the `"Attention Is All You Need"
    <https://arxiv.org/abs/1706.03762>`_ paper.

    .. math::

        PE(x)_{2 \cdot i} &= \sin(x / 10000^{2 \cdot i / d})

        PE(x)_{2 \cdot i + 1} &= \cos(x / 10000^{2 \cdot i / d})

    where :math:`x` is the position and :math:`i` is the dimension.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
        base_freq (float, optional): The base frequency of sinusoidal
            functions. (default: :obj:`1e-4`)
        granularity (float, optional): The granularity of the positions. If
            set to smaller value, the encoder will capture more fine-grained
            changes in positions. (default: :obj:`1.0`)
    """
    def __init__(
        self,
        out_channels: int,
        base_freq: float = 1e-4,
        granularity: float = 1.0,
    ):
        super(PositionalEncoding, self).__init__()

        if out_channels % 2 != 0:
            raise ValueError(f"Cannot use sinusoidal positional encoding with "
                             f"odd 'out_channels' (got {out_channels}).")

        self.out_channels = out_channels
        self.base_freq = base_freq
        self.granularity = granularity

        frequency = paddle.logspace(0, 1, out_channels // 2, base_freq)
        self.register_buffer('frequency', frequency)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        x = x / self.granularity if self.granularity != 1.0 else x
        out = x.unsqueeze(-1) * self.frequency.unsqueeze(0)
        return paddle.concat([paddle.sin(out), paddle.cos(out)], axis=-1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'


class TemporalEncoding(nn.Layer):
    r"""The time-encoding function from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.

    It first maps each entry to a vector with exponentially decreasing values,
    and then uses the cosine function to project all values to range
    :math:`[-1, 1]`.

    .. math::

        y_{i} = \cos \left(x \cdot \sqrt{d}^{-(i - 1)/\sqrt{d}} \right)

    where :math:`d` defines the output feature dimension, and
    :math:`1 \leq i \leq d`.

    Args:
        out_channels (int): Size :math:`d` of each output sample.
    """
    def __init__(self, out_channels: int):
        super(TemporalEncoding, self).__init__()
        self.out_channels = out_channels

        sqrt = math.sqrt(out_channels)
        weight = 1.0 / sqrt**paddle.linspace(0, sqrt, out_channels).unsqueeze(0)
        self.register_buffer('weight', weight)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor) -> Tensor:
        return paddle.cos(paddle.matmul(x.unsqueeze(-1), self.weight))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels})'
