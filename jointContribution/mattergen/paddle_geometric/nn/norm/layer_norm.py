from typing import List, Optional, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

from paddle_geometric.typing import OptTensor
from paddle_geometric.utils import degree, scatter


class LayerNorm(Layer):
    r"""Applies layer normalization over each individual example in a batch
    of features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated across all nodes and all
    node channels separately for each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        mode (str, optinal): The normalization mode to use for layer
            normalization (:obj:`"graph"` or :obj:`"node"`). If :obj:`"graph"`
            is used, each graph will be considered as an element to be
            normalized. If `"node"` is used, each node will be considered as
            an element to be normalized. (default: :obj:`"graph"`)
    """
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        mode: str = 'graph',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps
        self.affine = affine
        self.mode = mode

        if affine:
            self.weight = paddle.create_parameter(shape=[in_channels], dtype='float32', default_initializer=paddle.nn.initializer.Constant(1.0))
            self.bias = paddle.create_parameter(shape=[in_channels], dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        paddle.nn.initializer.XavierUniform()(self.weight)
        paddle.nn.initializer.Constant(0.0)(self.bias)

    def forward(self, x: Tensor, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (paddle.Tensor): The source tensor.
            batch (paddle.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        if self.mode == 'graph':
            if batch is None:
                x = x - x.mean()
                out = x / (x.std(unbiased=False) + self.eps)

            else:
                if batch_size is None:
                    batch_size = int(batch.max()) + 1

                norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
                norm = norm.mul_(x.shape[-1]).reshape(-1, 1)

                mean = scatter(x, batch, dim=0, dim_size=batch_size,
                               reduce='sum').sum(axis=-1, keepdim=True) / norm

                x = x - mean.index_select(0, batch)

                var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                              reduce='sum').sum(axis=-1, keepdim=True)
                var = var / norm

                out = x / (var + self.eps).sqrt().index_select(0, batch)

            if self.weight is not None and self.bias is not None:
                out = out * self.weight + self.bias

            return out

        if self.mode == 'node':
            return F.layer_norm(x, (self.in_channels, ), self.weight,
                                self.bias, self.eps)

        raise ValueError(f"Unknown normalization mode: {self.mode}")

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'affine={self.affine}, mode={self.mode})')


class HeteroLayerNorm(Layer):
    r"""Applies layer normalization over each individual example in a batch
    of heterogeneous features as described in the `"Layer Normalization"
    <https://arxiv.org/abs/1607.06450>`_ paper.
    Compared to :class:`LayerNorm`, :class:`HeteroLayerNorm` applies
    normalization individually for each node or edge type.

    Args:
        in_channels (int): Size of each input sample.
        num_types (int): The number of types.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        mode (str, optional): The normalization mode to use for layer
            normalization (:obj:`"node"`). If `"node"` is used, each node will
            be considered as an element to be normalized.
            (default: :obj:`"node"`)
    """
    def __init__(
        self,
        in_channels: int,
        num_types: int,
        eps: float = 1e-5,
        affine: bool = True,
        mode: str = 'node',
    ):
        super().__init__()
        assert mode == 'node'

        self.in_channels = in_channels
        self.num_types = num_types
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = paddle.create_parameter(shape=[num_types, in_channels], dtype='float32')
            self.bias = paddle.create_parameter(shape=[num_types, in_channels], dtype='float32')
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.affine:
            paddle.nn.initializer.Constant(1.0)(self.weight)
            paddle.nn.initializer.Constant(0.0)(self.bias)

    def forward(
        self,
        x: Tensor,
        type_vec: OptTensor = None,
        type_ptr: Optional[Union[Tensor, List[int]]] = None,
    ) -> Tensor:
        r"""Forward pass.

        .. note::
            Either :obj:`type_vec` or :obj:`type_ptr` needs to be specified.
            In general, relying on :obj:`type_ptr` is more efficient in case
            the input tensor is sorted by types.

        Args:
            x (paddle.Tensor): The input features.
            type_vec (paddle.Tensor, optional): A vector that maps each entry to
                a type. (default: :obj:`None`)
            type_ptr (paddle.Tensor or List[int]): A vector denoting the
                boundaries of types. (default: :obj:`None`)
        """
        if type_vec is None and type_ptr is None:
            raise ValueError("Either 'type_vec' or 'type_ptr' must be given")

        out = F.layer_norm(x, (self.in_channels, ), None, None, self.eps)

        if self.affine:
            if type_ptr is not None:
                h = paddle.empty_like(out)
                for i, (s, e) in enumerate(zip(type_ptr[:-1], type_ptr[1:])):
                    h[s:e] = out[s:e] * self.weight[i] + self.bias[i]
                out = h
            else:
                out = out * self.weight[type_vec] + self.bias[type_vec]

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'num_types={self.num_types})')
