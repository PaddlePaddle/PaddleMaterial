import math
from typing import Any, Callable, Dict, Optional, Union

import paddle
from paddle import Tensor
from paddle.nn import Layer
from paddle_geometric.nn.conv import GCNConv, MessagePassing
from paddle_geometric.nn.resolver import activation_resolver
from paddle_geometric.nn.inits import zeros
from paddle_geometric.typing import Adj


class AntiSymmetricConv(Layer):
    r"""The anti-symmetric graph convolutional operator from the
    `"Anti-Symmetric DGN: a stable architecture for Deep Graph Networks"
    <https://openreview.net/forum?id=J3Y7cgZOOS>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        phi (MessagePassing, optional): The message passing module
            :math:`\Phi`. If set to :obj:`None`, will use a
            :class:`~paddle_geometric.nn.conv.GCNConv` layer as default.
        num_iters (int, optional): The number of times the anti-symmetric deep
            graph network operator is called. (default: :obj:`1`)
        epsilon (float, optional): The discretization step size
            :math:`\epsilon`. (default: :obj:`0.1`)
        gamma (float, optional): The strength of the diffusion :math:`\gamma`.
        act (str, optional): The non-linear activation function :math:`\sigma`,
            *e.g.*, :obj:`"tanh"` or :obj:`"relu"`. (default: :class:`"tanh"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        phi: Optional[MessagePassing] = None,
        num_iters: int = 1,
        epsilon: float = 0.1,
        gamma: float = 0.1,
        act: Union[str, Callable, None] = 'tanh',
        act_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.act = activation_resolver(act, **(act_kwargs or {}))

        if phi is None:
            phi = GCNConv(in_channels, in_channels, bias_attr=False)

        self.W = self.create_parameter(shape=[in_channels, in_channels])
        self.eye = self.create_parameter(
            shape=[in_channels, in_channels],
            default_initializer=paddle.nn.initializer.Assign(paddle.eye(in_channels)),
            stop_gradient=True
        )
        self.phi = phi

        if bias:
            self.bias = self.create_parameter(shape=[in_channels])
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.initializer.KaimingUniform()(self.W)
        self.phi.reset_parameters()
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        antisymmetric_W = self.W - self.W.transpose([1, 0]) - self.gamma * self.eye

        for _ in range(self.num_iters):
            h = self.phi(x, edge_index, *args, **kwargs)
            h = paddle.matmul(x, antisymmetric_W.transpose([1, 0])) + h

            if self.bias is not None:
                h += self.bias

            if self.act is not None:
                h = self.act(h)

            x = x + self.epsilon * h

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.in_channels}, '
                f'phi={self.phi}, '
                f'num_iters={self.num_iters}, '
                f'epsilon={self.epsilon}, '
                f'gamma={self.gamma})')
