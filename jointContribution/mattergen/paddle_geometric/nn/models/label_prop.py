from typing import Callable, Optional

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.nn import functional as F

from paddle_geometric.nn import MessagePassing
from paddle_geometric.nn.conv.gcn_conv import gcn_norm
from paddle_geometric.typing import Adj, OptTensor, SparseTensor
from paddle_geometric.utils import one_hot, spmm


class LabelPropagation(MessagePassing):
    r"""The label propagation operator, firstly introduced in the
    `"Learning from Labeled and Unlabeled Data with Label Propagation"
    <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper.

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.
    This concrete implementation here is derived from the `"Combining Label
    Propagation And Simple Models Out-performs Graph Neural Networks"
    <https://arxiv.org/abs/2010.13993>`_ paper.

    .. note::

        For an example of using the :class:`LabelPropagation`, see
        `examples/label_prop.py
        <https://github.com/pyg-team/pypaddle_geometric/blob/master/examples/
        label_prop.py>`_.

    Args:
        num_layers (int): The number of propagations.
        alpha (float): The :math:`\alpha` coefficient.
    """
    def __init__(self, num_layers: int, alpha: float):
        super().__init__(aggr='add')
        self.num_layers = num_layers
        self.alpha = alpha

    @paddle.no_grad()
    def forward(
        self,
        y: Tensor,
        edge_index: Adj,
        mask: OptTensor = None,
        edge_weight: OptTensor = None,
        post_step: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            y (paddle.Tensor): The ground-truth label information
                :math:`\mathbf{Y}`.
            edge_index (paddle.Tensor or SparseTensor): The edge connectivity.
            mask (paddle.Tensor, optional): A mask or index tensor denoting
                which nodes are used for label propagation.
                (default: :obj:`None`)
            edge_weight (paddle.Tensor, optional): The edge weights.
                (default: :obj:`None`)
            post_step (callable, optional): A post step function specified
                to apply after label propagation. If no post step function
                is specified, the output will be clamped between 0 and 1.
                (default: :obj:`None`)
        """
        if y.dtype == paddle.int64 and y.shape[0] == y.numel():
            y = one_hot(y.reshape([-1]))

        out = y
        if mask is not None:
            out = paddle.zeros_like(y)
            out[mask] = y[mask]

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.shape[0],
                                               add_self_loops=False)

        res = (1 - self.alpha) * out
        for _ in range(self.num_layers):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight)
            out *= self.alpha
            out.add_(res)
            if post_step is not None:
                out = post_step(out)
            else:
                out = paddle.clip(out, min=0., max=1.)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.reshape([-1, 1]) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_layers={self.num_layers}, '
                f'alpha={self.alpha})')
