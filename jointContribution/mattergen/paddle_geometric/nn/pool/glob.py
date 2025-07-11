from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle_geometric.utils import scatter


def global_add_pool(x: Tensor, batch: Optional[Tensor], size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level outputs by adding node features
    across the node dimension.

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~paddle_geometric.nn.aggr.SumAggregation` module.

    Args:
        x (paddle.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (paddle.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    Returns:
        The pooled features for each graph in the batch.
    """
    dim = -1 if isinstance(x, Tensor) and x.ndim == 1 else -2

    if batch is None:
        return paddle.sum(x, axis=dim, keepdim=x.ndim <= 2)
    return scatter(x, batch, dim=dim, dim_size=size, reduce='sum')


def global_mean_pool(x: Tensor, batch: Optional[Tensor], size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level outputs by averaging node features
    across the node dimension.

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~paddle_geometric.nn.aggr.MeanAggregation` module.

    Args:
        x (paddle.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (paddle.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    Returns:
        The pooled features for each graph in the batch.
    """
    dim = -1 if isinstance(x, Tensor) and x.ndim == 1 else -2

    if batch is None:
        return paddle.mean(x, dim=dim, keepdim=x.ndim <= 2)
    return scatter(x, batch, dim=dim, dim_size=size, reduce='mean')


def global_max_pool(x: Tensor, batch: Optional[Tensor], size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level outputs by taking the channel-wise
    maximum across the node dimension.

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    .. math::
        \mathbf{r}_i = \max_{n=1}^{N_i} \, \mathbf{x}_n.

    Functional method of the
    :class:`~paddle_geometric.nn.aggr.MaxAggregation` module.

    Args:
        x (paddle.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (paddle.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    Returns:
        The pooled features for each graph in the batch.
    """
    dim = -1 if isinstance(x, Tensor) and x.ndim == 1 else -2

    if batch is None:
        return paddle.max(x, dim=dim, keepdim=x.ndim <= 2)[0]
    return scatter(x, batch, dim=dim, dim_size=size, reduce='max')
