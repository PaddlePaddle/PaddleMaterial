from typing import Optional

import paddle
from paddle import Tensor
from paddle_geometric.nn import knn  # Assuming knn is implemented in paddle_geometric
from paddle_geometric.utils import scatter  # Assuming scatter is available in paddle_geometric


def knn_interpolate(x: Tensor, pos_x: Tensor, pos_y: Tensor,
                    batch_x: Optional[Tensor] = None, batch_y: Optional[Tensor] = None,
                    k: int = 3, num_workers: int = 1):
    r"""The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.

    For each point :math:`y` with position :math:`\mathbf{p}(y)`, its
    interpolated features :math:`\mathbf{f}(y)` are given by

    .. math::
        \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
        w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
        \mathbf{p}(x_i))^2}

    and :math:`\{ x_1, \ldots, x_k \}` denoting the :math:`k` nearest points
    to :math:`y`.

    Args:
        x (paddle.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        pos_x (paddle.Tensor): Node position matrix
            :math:`\in \mathbb{R}^{N \times d}`.
        pos_y (paddle.Tensor): Upsampled node position matrix
            :math:`\in \mathbb{R}^{M \times d}`.
        batch_x (paddle.Tensor, optional): Batch vector
            :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{X}` to a specific example.
            (default: :obj:`None`)
        batch_y (paddle.Tensor, optional): Batch vector
            :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{Y}` to a specific example.
            (default: :obj:`None`)
        k (int, optional): Number of neighbors. (default: :obj:`3`)
        num_workers (int, optional): Number of workers to use for computation.
            Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    """
    with paddle.no_grad():
        # Assuming knn is implemented in paddle_geometric, otherwise implement it manually
        assign_index = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y,
                           num_workers=num_workers)
        y_idx, x_idx = assign_index[0], assign_index[1]

        # Calculate pairwise distance squared between points
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(axis=-1, keepdim=True)

        # Calculate the weights for interpolation
        weights = 1.0 / paddle.clamp(squared_distance, min=1e-16)

    # Interpolate the features using the calculated weights
    y = scatter(x[x_idx] * weights, y_idx, 0, pos_y.shape[0], reduce='sum')
    y = y / scatter(weights, y_idx, 0, pos_y.shape[0], reduce='sum')

    return y
