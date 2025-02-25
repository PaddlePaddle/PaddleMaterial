import sys


from typing import Literal, Optional

import paddle
from mattergen.diffusion.corruption.sde_lib import SDE, maybe_expand
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.training.field_loss import aggregate_per_sample
from paddle_utils import *


def get_pbc_offsets(pbc: paddle.Tensor, max_offset_integer: int = 3) -> paddle.Tensor:
    """Build the Cartesian product of integer offsets of the periodic boundary. That is, if dim=3 and max_offset_integer=1 we build the (2*1 + 1)^3 = 27
       possible combinations of the Cartesian product of (i,j,k) for i,j,k in -max_offset_integer, ..., max_offset_integer. Then, we construct
       the tensor of integer offsets of the pbc vectors, i.e., L_{ijk} = row_stack([i * l_1, j * l_2, k * l_3]).

    Args:
        pbc (torch.Tensor, [batch_size, dim, dim]): The input pbc matrix.
        max_offset_integer (int): The maximum integer offset per dimension to consider for the Cartesian product. Defaults to 3.

    Returns:
        torch.Tensor, [batch_size, (2 * max_offset_integer + 1)^dim, dim]: The tensor containing the integer offsets of the pbc vectors.
    """
    offset_range = paddle.arange(start=-max_offset_integer, end=max_offset_integer + 1)
    meshgrid = paddle.stack(
        x=list(
            [i.T for i in paddle.meshgrid(offset_range, offset_range, offset_range)]
        ),
        axis=-1,
    )
    offset = (pbc[:, None, None, None] * meshgrid[None, :, :, :, :, None].astype("float32")).sum(axis=-2)
    pbc_offset_per_molecule = offset.reshape(tuple(pbc.shape)[0], -1, 3)
    return pbc_offset_per_molecule


def wrapped_normal_score(
    x: paddle.Tensor,
    mean: paddle.Tensor,
    wrapping_boundary: paddle.Tensor,
    variance_diag: paddle.Tensor,
    batch: paddle.Tensor,
    max_offset_integer: int = 3,
) -> paddle.Tensor:
    """Approximate the the score of a 3D wrapped normal distribution with diagonal covariance matrix w.r.t. x via a truncated sum.
       See docstring of `wrapped_normal_score` for details about the arguments

    Args:
        x (torch.Tensor, [num_atoms, dim])
        mean (torch.Tensor, [num_atoms, dim])
        wrapping_boundary (torch.Tensor, [num_molecules, dim, dim])
        variance_diag (torch.Tensor, [num_atoms,])
        batch (torch.Tensor, [num_atoms, ])
        max_offset_integer (int), Defaults to 3.

    Returns:
        torch.Tensor, [num_atoms, dim]: The approximated score of the wrapped normal distribution.
    """
    offset_add = get_pbc_offsets(wrapping_boundary, max_offset_integer)
    diffs_k = (x - mean)[:, None] + offset_add[batch]
    dists_sqr_k = diffs_k.pow(y=2).sum(axis=-1)
    score_softmax = paddle.nn.functional.softmax(
        x=-dists_sqr_k / (2 * variance_diag[:, None]), axis=-1
    )
    score = -(score_softmax[:, :, None] * diffs_k).sum(axis=-2) / variance_diag[:, None]
    return score


def wrapped_normal_loss(
    *,
    corruption: SDE,
    score_model_output: paddle.Tensor,
    t: paddle.Tensor,
    batch_idx: Optional[paddle.Tensor],
    batch_size: int,
    x: paddle.Tensor,
    noisy_x: paddle.Tensor,
    reduce: Literal["sum", "mean"],
    batch: BatchedData,
    **_
) -> paddle.Tensor:
    """Compute the loss for a wrapped normal distribution.
    Compares the score of the wrapped normal distribution to the score of the score model.
    """
    assert len(t) == batch_size
    _, std = corruption.marginal_prob(
        x=paddle.zeros(shape=(tuple(x.shape)[0], 1)),
        t=t,
        batch_idx=batch_idx,
        batch=batch,
    )
    pred: paddle.Tensor = score_model_output
    if pred.ndim != 2:
        raise NotImplementedError
    assert hasattr(
        corruption, "wrapping_boundary"
    ), "SDE must be a WrappedSDE, i.e., must have a wrapping boundary."
    wrapping_boundary = corruption.wrapping_boundary
    wrapping_boundary = wrapping_boundary * paddle.eye(num_rows=tuple(x.shape)[-1])[
        None
    ].expand(shape=[batch_size, -1, -1])
    target = (
        wrapped_normal_score(
            x=noisy_x,
            mean=x,
            wrapping_boundary=wrapping_boundary,
            variance_diag=std.squeeze() ** 2,
            batch=batch_idx,
        )
        * std
    )
    delta = target - pred
    losses = delta.square()
    return aggregate_per_sample(losses, batch_idx, reduce=reduce, batch_size=batch_size)
