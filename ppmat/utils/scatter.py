# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is adapted from https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py

from typing import Optional, Tuple

import paddle


def _broadcast(src: paddle.Tensor, other: paddle.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.shape)
    return src


def scatter_argmin(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    """Return the source index of the minimum value in each group.

    ``src`` and ``index`` must be one-dimensional. Empty groups are assigned
    ``-1``. Ties are resolved by selecting the first occurrence in ``src``.
    """
    if src.ndim != 1 or index.ndim != 1 or src.shape[0] != index.shape[0]:
        raise ValueError("src and index must be one-dimensional with equal length")

    if dim_size is None:
        dim_size = 0 if index.shape[0] == 0 else int(index.max()) + 1

    out = paddle.full([dim_size], -1, dtype="int64")
    if index.shape[0] == 0:
        return out

    order = paddle.argsort(src, stable=True)
    groups, first = paddle.unique(index[order], return_index=True)
    return paddle.scatter(out, groups, order[first], overwrite=True)


def scatter_argmax(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    """Return the source index of the maximum value in each group.

    ``src`` and ``index`` must be one-dimensional. Empty groups are assigned
    ``0``. Ties are resolved by selecting the last occurrence in ``src``.
    """
    if src.ndim != 1 or index.ndim != 1 or src.shape[0] != index.shape[0]:
        raise ValueError("src and index must be one-dimensional with equal length")

    if dim_size is None:
        dim_size = 0 if index.shape[0] == 0 else int(index.max()) + 1
    if index.shape[0] == 0:
        return paddle.zeros([dim_size], dtype="int64")

    # paddle.geometric.segment_max requires sorted segment ids; sort first.
    sorted_order = paddle.argsort(index, stable=True)
    sorted_index = index[sorted_order]
    sorted_src = src[sorted_order]
    seg_size = int(sorted_index.max().item()) + 1

    group_counts = paddle.bincount(sorted_index, minlength=seg_size).cast(paddle.bool)
    empty_mask = ~group_counts

    max_values = paddle.geometric.segment_max(sorted_src, sorted_index)
    n = src.shape[0]
    weights = paddle.arange(n, dtype=paddle.float32)
    is_max = sorted_src == max_values[sorted_index]
    max_weights = paddle.where(is_max, weights, paddle.to_tensor(-float("inf")))
    argmax_sorted = paddle.geometric.segment_max(max_weights, sorted_index)
    argmax = paddle.where(
        empty_mask,
        paddle.zeros([seg_size], dtype="int64"),
        sorted_order[argmax_sorted.cast(paddle.int64)],
    )
    out = paddle.zeros([dim_size], dtype="int64")
    if seg_size > 0:
        out[:seg_size] = argmax
    return out


def _scatter_sum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = paddle.zeros(size, dtype=src.dtype)
    # FIXME: Paddle's put_along_axis backward (PutAlongAxisGradNode) crashes
    # for dim=0; use one-hot + matmul as drop-in replacement.
    if dim == 0:
        # _broadcast expanded index to src.shape; collapse back to 1D via first column
        idx_1d = index.reshape([-1, src.shape[1]])[:, 0] if index.ndim > 1 else index
        one_hot = paddle.nn.functional.one_hot(idx_1d, out.shape[0]).cast(src.dtype)
        # one_hot: [N, out_dim] -> [out_dim, N] @ [N, C] = [out_dim, C]
        return paddle.mm(one_hot.t(), src)
    else:
        return paddle.put_along_axis(
            arr=out, indices=index, values=src, axis=dim, reduce="add"
        )


def _scatter_mean(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    out = _scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.shape[dim]

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = paddle.ones(index.shape, dtype=src.dtype)
    count = _scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out = paddle.divide(out, count)
    else:
        out = paddle.floor_divide(out, count)
    return out


def _scatter_min(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = paddle.full(size, float("inf"), dtype=src.dtype)
    return paddle.put_along_axis(
        arr=out, indices=index, values=src, axis=dim, reduce="amin"
    )


def scatter(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    """
    Implement paddle version API like torch_scatter.scatter
    """
    if reduce == "sum" or reduce == "add":
        return _scatter_sum(src, index, dim, out, dim_size)
    elif reduce == "mean":
        return _scatter_mean(src, index, dim, out, dim_size)
    elif reduce == "min":
        return _scatter_min(src, index, dim, out, dim_size)
    else:
        raise ValueError("Only support add, mean, or min")


def scatter_mean(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
):
    return _scatter_mean(src, index, dim, out, dim_size)


def scatter_sum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
):
    return _scatter_sum(src, index, dim, out, dim_size)


def scatter_min_with_argmin(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    if dim_size is None:
        dim_size = 0 if index.shape[0] == 0 else int(index.max().item()) + 1
    if index.shape[0] == 0:
        return (
            paddle.full([dim_size], float("inf"), dtype=src.dtype),
            paddle.full([dim_size], dim_size, dtype="int64"),
        )

    # paddle.geometric.segment_min requires sorted segment ids; sort first.
    sorted_order = paddle.argsort(index, stable=True)
    sorted_index = index[sorted_order]
    sorted_src = src[sorted_order]

    seg_size = int(sorted_index.max().item()) + 1
    group_counts = paddle.bincount(sorted_index, minlength=seg_size).cast(paddle.bool)
    empty_mask = ~group_counts

    min_values = paddle.geometric.segment_min(sorted_src, sorted_index)
    min_values = paddle.where(
        empty_mask,
        paddle.full([seg_size], float("inf"), dtype=src.dtype),
        min_values,
    )
    if seg_size < dim_size:
        min_values = paddle.concat(
            [
                min_values,
                paddle.full(
                    [dim_size - seg_size], float("inf"), dtype=src.dtype
                ),
            ]
        )

    n = src.shape[0]
    weights = paddle.arange(n, dtype=paddle.float32)
    is_min = (sorted_src == min_values[sorted_index])
    min_weights = paddle.where(is_min, weights, paddle.to_tensor(float('inf')))
    argmin_sorted = paddle.geometric.segment_min(min_weights, sorted_index)
    argmin = paddle.where(
        empty_mask,
        paddle.full([seg_size], dim_size, dtype=paddle.int64),
        sorted_order[argmin_sorted.cast(paddle.int64)],
    )
    if seg_size < dim_size:
        argmin = paddle.concat(
            [argmin, paddle.full([dim_size - seg_size], dim_size, dtype="int64")]
        )
    return min_values, argmin


def scatter_min_indices(
    ov_row: paddle.Tensor,
    ov_col: paddle.Tensor,
    n_total: int,
) -> paddle.Tensor:
    if ov_row.shape[0] == 0:
        return paddle.arange(n_total, dtype=paddle.int64)
    # paddle.geometric.segment_min requires sorted segment ids; sort first.
    order = paddle.argsort(ov_row, stable=True)
    sorted_row = ov_row[order]
    sorted_col = ov_col[order]
    min_per_row = paddle.geometric.segment_min(
        sorted_col.cast(paddle.float32), sorted_row
    )
    unique_rows = paddle.unique(ov_row)
    result = paddle.arange(n_total, dtype=paddle.float32)
    result = paddle.scatter(
        result, unique_rows, min_per_row[unique_rows].cast(paddle.float32)
    )
    return result.cast(paddle.int64)
