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

from typing import Optional

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
        if src.shape[0] == 0:
            # Paddle cannot infer the ``-1`` dimension when reshaping an empty
            # tensor. Keep a zero-valued dependency so backward still produces
            # the expected empty source gradient.
            return out + src.sum() * 0
        # ``index`` is constant across all non-scatter dimensions after
        # broadcasting. Collapse it back to one group id per source row.
        idx_1d = index.reshape([index.shape[0], -1])[:, 0]
        one_hot = paddle.nn.functional.one_hot(idx_1d, out.shape[0]).cast(src.dtype)
        # Flatten arbitrary trailing dimensions for matmul, then restore them.
        flat_src = src.reshape([src.shape[0], -1])
        flat_out = paddle.mm(one_hot.t(), flat_src)
        return flat_out.reshape(out.shape)
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
