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

from __future__ import annotations

from typing import Optional

import paddle
from paddle_scatter import scatter as paddle_scatter


def _scatter_sum_dim0(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    if index.ndim != 1:
        raise ValueError("index must be 1-D when using dim=0 safe scatter.")
    if src.shape[0] != index.shape[0]:
        raise ValueError("src.shape[0] must equal index.shape[0] for dim=0 safe scatter.")

    if dim_size is None:
        if index.shape[0] == 0:
            dim_size = 0
        else:
            dim_size = int(index.max().item()) + 1

    out_shape = [dim_size] + list(src.shape[1:])
    out = paddle.zeros(out_shape, dtype=src.dtype)
    if src.shape[0] == 0:
        return out

    scatter_index = index.cast(paddle.int64).reshape([-1, 1])
    return paddle.scatter_nd_add(out, scatter_index, src)


def safe_scatter(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: Optional[str] = "sum",
) -> paddle.Tensor:
    if dim < 0:
        dim = dim + src.ndim

    use_safe_impl = (
        dim == 0
        and index.ndim == 1
        and reduce in {"sum", "add", "mean"}
        and src.shape[0] == index.shape[0]
    )

    if not use_safe_impl:
        return paddle_scatter(
            src=src,
            index=index,
            dim=dim,
            out=out,
            dim_size=dim_size,
            reduce=reduce,
        )

    sum_out = _scatter_sum_dim0(src, index, dim_size)
    if reduce in {"sum", "add"}:
        result = sum_out
    else:
        counts = _scatter_sum_dim0(
            src=paddle.ones([index.shape[0]], dtype=sum_out.dtype),
            index=index,
            dim_size=sum_out.shape[0],
        )
        counts = paddle.clip(counts, min=1)

        if sum_out.ndim > 1:
            counts = counts.reshape([counts.shape[0]] + [1] * (sum_out.ndim - 1))

        if sum_out.is_floating_point():
            result = sum_out / counts
        else:
            result = paddle.floor_divide(sum_out, counts.cast(sum_out.dtype))

    if out is None:
        return result

    paddle.assign(result, out)
    return out
