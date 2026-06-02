from __future__ import annotations

import paddle
from ..paddle_utils import *

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Add `extra_repr` into DropPath implemented by timm
for displaying more info.
"""

from e3nn import o3


def drop_path(
    x: paddle.Tensor, drop_prob: float = 0.0, training: bool = False
) -> paddle.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(paddle.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class GraphDropPath(paddle.nn.Module):
    """
    Consider batch for graph data when dropping paths.
    """

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: paddle.Tensor, batch) -> paddle.Tensor:
        batch_size = batch._max() + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)
        ones = paddle.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        return x * drop[batch]

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class EquivariantDropout(paddle.nn.Module):
    def __init__(self, irreps, drop_prob: float) -> None:
        super().__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = paddle.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(
            irreps, o3.Irreps(f"{self.num_irreps}x0e")
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = x.shape[0], self.num_irreps
        mask = paddle.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        return self.mul(x, mask)


class EquivariantScalarsDropout(paddle.nn.Module):
    def __init__(self, irreps, drop_prob: float) -> None:
        super().__init__()
        self.irreps = irreps
        self.drop_prob = drop_prob

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        out = []
        start_idx = 0
        for mul, ir in self.irreps:
            temp = x.narrow(-1, start_idx, mul * ir.dim)
            start_idx += mul * ir.dim
            if ir.is_scalar():
                temp = paddle.nn.functional.dropout(
                    temp, p=self.drop_prob, training=self.training
                )
            out.append(temp)
        return paddle.cat(out, dim=-1)

    def extra_repr(self) -> str:
        return f"irreps={self.irreps}, drop_prob={self.drop_prob}"


class EquivariantDropoutArraySphericalHarmonics(paddle.nn.Module):
    def __init__(self, drop_prob: float, drop_graph: bool = False) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.drop = paddle.nn.Dropout(drop_prob, True)
        self.drop_graph = drop_graph

    def forward(self, x: paddle.Tensor, batch=None) -> paddle.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        assert len(x.shape) == 3
        if self.drop_graph:
            assert batch is not None
            batch_size = batch._max() + 1
            shape = batch_size, 1, x.shape[2]
            mask = paddle.ones(shape, dtype=x.dtype, device=x.device)
            mask = self.drop(mask)
            out = x * mask[batch]
        else:
            shape = x.shape[0], 1, x.shape[2]
            mask = paddle.ones(shape, dtype=x.dtype, device=x.device)
            mask = self.drop(mask)
            out = x * mask
        return out

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}, drop_graph={self.drop_graph}"
