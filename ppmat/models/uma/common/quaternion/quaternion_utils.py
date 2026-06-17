# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import paddle

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

BLEND_START = -0.9
BLEND_WIDTH = 1.8


def _smooth_step_cinf(t: paddle.Tensor) -> paddle.Tensor:
    """
    C-infinity smooth step function based on the classic bump function.

    Uses f(x) = exp(-1/x) for x > 0 (0 otherwise), then:
    step(t) = f(t) / (f(t) + f(1-t)) = sigmoid((2t-1)/(t*(1-t)))

    Properties:
    - C-infinity smooth everywhere
    - All derivatives are exactly zero at t=0 and t=1
    - Values: f(0)=0, f(1)=1
    - Symmetric: f(t) + f(1-t) = 1

    Args:
        t: Input tensor, will be clamped to [0, 1]

    Returns:
        Smooth step values in [0, 1]
    """
    t_clamped = t.clamp(0, 1)
    eps = paddle.finfo(t.dtype).eps
    numerator = 2.0 * t_clamped - 1.0
    denominator = t_clamped * (1.0 - t_clamped)
    denom_safe = denominator.clamp(min=eps)
    arg = numerator / denom_safe
    result = paddle.sigmoid(arg)
    result = paddle.where(t_clamped < eps, paddle.zeros_like(result), result)
    result = paddle.where(t_clamped > 1 - eps, paddle.ones_like(result), result)
    return result


def quaternion_multiply(q1: paddle.Tensor, q2: paddle.Tensor) -> paddle.Tensor:
    """
    Multiply two quaternions: q1 * q2.

    Uses Hamilton product convention: (w, x, y, z).

    Args:
        q1: First quaternion of shape (N, 4) or (4,)
        q2: Second quaternion of shape (N, 4) or (4,)

    Returns:
        Product quaternion of shape (N, 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return paddle.stack([w, x, y, z], dim=-1)


def quaternion_y_rotation(gamma: paddle.Tensor) -> paddle.Tensor:
    """
    Create quaternion for rotation about Y-axis by angle gamma.

    Args:
        gamma: Rotation angles of shape (N,)

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    half_gamma = gamma / 2
    w = paddle.cos(half_gamma)
    x = paddle.zeros_like(gamma)
    y = paddle.sin(half_gamma)
    z = paddle.zeros_like(gamma)
    return paddle.stack([w, x, y, z], dim=-1)


def quaternion_nlerp(
    q1: paddle.Tensor, q2: paddle.Tensor, t: paddle.Tensor
) -> paddle.Tensor:
    """
    Normalized linear interpolation between quaternions.

    nlerp(q1, q2, t) = normalize((1-t) * q1 + t * q2)

    Args:
        q1: First quaternion, shape (..., 4)
        q2: Second quaternion, shape (..., 4)
        t: Interpolation parameter, shape (...)

    Returns:
        Interpolated quaternion, shape (..., 4)
    """
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    q1_aligned = paddle.where(dot < 0, -q1, q1)
    t_expanded = t.unsqueeze(-1) if t.dim() < q1.dim() else t
    result = paddle.nn.functional.normalize(
        (1.0 - t_expanded) * q1_aligned + t_expanded * q2, dim=-1
    )
    return result


def _quaternion_chart1_standard(
    ex: paddle.Tensor, ey: paddle.Tensor, ez: paddle.Tensor
) -> paddle.Tensor:
    """
    Standard quaternion: edge -> +Y directly. Singular at edge = -Y.

    Uses the half-vector formula:
        q = normalize(1 + ey, -ez, 0, ex)

    Args:
        ex, ey, ez: Edge vector components

    Returns:
        Quaternions of shape (..., 4) in (w, x, y, z) convention
    """
    w = 1.0 + ey
    x = -ez
    y = paddle.zeros_like(ex)
    z = ex
    q = paddle.stack([w, x, y, z], dim=-1)
    q_sq = paddle.sum(q**2, dim=-1, keepdim=True)
    eps = paddle.finfo(ex.dtype).eps
    norm = paddle.sqrt(paddle.clamp(q_sq, min=eps))
    return q / norm


def _quaternion_chart2_via_minus_y(
    ex: paddle.Tensor, ey: paddle.Tensor, ez: paddle.Tensor
) -> paddle.Tensor:
    """
    Alternative quaternion: edge -> +Y via -Y. Singular at edge = +Y.

    Path: edge -> -Y -> +Y (compose with 180 deg about X)

    Args:
        ex, ey, ez: Edge vector components

    Returns:
        Quaternions of shape (..., 4) in (w, x, y, z) convention
    """
    w = -ez
    x = 1.0 - ey
    y = ex
    z = paddle.zeros_like(ex)
    q = paddle.stack([w, x, y, z], dim=-1)
    q_sq = paddle.sum(q**2, dim=-1, keepdim=True)
    eps = paddle.finfo(ex.dtype).eps
    norm = paddle.sqrt(paddle.clamp(q_sq, min=eps))
    return q / norm


def quaternion_edge_to_y_stable(edge_vec: paddle.Tensor) -> paddle.Tensor:
    """
    Compute quaternion for edge -> +Y using two charts with NLERP blending.

    Uses two quaternion charts to avoid singularities:
    - Chart 1: q = normalize(1+ey, -ez, 0, ex) - singular at -Y
    - Chart 2: q = normalize(-ez, 1-ey, ex, 0) - singular at +Y

    NLERP blend in ey in [-0.9, 0.9]:
    - Uses Chart 2 when near -Y (stable there)
    - Uses Chart 1 when near +Y (stable there)
    - Smoothly interpolates in between

    Args:
        edge_vec: Edge vectors of shape (N, 3), assumed normalized

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) convention
    """
    ex = edge_vec[..., 0]
    ey = edge_vec[..., 1]
    ez = edge_vec[..., 2]
    q_chart1 = _quaternion_chart1_standard(ex, ey, ez)
    q_chart2 = _quaternion_chart2_via_minus_y(ex, ey, ez)
    t = (ey - BLEND_START) / BLEND_WIDTH
    t_smooth = _smooth_step_cinf(t)
    q = quaternion_nlerp(q_chart2, q_chart1, t_smooth)
    return q
