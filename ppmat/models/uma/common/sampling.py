from __future__ import annotations

import paddle

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math


def CalcSpherePoints(num_points: int, device: str = "cpu") -> paddle.Tensor:
    goldenRatio = (1 + 5**0.5) / 2
    i = paddle.arange(num_points, device=device).view(-1, 1)
    theta = 2 * math.pi * i / goldenRatio
    phi = paddle.acos(1 - 2 * (i + 0.5) / num_points)
    points = paddle.cat(
        [
            paddle.cos(theta) * paddle.sin(phi),
            paddle.sin(theta) * paddle.sin(phi),
            paddle.cos(phi),
        ],
        dim=1,
    )
    pt_cross = points.view(1, -1, 3) - points.view(-1, 1, 3)
    pt_cross = paddle.sum(pt_cross**2, dim=2)
    pt_cross = paddle.exp(-pt_cross / (0.5 * 0.3))
    scalar = 1.0 / paddle.sum(pt_cross, dim=1)
    scalar = num_points * scalar / paddle.sum(scalar)
    return points * scalar.view(-1, 1)


def CalcSpherePointsRandom(num_points: int, device) -> paddle.Tensor:
    pts = 2.0 * (paddle.rand(num_points, 3, device=device) - 0.5)
    radius = paddle.sum(pts**2, dim=1)
    while paddle.compat.max(radius) > 1.0:
        replace_pts = 2.0 * (paddle.rand(num_points, 3, device=device) - 0.5)
        replace_mask = radius.gt(0.99)
        pts.masked_scatter_(replace_mask.view(-1, 1).repeat(1, 3), replace_pts)
        radius = paddle.sum(pts**2, dim=1)
    return pts / radius.view(-1, 1)
