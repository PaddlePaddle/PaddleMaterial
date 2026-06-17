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

EPS = 1e-07


class Safeacos(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_clamped = x.clamp(-1 + EPS, 1 - EPS)
        ctx.save_for_backward(x_clamped)
        return paddle.acos(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x_clamped,) = ctx.saved_tensor()
        denom = paddle.sqrt(1 - x_clamped.pow(2)).clamp(min=EPS)
        return -grad_output / denom


class Safeatan2(paddle.autograd.Function):
    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return paddle.atan2(x=y, y=x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensor()
        denom = (x.pow(2) + y.pow(2)).clamp(min=EPS)
        return x / denom * grad_output, -y / denom * grad_output


def init_edge_rot_euler_angles(edge_distance_vec):
    xyz = paddle.nn.functional.normalize(edge_distance_vec).clamp(-1.0, 1.0)
    x, y, z = paddle.compat.split(xyz, 1, dim=1)
    beta = Safeacos.apply(y.squeeze(-1))
    alpha = Safeatan2.apply(x.squeeze(-1), z.squeeze(-1))
    gamma = paddle.rand_like(alpha) * 2 * paddle.pi
    return -gamma, -beta, -alpha


def wigner_D(
    lv: int,
    alpha: paddle.Tensor,
    beta: paddle.Tensor,
    gamma: paddle.Tensor,
    _Jd: list[paddle.Tensor],
) -> paddle.Tensor:
    alpha, beta, gamma = paddle.broadcast_tensors(input=[alpha, beta, gamma])
    J = _Jd[lv]
    Xa = _z_rot_mat(alpha, lv)
    Xb = _z_rot_mat(beta, lv)
    Xc = _z_rot_mat(gamma, lv)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle: paddle.Tensor, lv: int) -> paddle.Tensor:
    M = angle.new_zeros((*angle.shape, 2 * lv + 1, 2 * lv + 1))
    inds = list(range(0, 2 * lv + 1, 1))
    reversed_inds = list(range(2 * lv, -1, -1))
    frequencies = list(range(lv, -lv - 1, -1))
    for i in range(len(frequencies)):
        M[..., inds[i], reversed_inds[i]] = paddle.sin(frequencies[i] * angle)
        M[..., inds[i], inds[i]] = paddle.cos(frequencies[i] * angle)
    return M


def eulers_to_wigner(
    eulers: paddle.Tensor, start_lmax: int, end_lmax: int, Jd: list[paddle.Tensor]
) -> paddle.Tensor:
    """
    set <rot_clip=True> to handle gradient instability when using gradient-based force/stress prediction.
    """
    alpha, beta, gamma = eulers
    size = int((end_lmax + 1) ** 2) - int(start_lmax**2)
    wigner = paddle.zeros(
        len(alpha), size, size, device=alpha.device, dtype=alpha.dtype
    )
    start = 0
    for lmax in range(start_lmax, end_lmax + 1):
        block = wigner_D(lmax, alpha, beta, gamma, Jd)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end
    return wigner
