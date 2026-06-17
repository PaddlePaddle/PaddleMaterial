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

import math

import numpy as np
from ppmat.models.common.e3nn.o3 import FromS2Grid, ToS2Grid


def _as_paddle_tensor(x):
    if paddle.is_tensor(x):
        return x
    if hasattr(x, "detach"):
        return paddle.to_tensor(x.detach().cpu().numpy())
    return paddle.to_tensor(np.asarray(x))


class CoefficientMapping(paddle.nn.Module):
    """
    Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

    Args:
        lmax_list (list:int):   List of maximum degree of the spherical harmonics
        mmax_list (list:int):   List of maximum order of the spherical harmonics
        use_rotate_inv_rescale (bool):  Whether to pre-compute inverse rotation rescale matrices
    """

    def __init__(self, lmax, mmax):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        l_harmonic = paddle.tensor([]).long()
        m_harmonic = paddle.tensor([]).long()
        m_complex = paddle.tensor([]).long()
        for l in range(self.lmax + 1):
            mmax = min(self.mmax, l)
            m = paddle.arange(-mmax, mmax + 1).long()
            m_complex = paddle.cat([m_complex, m], dim=0)
            m_harmonic = paddle.cat([m_harmonic, paddle.abs(m).long()], dim=0)
            l_harmonic = paddle.cat([l_harmonic, m.fill_(l).long()], dim=0)
        self.res_size = len(l_harmonic)
        num_coefficients = len(l_harmonic)
        to_m = paddle.zeros([num_coefficients, num_coefficients])
        self.m_size = paddle.zeros([self.mmax + 1]).long().tolist()
        offset = 0
        for m in range(self.mmax + 1):
            idx_r, idx_i = self.complex_idx(m, -1, m_complex, l_harmonic)
            for idx_out, idx_in in enumerate(idx_r):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            self.m_size[m] = len(idx_r)
            for idx_out, idx_in in enumerate(idx_i):
                to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)
        to_m = to_m.detach()
        self.register_buffer("l_harmonic", l_harmonic, persistent=False)
        self.register_buffer("m_harmonic", m_harmonic, persistent=False)
        self.register_buffer("m_complex", m_complex, persistent=False)
        self.register_buffer("to_m", to_m, persistent=False)
        self.pre_compute_coefficient_idx()

    def complex_idx(self, m, lmax, m_complex, l_harmonic):
        """
        Add `m_complex` and `l_harmonic` to the input arguments
        since we cannot use `self.m_complex`.
        """
        if lmax == -1:
            lmax = self.lmax
        indices = paddle.arange(len(l_harmonic))
        mask_r = paddle.bitwise_and(x=(l_harmonic <= lmax), y=(m_complex == m))
        mask_idx_r = paddle.masked_select(indices, mask_r)
        mask_idx_i = paddle.tensor([]).long()
        if m != 0:
            mask_i = paddle.bitwise_and(x=(l_harmonic <= lmax), y=(m_complex == -m))
            mask_idx_i = paddle.masked_select(indices, mask_i)
        return mask_idx_r, mask_idx_i

    def pre_compute_coefficient_idx(self):
        """
        Pre-compute the results of `coefficient_idx()` and access them with `prepare_coefficient_idx()`
        """
        lmax = self.lmax
        for l in range(lmax + 1):
            for m in range(lmax + 1):
                mask = paddle.bitwise_and(
                    x=(self.l_harmonic <= l), y=(self.m_harmonic <= m)
                )
                indices = paddle.arange(len(mask))
                mask_indices = paddle.masked_select(indices, mask)
                self.register_buffer(
                    f"coefficient_idx_l{l}_m{m}", mask_indices, persistent=False
                )

    def prepare_coefficient_idx(self):
        """
        Construct a list of buffers
        """
        lmax = self.lmax
        coefficient_idx_list = []
        for l in range(lmax + 1):
            l_list = []
            for m in range(lmax + 1):
                l_list.append(getattr(self, f"coefficient_idx_l{l}_m{m}", None))
            coefficient_idx_list.append(l_list)
        return coefficient_idx_list

    def coefficient_idx(self, lmax: int, mmax: int):
        if lmax > self.lmax or mmax > self.lmax:
            mask = paddle.bitwise_and(
                x=(self.l_harmonic <= lmax), y=(self.m_harmonic <= mmax)
            )
            indices = paddle.arange(len(mask), device=mask.device)
            return paddle.masked_select(indices, mask)
        else:
            temp = self.prepare_coefficient_idx()
            return temp[lmax][mmax]

    def pre_compute_rotate_inv_rescale(self):
        lmax = self.lmax
        for l in range(lmax + 1):
            for m in range(lmax + 1):
                mask_indices = self.coefficient_idx(l, m)
                rotate_inv_rescale = paddle.ones(
                    (1, int((l + 1) ** 2), int((l + 1) ** 2))
                )
                for l_sub in range(l + 1):
                    if l_sub <= m:
                        continue
                    start_idx = l_sub**2
                    length = 2 * l_sub + 1
                    rescale_factor = math.sqrt(length / (2 * m + 1))
                    rotate_inv_rescale[
                        :,
                        start_idx : start_idx + length,
                        start_idx : start_idx + length,
                    ] = rescale_factor
                rotate_inv_rescale = rotate_inv_rescale[:, :, mask_indices]
                self.register_buffer(
                    f"rotate_inv_rescale_l{l}_m{m}",
                    rotate_inv_rescale,
                    persistent=False,
                )

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, mmax={self.mmax})"


class SO3_Grid(paddle.nn.Module):
    """
    Helper functions for grid representation of the irreps

    Args:
        lmax (int):   Maximum degree of the spherical harmonics
        mmax (int):   Maximum order of the spherical harmonics
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        normalization: str = "integral",
        resolution: (int | None) = None,
        rescale: bool = True,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * self.mmax + 1
        if resolution is not None:
            self.lat_resolution = resolution
            self.long_resolution = resolution
        self.mapping = CoefficientMapping(self.lmax, self.lmax)
        self.rescale = rescale
        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,
        )
        to_grid_shb = _as_paddle_tensor(to_grid.shb)
        to_grid_sha = _as_paddle_tensor(to_grid.sha)
        to_grid_mat = paddle.einsum("mbi, am -> bai", to_grid_shb, to_grid_sha).detach()
        if rescale and lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : start_idx + length] = (
                    to_grid_mat[:, :, start_idx : start_idx + length] * rescale_factor
                )
        to_grid_mat = to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]
        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,
        )
        from_grid_sha = _as_paddle_tensor(from_grid.sha)
        from_grid_shb = _as_paddle_tensor(from_grid.shb)
        from_grid_mat = paddle.einsum(
            "am, mbi -> bai", from_grid_sha, from_grid_shb
        ).detach()
        if rescale and lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval**2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : start_idx + length] = (
                    from_grid_mat[:, :, start_idx : start_idx + length] * rescale_factor
                )
        from_grid_mat = from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]
        self.register_buffer("to_grid_mat", to_grid_mat, persistent=False)
        self.register_buffer("from_grid_mat", from_grid_mat, persistent=False)

    def get_to_grid_mat(self, device=None):
        return self.to_grid_mat

    def get_from_grid_mat(self, device=None):
        return self.from_grid_mat

    def to_grid(self, embedding, lmax: int, mmax: int):
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        return paddle.einsum("bai, zic -> zbac", to_grid_mat, embedding)

    def from_grid(self, grid, lmax: int, mmax: int):
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        return paddle.einsum("bai, zbac -> zic", from_grid_mat, grid)
