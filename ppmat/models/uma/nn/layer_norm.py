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

1. Normalize features of shape (N, sphere_basis, C),
with sphere_basis = (lmax + 1) ** 2.

2. The difference from `layer_norm.py` is that all type-L vectors have
the same number of channels and input features are of shape (N, sphere_basis, C).
"""

import math
from typing import Literal


def get_normalization_layer(
    norm_type: Literal["layer_norm", "layer_norm_sh", "rms_norm_sh"],
    lmax: int,
    num_channels: int,
    eps: float = 1e-05,
    affine: bool = True,
    normalization: str = "component",
):
    assert norm_type in ["layer_norm", "layer_norm_sh", "rms_norm_sh"]
    if norm_type == "layer_norm":
        norm_class = EquivariantLayerNormArray
    elif norm_type == "layer_norm_sh":
        norm_class = EquivariantLayerNormArraySphericalHarmonics
    elif norm_type == "rms_norm_sh":
        norm_class = EquivariantRMSNormArraySphericalHarmonicsV2
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)


def get_l_to_all_m_expand_index(lmax: int):
    expand_index = paddle.zeros([(lmax + 1) ** 2]).long()
    for lval in range(lmax + 1):
        start_idx = lval**2
        length = 2 * lval + 1
        expand_index[start_idx : start_idx + length] = lval
    return expand_index


class EquivariantLayerNormArray(paddle.nn.Module):
    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        normalization: str = "component",
    ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = paddle.nn.Parameter(
                paddle.ones(lmax + 1, num_channels)
            )
            self.affine_bias = paddle.nn.Parameter(paddle.zeros(num_channels))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    @paddle.autocast("cuda", enabled=False)
    @paddle.autocast("cpu", enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out = []
        for lval in range(self.lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            feature = node_input.narrow(1, start_idx, length)
            if lval == 0:
                feature_mean = paddle.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
            elif self.normalization == "component":
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
            feature_norm = paddle.mean(feature_norm, dim=2, keepdim=True)
            feature_norm = (feature_norm + self.eps).pow(-0.5)
            if self.affine:
                weight = self.affine_weight.narrow(0, lval, 1)
                weight = weight.view(1, 1, -1)
                feature_norm = feature_norm * weight
            feature = feature * feature_norm
            if self.affine and lval == 0:
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias
            out.append(feature)
        return paddle.cat(out, dim=1)


class EquivariantLayerNormArraySphericalHarmonics(paddle.nn.Module):
    """
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        normalization: str = "component",
        std_balance_degrees: bool = True,
    ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        self.norm_l0 = paddle.nn.LayerNorm(
            self.num_channels, eps=self.eps, elementwise_affine=self.affine
        )
        if self.affine:
            self.affine_weight = paddle.nn.Parameter(
                paddle.ones(self.lmax, self.num_channels)
            )
        else:
            self.register_parameter("affine_weight", None)
        assert normalization in ["norm", "component"]
        self.normalization = normalization
        if self.std_balance_degrees:
            balance_degree_weight = paddle.zeros((self.lmax + 1) ** 2 - 1, 1)
            for lval in range(1, self.lmax + 1):
                start_idx = lval**2 - 1
                length = 2 * lval + 1
                balance_degree_weight[start_idx : start_idx + length, :] = 1.0 / length
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer(
                "balance_degree_weight", balance_degree_weight, persistent=False
            )
        else:
            self.balance_degree_weight = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    @paddle.autocast("cuda", enabled=False)
    @paddle.autocast("cpu", enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out = []
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    feature_norm = feature.pow(2)
                    feature_norm = paddle.einsum(
                        "nic, ia -> nac", feature_norm, self.balance_degree_weight
                    )
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
            feature_norm = paddle.mean(feature_norm, dim=2, keepdim=True)
            feature_norm = (feature_norm + self.eps).pow(-0.5)
            for lval in range(1, self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                feature = node_input.narrow(1, start_idx, length)
                if self.affine:
                    weight = self.affine_weight.narrow(0, lval - 1, 1)
                    weight = weight.view(1, 1, -1)
                    feature_scale = feature_norm * weight
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)
        return paddle.cat(out, dim=1)


class EquivariantRMSNormArraySphericalHarmonics(paddle.nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        normalization: str = "component",
    ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = paddle.nn.Parameter(
                paddle.ones(self.lmax + 1, self.num_channels)
            )
        else:
            self.register_parameter("affine_weight", None)
        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    @paddle.autocast("cuda", enabled=False)
    @paddle.autocast("cpu", enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        out = []
        feature = node_input
        if self.normalization == "norm":
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
        elif self.normalization == "component":
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
        feature_norm = paddle.mean(feature_norm, dim=2, keepdim=True)
        feature_norm = (feature_norm + self.eps).pow(-0.5)
        for lval in range(self.lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            feature = node_input.narrow(1, start_idx, length)
            if self.affine:
                weight = self.affine_weight.narrow(0, lval, 1)
                weight = weight.view(1, 1, -1)
                feature_scale = feature_norm * weight
            else:
                feature_scale = feature_norm
            feature = feature * feature_scale
            out.append(feature)
        return paddle.cat(out, dim=1)


class EquivariantRMSNormArraySphericalHarmonicsV2(paddle.nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    def __init__(
        self,
        lmax: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        normalization: str = "component",
        centering: bool = True,
        std_balance_degrees: bool = True,
    ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees
        if self.affine:
            self.affine_weight = paddle.nn.Parameter(
                paddle.ones(self.lmax + 1, self.num_channels)
            )
            if self.centering:
                self.affine_bias = paddle.nn.Parameter(paddle.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        assert normalization in ["norm", "component"]
        self.normalization = normalization
        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index, persistent=False)
        if self.std_balance_degrees:
            balance_degree_weight = paddle.zeros((self.lmax + 1) ** 2, 1)
            for lval in range(self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                balance_degree_weight[start_idx : start_idx + length, :] = 1.0 / length
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer(
                "balance_degree_weight", balance_degree_weight, persistent=False
            )
        else:
            self.balance_degree_weight = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"

    @paddle.autocast("cuda", enabled=False)
    @paddle.autocast("cpu", enabled=False)
    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """
        feature = node_input
        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)
            feature_l0 = feature_l0 - feature_l0_mean
            feature = paddle.cat(
                (feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1
            )
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)
                feature_norm = paddle.einsum(
                    "nic, ia -> nac", feature_norm, self.balance_degree_weight
                )
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)
        feature_norm = paddle.mean(feature_norm, dim=2, keepdim=True)
        feature_norm = (feature_norm + self.eps).pow(-0.5)
        if self.affine:
            weight = self.affine_weight.view(1, self.lmax + 1, self.num_channels)
            weight = paddle.index_select(weight, dim=1, index=self.expand_index)
            feature_norm = feature_norm * weight
        out = feature * feature_norm
        if self.affine and self.centering:
            bias = self.affine_bias.view(1, 1, self.num_channels)
            out0 = out.narrow(1, 0, 1) + bias
            if out.shape[1] > 1:
                out = paddle.concat([out0, out[:, 1:, :]], axis=1)
            else:
                out = out0
        return out


class EquivariantDegreeLayerScale(paddle.nn.Module):
    """
    1. Similar to Layer Scale used in CaiT (Going Deeper With Image Transformers (ICCV'21)), we scale the output of both attention and FFN.
    2. For degree L > 0, we scale down the square root of 2 * L, which is to emulate halving the number of channels when using higher L.
    """

    def __init__(self, lmax: int, num_channels: int, scale_factor: float = 2.0) -> None:
        super().__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        self.affine_weight = paddle.nn.Parameter(
            paddle.ones(1, self.lmax + 1, self.num_channels)
        )
        for lval in range(1, self.lmax + 1):
            self.affine_weight.data[0, lval, :].mul_(
                1.0 / math.sqrt(self.scale_factor * lval)
            )
        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index, persistent=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, scale_factor={self.scale_factor})"

    def forward(self, node_input):
        weight = paddle.index_select(self.affine_weight, dim=1, index=self.expand_index)
        return node_input * weight
