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

from .radial import RadialMLP


class EdgeDegreeEmbedding(paddle.nn.Layer):
    """Initialize equivariant node features from radial edge features."""

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        mmax: int,
        edge_channels_list: list[int],
        rescale_factor: float,
        mapping,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.m0_components = mapping.m_size[0]
        channels = [
            *edge_channels_list,
            self.m0_components * sphere_channels,
        ]
        self.radial = RadialMLP(channels)
        self.rescale_factor = rescale_factor

    def forward(
        self,
        x: paddle.Tensor,
        edge_features: paddle.Tensor,
        edge_index: paddle.Tensor,
        wigner_inv: paddle.Tensor,
    ) -> paddle.Tensor:
        radial = self.radial(edge_features).reshape(
            [-1, self.m0_components, self.sphere_channels]
        )
        edge_embedding = paddle.bmm(
            wigner_inv[:, :, : self.m0_components], radial
        ).astype(x.dtype)
        return x.index_add(
            axis=0,
            index=edge_index[1],
            value=edge_embedding / self.rescale_factor,
        )
