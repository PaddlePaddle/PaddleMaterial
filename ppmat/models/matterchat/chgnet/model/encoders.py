# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import Tensor, nn

from ppmat.models.matterchat.chgnet.model.basis import Fourier, RadialBessel


class AtomEmbedding(nn.Layer):
    """Encode an atom by its atomic number."""

    def __init__(self, atom_feature_dim: int, max_num_elements: int = 94) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_num_elements, atom_feature_dim)

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        return self.embedding(atomic_numbers)


class BondEncoder(nn.Layer):
    """Encode bonds using Gaussian distance expansion."""

    def __init__(
        self,
        atom_graph_cutoff: float = 5,
        bond_graph_cutoff: float = 3,
        num_radial: int = 9,
        cutoff_coeff: int = 5,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.rbf_expansion_ag = RadialBessel(
            num_radial=num_radial,
            cutoff=atom_graph_cutoff,
            smooth_cutoff=cutoff_coeff,
            learnable=learnable,
        )
        self.rbf_expansion_bg = RadialBessel(
            num_radial=num_radial,
            cutoff=bond_graph_cutoff,
            smooth_cutoff=cutoff_coeff,
            learnable=learnable,
        )

    def forward(
        self,
        center: Tensor,
        neighbor: Tensor,
        undirected2directed: Tensor,
        image: Tensor,
        lattice: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        neighbor = neighbor + image @ lattice
        bond_vectors = center - neighbor
        bond_lengths = paddle.norm(bond_vectors, axis=1)
        bond_vectors = bond_vectors / bond_lengths[:, None]

        undirected_bond_lengths = paddle.index_select(
            bond_lengths, 0, undirected2directed
        )
        bond_basis_ag = self.rbf_expansion_ag(undirected_bond_lengths)
        bond_basis_bg = self.rbf_expansion_bg(undirected_bond_lengths)
        return bond_basis_ag, bond_basis_bg, bond_vectors


class AngleEncoder(nn.Layer):
    """Encode angles using Fourier expansion."""

    def __init__(self, num_angular: int = 9, learnable: bool = True) -> None:
        super().__init__()
        if num_angular % 2 != 1:
            raise ValueError(f"{num_angular=} must be an odd integer")
        circular_harmonics_order = (num_angular - 1) // 2
        self.fourier_expansion = Fourier(
            order=circular_harmonics_order, learnable=learnable
        )

    def forward(self, bond_i: Tensor, bond_j: Tensor) -> Tensor:
        # 1 - 1e-6 for acos stability
        cosine_ij = paddle.sum(bond_i * bond_j, axis=1) * (1 - 1e-6)
        angle = paddle.acos(cosine_ij)
        return self.fourier_expansion(angle)
