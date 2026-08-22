# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np


def atomic_number_to_index(atomic_numbers):
    """Convert supported MACE atomic numbers to fixed embedding indices."""
    supported = list(range(1, 84)) + list(range(89, 96))
    atom_to_idx = {atomic_number: index for index, atomic_number in enumerate(supported)}
    return [atom_to_idx[int(atomic_number)] for atomic_number in atomic_numbers]


def radial_basis(r, r_max, num_basis):
    """Radial basis function"""
    r = paddle.clip(r, 0, r_max)
    scaled_r = 2 * r / r_max - 1
    basis = []
    for n in range(num_basis):
        basis.append(paddle.cos(n * np.pi * scaled_r))
    return paddle.stack(basis, axis=-1)


def get_edge_vectors(positions, cell=None, pbc=False):
    """Compute edge vectors between atoms"""
    num_atoms = positions.shape[0]
    # Compute pairwise differences
    pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    # Reshape to (num_atoms, num_atoms, 3)
    edge_vectors = pos_diff.reshape((num_atoms, num_atoms, 3))
    # Compute distances
    edge_distances = paddle.norm(edge_vectors, axis=-1)
    return edge_vectors, edge_distances
