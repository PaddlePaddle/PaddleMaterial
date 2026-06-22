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
import paddle.nn as nn
import paddle.nn.functional as F
from .layers import EquivariantLayer
from .utils import get_edge_vectors, atomic_number_to_index


class MACE(nn.Layer):
    """MACE model implementation"""

    def __init__(self,
                 hidden_dim=128,
                 num_layers=3,
                 num_basis=8,
                 r_max=5.0,
                 num_elements=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_basis = num_basis
        self.r_max = r_max
        self.num_elements = num_elements

        # Embedding layer for atomic numbers
        self.embedding = nn.Embedding(num_elements, hidden_dim)

        # Equivariant layers
        self.layers = nn.LayerList()
        for _ in range(num_layers):
            self.layers.append(EquivariantLayer(hidden_dim, num_basis, r_max))

        # Output layer for energy prediction
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, atomic_numbers, positions, cell=None, pbc=False, compute_stress=False):
        """Forward pass of MACE model.
        
        Args:
            atomic_numbers: Atomic numbers of atoms
            positions: Atomic positions, shape [num_atoms, 3]
            cell: Unit cell matrix, shape [3, 3], optional
            pbc: Periodic boundary conditions, default False
            compute_stress: Whether to compute stress tensor, default False
            
        Returns:
            total_energy: Total energy of the system
            forces: Forces on atoms, shape [num_atoms, 3]
            stress: Stress tensor (Voigt notation), shape [6], optional
        """
        # Convert atomic numbers to indices
        indices = paddle.to_tensor(atomic_number_to_index(atomic_numbers), dtype='int64')
        
        # Get initial embeddings
        x = self.embedding(indices)
        
        # Compute edge vectors and distances
        edge_vectors, edge_distances = get_edge_vectors(positions, cell, pbc)
        
        # Filter edges within cutoff radius
        mask = edge_distances < self.r_max
        edge_indices = paddle.where(mask)
        edge_distances = edge_distances[mask]
        edge_vectors = edge_vectors[mask]
        
        # Forward pass through equivariant layers
        for layer in self.layers:
            x = layer(x, edge_indices, edge_distances)
        
        # Predict atomic energies
        atomic_energies = self.output(x)
        
        # Total energy
        total_energy = paddle.sum(atomic_energies)
        
        # Compute forces (negative gradient of energy w.r.t. positions)
        positions.stop_gradient = False
        forces = -paddle.grad(total_energy, positions, create_graph=True)[0]
        
        # Compute stress if requested
        stress = None
        if compute_stress and cell is not None:
            # Stress computation using energy-strain relation
            stress = paddle.grad(total_energy, cell, create_graph=True)[0]
            # Convert to Voigt notation
            stress = paddle.stack([
                stress[0, 0], stress[1, 1], stress[2, 2],
                stress[1, 2], stress[0, 2], stress[0, 1]
            ])
        
        return total_energy, forces, stress
