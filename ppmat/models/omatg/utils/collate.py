# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OMatG data collation utilities.

DefaultCollator cannot serve OMatG for two reasons:
1. Variable-length sequences: different structures have different atom counts,
   so paddle.stack (used by DefaultCollator) fails; this collator uses
   paddle.concat instead.
2. The Structure class is not a type recognized by DefaultCollator's
   type-dispatched handler chain (only tensor/ndarray/dict/list/pgl.Graph).

This collator additionally builds the node2graph mapping required by the
model's flattened batch layout.
"""

import paddle
from typing import Dict, List


class OMATGStructureCollator:
    """Collator for OMatG Structure objects.

    Converts Structure objects to the format expected by OMatG models:
    - atom_types: Concatenated tensor of atomic numbers (flattened across batch)
    - frac_coords: Concatenated tensor of fractional coordinates (flattened across batch)
    - lattices: Stacked tensor of lattice vectors
    - num_atoms: Tensor of atom counts per structure
    - node2graph: Node to graph mapping for batched operations (flattened)

    Args:
        padding_value: Value to use for padding variable-length tensors (not used in flattened format)
    """

    def __init__(self, padding_value: int = 0):
        """Initialize OMATGStructureCollator.

        Args:
            padding_value: Value to use for padding variable-length tensors (not used in flattened format)
        """
        self.padding_value = padding_value

    def __call__(self, batch: list) -> dict:
        """Collate a batch of Structure objects.

        Args:
            batch: List of Structure objects

        Returns:
            Dictionary containing collated batch data in flattened format
        """
        # Extract data from each structure
        lattices = []
        atomic_numbers_list = []
        frac_coords_list = []
        num_atoms_list = []

        for structure in batch:
            # Ensure positions are in fractional coordinates
            if not structure.pos_is_fractional:
                structure.convert_to_fractional()

            lattices.append(structure.cell)
            atomic_numbers_list.append(structure.atomic_numbers)
            frac_coords_list.append(structure.pos)
            num_atoms_list.append(len(structure.atomic_numbers))

        # Stack lattices (all are 3x3)
        lattices = paddle.stack(lattices, axis=0)  # [batch_size, 3, 3]

        # Concatenate variable-length sequences (flatten across batch)
        atom_types = paddle.concat(atomic_numbers_list, axis=0)  # [total_atoms]
        frac_coords = paddle.concat(frac_coords_list, axis=0)  # [total_atoms, 3]

        # Create node2graph mapping (which node belongs to which graph)
        node2graph = []
        for i, num_atoms in enumerate(num_atoms_list):
            node2graph.extend([i] * num_atoms)

        node2graph = paddle.to_tensor(node2graph, dtype="int64")  # [total_atoms]

        # Convert num_atoms to tensor
        num_atoms = paddle.to_tensor(num_atoms_list, dtype="int64")  # [batch_size]

        return {
            "atom_types": atom_types,
            "frac_coords": frac_coords,
            "lattices": lattices,
            "num_atoms": num_atoms,
            "node2graph": node2graph,
        }
