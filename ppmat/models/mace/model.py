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
        atom_indices = atomic_number_to_index(atomic_numbers)
        atom_indices = paddle.to_tensor(atom_indices, dtype='int64')

        # Get edge vectors and distances
        edge_vectors, edge_distances = get_edge_vectors(positions, cell, pbc)

        # Create edge indices
        num_atoms = positions.shape[0]
        edge_indices = paddle.meshgrid(paddle.arange(num_atoms), paddle.arange(num_atoms))
        edge_indices = paddle.stack(edge_indices, axis=0).reshape((2, -1))

        # Remove self-edges
        mask = edge_indices[0] != edge_indices[1]
        edge_indices = edge_indices[:, mask]
        edge_distances = edge_distances.reshape((-1,))[mask]

        # Filter edges by distance
        distance_mask = edge_distances < self.r_max
        edge_indices = edge_indices[:, distance_mask]
        edge_distances = edge_distances[distance_mask]

        # Initialize node features
        x = self.embedding(atom_indices)

        # Pass through equivariant layers
        for layer in self.layers:
            x = layer(x, edge_indices, edge_distances)

        # Aggregate node features
        atomic_energies = self.output(x)
        total_energy = atomic_energies.sum()

        # Compute forces by differentiating energy with respect to positions
        forces = -paddle.grad(total_energy, positions, create_graph=compute_stress)[0]

        if compute_stress and cell is not None:
            # Compute stress tensor by differentiating energy with respect to cell
            # stress_ij = (1/V) * sum_k(dE/dcell_ik * cell_jk)
            volume = paddle.abs(paddle.linalg.det(cell))
            dE_dcell = paddle.grad(total_energy, cell, create_graph=False)[0]
            
            # Compute stress tensor
            stress = paddle.zeros([3, 3])
            for i in range(3):
                for j in range(3):
                    stress[i, j] = paddle.sum(dE_dcell[i, :] * cell[j, :]) / volume
            
            # Convert to Voigt notation: [xx, yy, zz, yz, xz, xy]
            stress_voigt = paddle.stack([
                stress[0, 0],
                stress[1, 1],
                stress[2, 2],
                (stress[1, 2] + stress[2, 1]) / 2.0,
                (stress[0, 2] + stress[2, 0]) / 2.0,
                (stress[0, 1] + stress[1, 0]) / 2.0
            ])
            
            return total_energy, forces, stress_voigt

        return total_energy, forces

    def predict(self, atomic_numbers, positions, cell=None, pbc=False, compute_stress=False):
        """Predict energy, forces and optionally stress.
        
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
        return self.forward(atomic_numbers, positions, cell, pbc, compute_stress)
