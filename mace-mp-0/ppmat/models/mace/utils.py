import paddle
import numpy as np


def atomic_number_to_index(atomic_numbers):
    """Convert atomic numbers to indices"""
    unique_atoms = sorted(list(set(atomic_numbers)))
    atom_to_idx = {atom: i for i, atom in enumerate(unique_atoms)}
    return [atom_to_idx[atom] for atom in atomic_numbers]


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
