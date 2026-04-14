"""
Simplified molecular featurization for wD-MPNN.

Provides MolGraph and BatchMolGraph classes for building molecular graph
representations suitable for message passing neural networks. Designed to
work without RDKit dependency by accepting pre-computed features.

Ported from: https://github.com/Ramprasad-Group/polymer-chemprop
"""

from typing import List, Optional, Tuple

import numpy as np
import paddle

ATOM_FDIM = 133
BOND_FDIM = 14


def index_select_ND(source: paddle.Tensor, index: paddle.Tensor) -> paddle.Tensor:
    """
    Select entries from source along dim=0 using a 2-D index tensor.

    Args:
        source: Tensor of shape (N, hidden_size).
        index:  Tensor of shape (M, max_neighbors) with integer indices into source.

    Returns:
        Tensor of shape (M, max_neighbors, hidden_size).
    """
    index_shape = index.shape  # (M, max_neighbors)
    suffix_dim = source.shape[1:]  # (hidden_size,) or similar
    final_shape = list(index_shape) + list(suffix_dim)

    flat_index = index.reshape([-1])  # (M * max_neighbors,)
    target = paddle.index_select(source, flat_index, axis=0)
    target = target.reshape(final_shape)
    return target


class MolGraph:
    """
    Molecular graph representation for a single molecule.

    Stores atom features, bond features, adjacency structures, and
    optional weight vectors for polymer-aware message passing.
    """

    def __init__(
        self,
        f_atoms: np.ndarray,
        f_bonds: np.ndarray,
        a2b: List[List[int]],
        b2a: np.ndarray,
        b2revb: np.ndarray,
        w_atoms: Optional[np.ndarray] = None,
        w_bonds: Optional[np.ndarray] = None,
        degree_of_polym: float = 1.0,
    ):
        """
        Args:
            f_atoms: Atom feature matrix of shape (n_atoms, atom_fdim).
            f_bonds: Bond feature matrix of shape (n_bonds, bond_fdim).
            a2b: List of lists mapping each atom to its incident bond indices.
            b2a: Array mapping each bond to its source atom.
            b2revb: Array mapping each bond to its reverse bond.
            w_atoms: Per-atom weights (default: all ones).
            w_bonds: Per-bond weights (default: all ones).
            degree_of_polym: Degree of polymerization multiplier.
        """
        self.n_atoms = f_atoms.shape[0]
        self.n_bonds = f_bonds.shape[0]
        self.f_atoms = f_atoms
        self.f_bonds = f_bonds
        self.a2b = a2b
        self.b2a = b2a
        self.b2revb = b2revb
        self.w_atoms = w_atoms if w_atoms is not None else np.ones(self.n_atoms, dtype=np.float32)
        self.w_bonds = w_bonds if w_bonds is not None else np.ones(self.n_bonds, dtype=np.float32)
        self.degree_of_polym = degree_of_polym


class BatchMolGraph:
    """
    Batched molecular graph that merges multiple MolGraph instances.

    Handles padding of adjacency lists and offset shifting so that the
    message passing encoder can process an entire batch in one forward call.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        self.atom_fdim = mol_graphs[0].f_atoms.shape[1]
        self.bond_fdim = mol_graphs[0].f_bonds.shape[1]
        self.n_mols = len(mol_graphs)

        # Running offsets
        n_atoms = 1  # leave index 0 as padding atom
        n_bonds = 1  # leave index 0 as padding bond

        f_atoms = [np.zeros((1, self.atom_fdim), dtype=np.float32)]  # padding row
        f_bonds = [np.zeros((1, self.bond_fdim), dtype=np.float32)]  # padding row
        w_atoms = [np.zeros(1, dtype=np.float32)]  # padding
        w_bonds = [np.zeros(1, dtype=np.float32)]  # padding
        a2b_all: List[List[int]] = [[]]  # padding atom's neighbor list
        b2a = [0]
        b2revb = [0]
        a_scope = []
        b_scope = []
        degree_of_polym = []

        for mg in mol_graphs:
            a_scope.append((n_atoms, mg.n_atoms))
            b_scope.append((n_bonds, mg.n_bonds))

            f_atoms.append(mg.f_atoms)
            f_bonds.append(mg.f_bonds)
            w_atoms.append(mg.w_atoms)
            w_bonds.append(mg.w_bonds)

            for atom_a2b in mg.a2b:
                a2b_all.append([b + n_bonds for b in atom_a2b])

            b2a.extend(mg.b2a + n_atoms)
            b2revb.extend(mg.b2revb + n_bonds)

            degree_of_polym.append(mg.degree_of_polym)

            n_atoms += mg.n_atoms
            n_bonds += mg.n_bonds

        self.f_atoms = paddle.to_tensor(np.concatenate(f_atoms, axis=0), dtype="float32")
        self.f_bonds = paddle.to_tensor(np.concatenate(f_bonds, axis=0), dtype="float32")
        self.w_atoms = paddle.to_tensor(np.concatenate(w_atoms, axis=0), dtype="float32")
        self.w_bonds = paddle.to_tensor(np.concatenate(w_bonds, axis=0), dtype="float32")
        self.b2a = paddle.to_tensor(np.array(b2a, dtype=np.int64))
        self.b2revb = paddle.to_tensor(np.array(b2revb, dtype=np.int64))
        self.a_scope = a_scope
        self.b_scope = b_scope
        self.degree_of_polym = degree_of_polym

        # Pad a2b to rectangular tensor
        max_num_bonds = max(len(bonds) for bonds in a2b_all) if a2b_all else 1
        max_num_bonds = max(max_num_bonds, 1)
        a2b_padded = np.zeros((n_atoms, max_num_bonds), dtype=np.int64)
        for i, bonds in enumerate(a2b_all):
            for j, b in enumerate(bonds):
                a2b_padded[i, j] = b
        self.a2b = paddle.to_tensor(a2b_padded)

    def get_components(self):
        """
        Return all graph components needed by MPNEncoder.

        Returns:
            Tuple of (f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb,
                      a_scope, b_scope, degree_of_polym).
        """
        return (
            self.f_atoms,
            self.f_bonds,
            self.w_atoms,
            self.w_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
            self.degree_of_polym,
        )
