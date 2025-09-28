# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import pgl
from p_tqdm import p_map
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

from ppmat.utils import logger
from ppmat.utils.crystal import lattice_params_to_matrix


class FindPointsInSpheres:
    """Convert crystal structure to graph representation using spherical neighborhood
    search.

    This tool identifies neighboring atoms within a cutoff radius for each atom in a
    crystal structure, building a graph representation suitable for material analysis
    applications.

    Args:
        cutoff (float, optional): Cutoff radius (in Ångström) for neighborhood search.
            Defaults to 5.0.
        pbc (tuple[int, int, int], optional):Periodic boundary conditions along x/y/z
            axes. Each element 0 (disabled) or 1 (enabled). Defaults to (1, 1, 1).
        num_cpus (Optional[int], optional): Number of CPU cores for parallel processing:
            - None: Auto-detect all available cores (recommended)
            - Positive integer: Explicit core count.
            Defaults to None.
        eps (float, optional): Floating-point tolerance for numerical comparisons.
            Defaults to 1e-8.
        **kwargs: Reserved for future expansion (currently unused parameters)
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        pbc: tuple[int, int, int] = (1, 1, 1),
        num_cpus: Optional[int] = None,
        eps: float = 1e-8,
        **kwargs,
    ) -> None:
        self.cutoff = cutoff
        self.pbc = np.array(pbc, dtype=int)
        self.num_cpus = num_cpus
        self.eps = eps

    def __call__(self, structure: Structure):
        if isinstance(structure, Structure):
            graph = self.get_graph_by_find_points_in_spheres(structure)
        elif isinstance(structure, list):
            graph = p_map(
                self.get_graph_by_find_points_in_spheres,
                structure,
                num_cpus=self.num_cpus,
            )
            # the following code is equivalent to the above line, it is slower,
            # but easier to debug.
            # graph = [
            # self.get_graph_by_find_points_in_spheres(struc)
            #     for struc in structure
            # ]
        else:
            raise TypeError("The input must be a pymatgen.Structure or a list of them.")
        return graph

    def get_graph_by_find_points_in_spheres(self, structure: Structure):
        lattice_matrix = structure.lattice.matrix
        cart_coords = structure.cart_coords

        cutoff = self.cutoff
        attempt = 3
        while attempt > 0:
            src_id, dst_id, images, bond_dist = find_points_in_spheres(
                cart_coords,
                cart_coords,
                r=cutoff,
                pbc=self.pbc,
                lattice=lattice_matrix,
                tol=self.eps,
            )
            exclude_self = (src_id != dst_id) | (bond_dist > self.eps)
            src_id, dst_id, images, bond_dist = (
                src_id[exclude_self],
                dst_id[exclude_self],
                images[exclude_self],
                bond_dist[exclude_self],
            )

            edge_indices = [(u, v) for u, v in zip(src_id, dst_id)]
            to_jimages = np.array(images, dtype="float32")
            if len(edge_indices) == 0:
                logger.warning(
                    f"No edges found within cutoff {cutoff:.5f}. Trying again with "
                    "larger cutoff."
                )
                cutoff *= 2
                attempt -= 1
            else:
                break
        if len(edge_indices) == 0:
            logger.warning(
                f"No edges found within cutoff {cutoff:.5f}. Set graph is None."
            )
            graph = None
        else:
            graph = self.build_pgl_graph(structure, edge_indices, to_jimages)
        return graph

    def build_pgl_graph(
        self,
        structure: Structure,
        edge_indices,
        to_jimages,
        node_features=None,
        edge_features=None,
    ):
        assert node_features is None or isinstance(node_features, dict)
        assert edge_features is None or isinstance(edge_features, dict)

        # get atom types
        atom_types = np.array([site.specie.Z for site in structure])

        # get lattice parameters and matrix
        lattice_parameters = structure.lattice.parameters
        lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
        angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
        lattice = structure.lattice.matrix.astype("float32")

        # convert to numpy array
        edge_indices = np.array(edge_indices)
        if to_jimages is not None:
            to_jimages = np.array(to_jimages)
        num_atoms = tuple(atom_types.shape)[0]

        # After multiple graph batch operations by the dataloader,
        # graph.num_nodes remains an integer, which is the sum of the number of
        # nodes in all graphs
        graph = pgl.Graph(edge_indices, num_nodes=num_atoms)
        # node features: frac_coords, cart_coords, atom_types
        graph.node_feat["frac_coords"] = structure.frac_coords.astype("float32")
        graph.node_feat["cart_coords"] = structure.cart_coords.astype("float32")
        graph.node_feat["atom_types"] = atom_types

        # graph features: lengths, angles, lattice, num_atoms
        # Due to the inability of pgl.graph to store graph level features,
        # we will store these features under node_feat
        graph.node_feat["lengths"] = lengths
        graph.node_feat["angles"] = angles
        graph.node_feat["lattice"] = lattice.reshape(1, 3, 3)
        # graph.node_feat['num_atoms'] is different from graph.num_nodes
        # After multiple graph batch operations by the dataloader,
        # graph.node_feat['num_atoms'] is a tensor of shape (batch_size),
        # where each value is the number of atoms in the corresponding graph.
        graph.node_feat["num_atoms"] = np.array([num_atoms])
        # edge features: pbc_offset, bond_vec, bond_dist
        if to_jimages is not None:
            graph.edge_feat["pbc_offset"] = to_jimages
            offset = np.matmul(to_jimages, lattice)
            dst_pos = graph.node_feat["cart_coords"][graph.edges[:, 1]] + offset
            src_pos = graph.node_feat["cart_coords"][graph.edges[:, 0]]
            bond_vec = dst_pos - src_pos
            bond_dist = np.linalg.norm(bond_vec, axis=1)
            graph.edge_feat["bond_vec"] = bond_vec.astype("float32")
            graph.edge_feat["bond_dist"] = bond_dist.astype("float32")
        graph.edge_feat["num_edges"] = np.array([edge_indices.shape[0]])

        if node_features is not None:
            graph.node_feat.update(node_features)
        if edge_features is not None:
            graph.edge_feat.update(edge_features)
        return graph


class CrystalNN:
    """Convert crystal structure to graph representation using
    CrystalNN-based graph generator method.

    This class uses pymatgen's CrystalNN local environment strategy to
    convert a pymatgen Structure into a PGL graph (pgl.Graph),
    capturing atomic connectivity under periodic boundary conditions.

    Core methods:
      - __init__: Configure neighbor search parameters and parallelism.
      - __call__:  Accepts a single Structure or a list and returns
                   a graph or list of graphs.
      - get_graph_by_crystalnn: Converts one Structure into a PGL graph.
      - build_pgl_graph: Assembles node/edge features into a PGL graph.

    Args:
        cutoff (float):
            Maximum neighbor search distance (in Å). Atom pairs farther
            apart than this are not considered bonded.
        pbc (tuple[int, int, int]):
            Periodic boundary flags along (a, b, c) axes.
            1 enables periodicity, 0 disables it.
        num_cpus (Optional[int]):
            Number of CPU cores to use when processing a list of structures
            in parallel. If None, processes sequentially.
        eps (float):
            Small constant for numerical stability (e.g., to avoid division
            by zero).
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        pbc: tuple[int, int, int] = (1, 1, 1),
        num_cpus: Optional[int] = None,
        eps: float = 1e-8,
    ):
        self.cutoff = cutoff
        self.pbc = np.array(pbc, dtype=int)
        self.num_cpus = num_cpus
        self.eps = eps
        self.CrystalNN = local_env.CrystalNN(
            distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
        )

    def __call__(self, structure: Structure):
        if isinstance(structure, Structure):
            graph = self.get_graph_by_crystalnn(structure)
        elif isinstance(structure, list):
            graph = p_map(
                self.get_graph_by_crystalnn, structure, num_cpus=self.num_cpus
            )
            # the following code is equivalent to the above line, it is slower,
            # but easier to debug.
            # graph = [self.get_graph_by_crystalnn(struc) for struc in structure]
        else:
            raise TypeError("The input must be a pymatgen.Structure or a list of them.")
        return graph

    def get_graph_by_crystalnn(self, structure: Structure):

        try:
            structure_graph = StructureGraph.with_local_env_strategy(
                structure, self.CrystalNN
            )
        except Exception:
            search_cutoff = 10
            while True:
                try:
                    crystalNN_tmp = local_env.CrystalNN(
                        distance_cutoffs=None,
                        x_diff_weight=-1,
                        porous_adjustment=False,
                        search_cutoff=search_cutoff,
                    )
                    structure_graph = StructureGraph.from_local_env_strategy(
                        structure, crystalNN_tmp
                    )
                    logger.info(
                        "Successfully generated graph by CrystalNN with "
                        f"search_cutoff={search_cutoff}."
                    )
                    break
                except Exception:
                    search_cutoff += 2
                    logger.info(f"Searching for new search_cutoff{search_cutoff}...")
                    if search_cutoff > 40:
                        logger.info(
                            "Failed to generate graph by CrystalNN with "
                            f"search_cutoff={search_cutoff}. "
                        )
                        break

        # atom_types = np.array(structure.atomic_numbers)
        lattice_parameters = structure.lattice.parameters
        lengths = lattice_parameters[:3]
        angles = lattice_parameters[3:]
        assert np.allclose(
            structure.lattice.matrix, lattice_params_to_matrix(*lengths, *angles)
        )

        edge_indices, to_jimages = [], []
        for i, j, to_jimage in structure_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

        graph = self.build_pgl_graph(structure, edge_indices, to_jimages)
        return graph

    def build_pgl_graph(
        self,
        structure: Structure,
        edge_indices,
        to_jimages,
        node_features=None,
        edge_features=None,
    ):
        assert node_features is None or isinstance(node_features, dict)
        assert edge_features is None or isinstance(edge_features, dict)

        # get atom types
        atom_types = np.array([site.specie.Z for site in structure])

        # get lattice parameters and matrix
        lattice_parameters = structure.lattice.parameters
        lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
        angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
        lattice = structure.lattice.matrix.astype("float32")

        # convert to numpy array
        edge_indices = np.array(edge_indices)
        if to_jimages is not None:
            to_jimages = np.array(to_jimages)
        num_atoms = tuple(atom_types.shape)[0]

        # After multiple graph batch operations by the dataloader,
        # graph.num_nodes remains an integer, which is the sum of the number of
        # nodes in all graphs
        graph = pgl.Graph(edge_indices, num_nodes=num_atoms)
        # node features: frac_coords, cart_coords, atom_types
        graph.node_feat["frac_coords"] = structure.frac_coords.astype("float32")
        graph.node_feat["cart_coords"] = structure.cart_coords.astype("float32")
        graph.node_feat["atom_types"] = atom_types

        # graph features: lengths, angles, lattice, num_atoms
        # Due to the inability of pgl.graph to store graph level features,
        # we will store these features under node_feat
        graph.node_feat["lengths"] = lengths
        graph.node_feat["angles"] = angles
        graph.node_feat["lattice"] = lattice.reshape(1, 3, 3)
        # graph.node_feat['num_atoms'] is different from graph.num_nodes
        # After multiple graph batch operations by the dataloader,
        # graph.node_feat['num_atoms'] is a tensor of shape (batch_size),
        # where each value is the number of atoms in the corresponding graph.
        graph.node_feat["num_atoms"] = np.array([num_atoms])
        # edge features: pbc_offset, bond_vec, bond_dist
        if to_jimages is not None:
            graph.edge_feat["pbc_offset"] = to_jimages
            offset = np.matmul(to_jimages, lattice)
            dst_pos = graph.node_feat["cart_coords"][graph.edges[:, 1]] + offset
            src_pos = graph.node_feat["cart_coords"][graph.edges[:, 0]]
            bond_vec = dst_pos - src_pos
            bond_dist = np.linalg.norm(bond_vec, axis=1)
            graph.edge_feat["bond_vec"] = bond_vec.astype("float32")
            graph.edge_feat["bond_dist"] = bond_dist.astype("float32")
        graph.edge_feat["num_edges"] = np.array([edge_indices.shape[0]])

        if node_features is not None:
            graph.node_feat.update(node_features)
        if edge_features is not None:
            graph.edge_feat.update(edge_features)
        return graph


class MolecularGraphConverter:
    """Convert RDKit Mol into PGL Graph.

    Args:
        remove_h (bool): Controls whether hydrogen atoms are removed before building
            the molecular graph. Defaults to False。
        atom_vocab (Optional[Dict[str,int]]): A dictionary mapping atomic symbols
            (e.g., "C", "O", "N") to unique integer indices for one-hot encoding.
        bond_vocab (Optional[Tuple[BT,...]]): A tuple defining the bond types
            (e.g., SINGLE, DOUBLE, AROMATIC) and their order for one-hot encoding.
        add_self_loops (bool): Adds self-loops to the graph (edges connecting each
            node to itself).
        num_cpus (Optional[int]): Number of CPUs for parallel graph construction.
            Defaults to 1。
    """

    def __init__(
        self,
        atom_vocab: Optional[Dict[str, int]] = None,
        bond_vocab: Optional[Tuple[BT, ...]] = None,
        remove_h: bool = True,
        add_self_loops: bool = False,
        edge_mode: str = "bidirectional",
        num_cpus: Optional[int] = None,
    ) -> None:
        if atom_vocab is None:
            if remove_h is False:
                atom_vocab = {
                    "H": 0,
                    "C": 1,
                    "N": 2,
                    "O": 3,
                    "F": 4,
                    "P": 5,
                    "S": 6,
                    "Cl": 7,
                    "Br": 8,
                    "I": 9,
                }
            else:
                atom_vocab = {
                    "C": 0,
                    "N": 1,
                    "O": 2,
                    "F": 3,
                    "P": 4,
                    "S": 5,
                    "Cl": 6,
                    "Br": 7,
                    "I": 8,
                }
        if bond_vocab is None:
            bond_vocab = (BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC)

        self.atom_vocab = dict(atom_vocab)
        self.bond_vocab = tuple(bond_vocab)
        self.remove_h = remove_h
        self.add_self_loops = add_self_loops
        self.edge_mode = edge_mode
        self.num_cpus = 1 if num_cpus is None else int(num_cpus)

    @staticmethod
    def build_one(
        mol: Chem.Mol,
        remove_h: bool,
        atom_vocab: Dict[str, int],
        bond_vocab: Tuple[BT, ...],
        add_self_loops: bool,
        edge_mode: str = "bidirectional",
    ) -> Optional[pgl.Graph]:
        if mol is None:
            return None
        if remove_h:
            mol = Chem.RemoveHs(mol)

        N = mol.GetNumAtoms()
        if N == 0:
            return None

        # 1) Node Features: One-hot encoding of atomic symbols.
        idxs: List[int] = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in atom_vocab:
                return None  # Unknown Elements: Can be replaced with an extended
                # vocabulary or placeholder <unk>
            idxs.append(atom_vocab[sym])
        idxs_np = np.asarray(idxs, dtype=np.int64)  # [N]
        x = np.eye(len(atom_vocab), dtype=np.float32)[idxs_np]  # [N, num_atom_types]

        # 2) Build the edges first (construct edge_index/edge_attr)
        rows, cols, etypes = [], [], []
        bt2id = {bt: i + 1 for i, bt in enumerate(bond_vocab)}  # 0 for empty values

        def push(u, v, et):
            rows.append(u)
            cols.append(v)
            etypes.append(et)

        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            et = bt2id.get(bond_name(b.GetBondType()), 0)
            if edge_mode == "directed":
                push(u, v, et)
            elif edge_mode == "undirected":
                uu, vv = (u, v) if u < v else (v, u)
                push(uu, vv, et)
            elif edge_mode == "bidirectional":
                push(u, v, et)
                push(v, u, et)
            else:
                raise ValueError(f"Unknown edge_mode: {edge_mode}")

        if len(rows) == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, len(bond_vocab) + 1), dtype=np.float32)
        else:
            row_np = np.asarray(rows, dtype=np.int64)
            col_np = np.asarray(cols, dtype=np.int64)
            et_np = np.asarray(etypes, dtype=np.int64)
            edge_attr = np.eye(len(bond_vocab) + 1, dtype=np.float32)[et_np]  # [E, K]
            # Deterministic ordering by (row, col)
            order = np.argsort(row_np * max(1, N) + col_np, kind="mergesort")
            row_np, col_np, edge_attr = row_np[order], col_np[order], edge_attr[order]
            edge_index = np.stack([row_np, col_np], axis=0)  # [2, E]

        # 3）Hydrogen removal on the graph (masking+relabeling). The Mol keep unchanged.
        if remove_h:
            h_id = atom_vocab.get("H", None)
            if h_id is not None:
                to_keep_nodes = idxs_np != h_id  # keep only non-H atoms
                edge_index, edge_attr = subgraph(
                    subset=to_keep_nodes,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    relabel_nodes=True,
                    num_nodes=N,
                )
                # Remove the H channel from node features and filter rows to kept nodes
                keep_cols = np.array(
                    [i for i in range(len(atom_vocab)) if i != h_id], dtype=np.int64
                )
                x = x[to_keep_nodes][:, keep_cols]
            else:
                # If "H" is not in the vocab, we leave the graph/features as-is.
                pass

        # 4) (Optional) Add self-loops based on the updated node count
        N_new = int(x.shape[0])
        edges_e2 = edge_index.T.astype(np.int64)  # [E,2]
        if add_self_loops and N_new > 0:
            self_e2 = np.stack([np.arange(N_new), np.arange(N_new)], axis=1).astype(
                np.int64
            )
            self_ea = np.eye(len(bond_vocab) + 1, dtype=np.float32)[
                np.zeros((N_new,), dtype=np.int64)
            ]
            edges_e2 = np.concatenate([edges_e2, self_e2], axis=0)
            edge_attr = np.concatenate([edge_attr, self_ea], axis=0)

        # 5) Return a PGL graph. (If running in worker processes, consider returning
        #    NumPy arrays and wrapping into pgl.Graph)
        return pgl.Graph(
            num_nodes=int(x.shape[0]),
            edges=edges_e2,
            node_feat={"feat": x},
            edge_feat={"feat": edge_attr},
        )

    def __call__(
        self, mols: Union[Sequence[Chem.Mol], Chem.Mol]
    ) -> Union[List[pgl.Graph], pgl.Graph, None]:
        if isinstance(mols, (list, tuple)):
            return p_map(
                MolecularGraphConverter.build_one,
                mols,
                [self.remove_h] * len(mols),
                [self.atom_vocab] * len(mols),
                [self.bond_vocab] * len(mols),
                [self.add_self_loops] * len(mols),
                [self.edge_mode] * len(mols),
                num_cpus=self.num_cpus,
                desc="Building graphs",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        else:
            return MolecularGraphConverter.build_one(
                mols,
                self.remove_h,
                self.atom_vocab,
                self.bond_vocab,
                self.add_self_loops,
                self.edge_mode,
            )


def subgraph(
    subset: Union[np.ndarray, List[int]],
    edge_index: np.ndarray,
    edge_attr: Optional[np.ndarray] = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    *,
    return_edge_mask: bool = False,
) -> Union[Tuple[paddle.Tensor], Tuple[paddle.Tensor]]:
    """
    Build the induced subgraph for the nodes specified by `subset`, in NumPy only.

    Args:
        subset: Node subset as a boolean mask of shape (N,) or as an index list/array.
        edge_index: Array of shape [2, E] with directed edges (u, v), dtype int64.
        edge_attr: Optional edge features of shape [E, D]; filtered alongside edges.
        relabel_nodes: If True, remap kept nodes to a compact 0..K-1 range.
        num_nodes: Total number of nodes N (inferred if None).
        return_edge_mask: If True, also return the boolean mask over original edges.

    Returns:
        (edge_index_new, edge_attr_new[, edge_mask])
        - edge_index_new: [2, E_kept] int64
        - edge_attr_new:  [E_kept, D] or None
        - edge_mask:      [E] bool (only if return_edge_mask=True)
    """

    edge_index = np.asarray(edge_index, dtype=np.int64)
    E = edge_index.shape[1]
    assert edge_index.shape[0] == 2, "edge_index must be [2, E]"

    # Normalize `subset` to a boolean node mask of length N
    if isinstance(subset, (list, tuple, np.ndarray)) and (
        not np.asarray(subset).dtype == bool
    ):
        subset = np.asarray(subset, dtype=np.int64)
        if num_nodes is None:
            num_nodes = int(edge_index.max()) + 1 if E > 0 else (int(subset.max()) + 1)
        node_mask = np.zeros((num_nodes,), dtype=bool)
        node_mask[subset] = True
    else:
        node_mask = np.asarray(subset, dtype=bool)
        if num_nodes is None:
            num_nodes = node_mask.shape[0]

    # Keep edges whose both endpoints are inside the node subset
    src = edge_index[0]
    dst = edge_index[1]
    edge_mask = node_mask[src] & node_mask[dst]
    keep_idx = np.nonzero(edge_mask)[0]

    if keep_idx.size == 0:
        new_edge_index = np.empty((2, 0), dtype=np.int64)
        new_edge_attr = (
            np.empty((0, edge_attr.shape[1]), dtype=edge_attr.dtype)
            if edge_attr is not None
            else None
        )
        if return_edge_mask:
            return new_edge_index, new_edge_attr, edge_mask
        return new_edge_index, new_edge_attr

    # Filter edges (and attributes) by mask
    new_edge_index = edge_index[:, keep_idx]
    new_edge_attr = edge_attr[keep_idx] if edge_attr is not None else None

    # Optionally remap node ids to 0..K-1 over the kept nodes
    if relabel_nodes:
        subset_idx = np.nonzero(node_mask)[0]
        mapping = -np.ones((num_nodes,), dtype=np.int64)
        mapping[subset_idx] = np.arange(subset_idx.shape[0], dtype=np.int64)
        new_edge_index = mapping[new_edge_index]

    if return_edge_mask:
        return new_edge_index, new_edge_attr, edge_mask
    return new_edge_index, new_edge_attr


def bond_name(bt):
    # Compatible across RDKit versions
    try:
        return bt.name
    except AttributeError:
        return str(bt).split(".")[-1]
