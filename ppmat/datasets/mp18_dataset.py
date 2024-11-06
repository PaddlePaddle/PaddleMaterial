from __future__ import absolute_import
from __future__ import annotations

import os.path as osp
import pickle

import numpy as np
import pandas as pd
from paddle.io import Dataset
from pymatgen.core import Structure
from tqdm import trange

from ppmat.datasets.ext_pymatgen import Structure2Graph
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import logger


def compute_pair_vector_and_distance(g):
    """Calculate bond vectors and distances using pgl graphs.

    Args:
    g: PGL graph

    Returns:
    bond_vec (paddle.tensor): bond distance between two atoms
    bond_dist (paddle.tensor): vector from src node to dst node
    """
    dst_pos = g.node_feat["pos"][g.edges[:, 1]] + g.edge_feat["pbc_offshift"]
    src_pos = g.node_feat["pos"][g.edges[:, 0]]
    bond_vec = dst_pos - src_pos
    bond_dist = np.linalg.norm(bond_vec, axis=1)
    return bond_vec, bond_dist


class MP18Dataset(Dataset):
    """mp.2018.6.1 dataset."""

    def __init__(
        self,
        path: str = "./data/mp18/mp.2018.6.1.json",
        converter_cutoff: float = 4.0,
        transforms=None,
        cache: bool = True,
        **kwargs,
    ):

        self.path = path
        self.converter = Structure2Graph(
            element_types=DEFAULT_ELEMENTS, cutoff=converter_cutoff
        )
        self.transforms = transforms

        cache_path = osp.join(path.rsplit(".", 1)[0] + ".pkl")
        if cache and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                (
                    self.structures,
                    self.graphs,
                    self.mp_ids,
                    self.formation_energy_per_atom,
                ) = pickle.load(f)
            logger.message(f"Loaded cached data from {cache_path}")
            logger.message(f"Total number of samples loaded is {len(self.structures)}.")
        else:
            (
                self.structures,
                self.mp_ids,
                self.formation_energy_per_atom,
            ) = self.convert_to_structure(path)
            self.graphs = self.structure_to_graph(self.structures)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    (
                        self.structures,
                        self.graphs,
                        self.mp_ids,
                        self.formation_energy_per_atom,
                    ),
                    f,
                )
            logger.message(f"Cached data saved to {cache_path}")

    def convert_to_structure(self, path):
        data = pd.read_json(path)
        structures = []
        mp_ids = []
        total_structures = len(data["structure"])
        for i in trange(total_structures, desc="Converting string to structures..."):
            mid, structure_str = data["material_id"][i], data["structure"][i]
            struct = Structure.from_str(structure_str, fmt="cif")
            structures.append(struct)
            mp_ids.append(mid)
        formation_energy_per_atom = data["formation_energy_per_atom"].tolist()
        return structures, mp_ids, formation_energy_per_atom

    def structure_to_graph(self, structures):
        """Convert Pymatgen structure into pgl graphs."""
        num_graphs = len(structures)
        graphs = []
        # not_use_idxs = []
        for idx in trange(num_graphs, desc="Converting structures to graphs..."):
            structure = self.structures[idx]
            graph, lattice, state_attr = self.converter.get_graph(structure)
            # if graph.num_edges == 0:
            #     not_use_idxs.append(idx)
            #     continue
            graphs.append(graph)
            # lattices.append(lattice)
            graph.node_feat["pos"] = structure.cart_coords.astype("float32")
            graph.edge_feat["pbc_offshift"] = np.matmul(
                graph.edge_feat["pbc_offset"], lattice[0]
            )
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edge_feat["bond_vec"] = bond_vec
            graph.edge_feat["bond_dist"] = bond_dist
            graph.node_feat.pop("pos")
            graph.edge_feat.pop("pbc_offshift")
            graph.numpy()

        return graphs

    def __getitem__(self, idx: int):
        """Get item at index idx."""
        # idx  =1

        data = {}
        # get graph
        # graph, lattice, state_attr = self.converter.get_graph(structure)
        # graph.node_feat["pos"] = structure.cart_coords.astype("float32")
        # graph.edge_feat["pbc_offshift"] = np.matmul(
        #     graph.edge_feat["pbc_offset"], lattice[0]
        # )
        # bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
        # graph.edge_feat["bond_vec"] = bond_vec
        # graph.edge_feat["bond_dist"] = bond_dist
        # graph.node_feat.pop("pos")
        # graph.edge_feat.pop("pbc_offshift")
        # graph.numpy()
        data["graph"] = self.graphs[idx]
        # data["lattice"] = lattice

        # get formation energy per atom
        data["formation_energy_per_atom"] = np.array(
            [self.formation_energy_per_atom[idx]]
        ).astype("float32")
        # get material id
        # data["mp_id"] = self.mp_ids[idx]
        data["state_attr"] = np.array([0.0, 0.0]).astype("float32")

        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return len(self.structures)
