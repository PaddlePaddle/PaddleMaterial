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

"""MD17 dataset for molecular dynamics trajectories.

The MD17 dataset (Chmiela et al., 2017) contains ab-initio molecular dynamics
trajectories for small organic molecules. Each snapshot includes atomic
positions, total energy, and per-atom forces.

Reference:
  S. Chmiela, A. Tkatchenko, H. E. Sauceda, I. Poltavsky, K. T. Schütt,
  K.-R. Müller. Machine Learning of Accurate Energy-Conserving Molecular
  Force Fields. Science Advances, 2017.
"""

from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.models import build_graph_converter
from ppmat.utils import download, logger

try:
    from pymatgen.core import Lattice, Structure
except ImportError:
    Structure = None
    Lattice = None


# Mapping from molecule name to original MD17 NPZ filename.
MD17_FILES = {
    "benzene": "md17_benzene2017.npz",
    "uracil": "md17_uracil.npz",
    "naphthalene": "md17_naphthalene.npz",
    "aspirin": "md17_aspirin.npz",
    "salicylic_acid": "md17_salicylic.npz",
    "malonaldehyde": "md17_malonaldehyde.npz",
    "ethanol": "md17_ethanol.npz",
    "toluene": "md17_toluene.npz",
}

# Atomic number → element symbol (for pymatgen Structure creation).
_Z_TO_SYMBOL = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S",
}


class MD17Dataset(Dataset):
    """MD17 molecular dynamics trajectory dataset.

    Loads an MD17 NPZ file and converts each snapshot into a dict compatible
    with PaddleMaterials models (optionally building a PGL graph via the
    graph converter).

    The NPZ files contain:
      - ``z``: atomic numbers, shape [num_atoms]
      - ``R``: positions, shape [num_snapshots, num_atoms, 3] (Angstrom)
      - ``E``: energies, shape [num_snapshots, 1] (kcal/mol)
      - ``F``: forces, shape [num_snapshots, num_atoms, 3] (kcal/mol/A)

    Args:
        path (str): Root directory for dataset storage.
        molecule (str): Molecule name (e.g. "ethanol").
        property_names (str or list): Target property name(s). Default: "energy".
        build_graph_cfg (dict, optional): Config for graph converter.
        max_samples (int, optional): Limit dataset size (for faster debugging).
        url (str, optional): Custom download URL. Default: BCS mirror.
        box_size (float): Side length (A) of the cubic cell used for
            non-periodic molecules. Default: 100.0.
        cache_graphs (bool): Whether to cache built graphs to disk. Default: True.
    """

    # BCS mirror for MD17 NPZ files
    default_url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17"

    def __init__(
        self,
        path: str,
        molecule: str = "ethanol",
        property_names: Union[str, List[str]] = "energy",
        *,
        build_graph_cfg: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        url: Optional[str] = None,
        box_size: float = 100.0,
        cache_graphs: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if molecule not in MD17_FILES:
            raise ValueError(
                f"Unknown molecule '{molecule}'. Choose from: {list(MD17_FILES)}"
            )

        if isinstance(property_names, str):
            property_names = [property_names]
        self.property_names = property_names
        self.molecule = molecule
        self.box_size = box_size
        self.build_graph_cfg = build_graph_cfg

        # Paths
        os.makedirs(path, exist_ok=True)
        self.raw_dir = osp.join(path, "raw_md17")
        os.makedirs(self.raw_dir, exist_ok=True)

        npz_name = MD17_FILES[molecule]
        npz_path = osp.join(self.raw_dir, npz_name)

        # Download if needed
        if not osp.exists(npz_path):
            base_url = url or self.default_url
            full_url = f"{base_url}/{npz_name}"
            logger.info(f"Downloading MD17 {molecule} from {full_url}")
            download.download_file(full_url, npz_path)

        # Load raw data
        raw = np.load(npz_path)
        self.atomic_numbers = raw["z"].astype(np.int64)  # [num_atoms]
        self.positions = raw["R"].astype(np.float32)      # [N, num_atoms, 3]
        self.energies = raw["E"].astype(np.float32)        # [N, 1] or [N]
        self.forces = raw["F"].astype(np.float32)          # [N, num_atoms, 3]

        if self.energies.ndim == 1:
            self.energies = self.energies[:, None]

        if max_samples is not None:
            self.positions = self.positions[:max_samples]
            self.energies = self.energies[:max_samples]
            self.forces = self.forces[:max_samples]

        self.num_samples = len(self.positions)
        logger.info(
            f"MD17 {molecule}: {self.num_samples} snapshots, "
            f"{len(self.atomic_numbers)} atoms/snapshot"
        )

        # Build and cache graphs if configured
        self.graphs = None
        if build_graph_cfg is not None:
            graph_converter_name = build_graph_cfg.get("__class_name__", "custom")
            cutoff = build_graph_cfg.get("__init_params__", {}).get("cutoff", 5)
            cache_dir = osp.join(
                path,
                f"md17_{molecule}_cache_{graph_converter_name}_cutoff_{int(cutoff)}",
                "graphs",
            )

            done_flag = osp.join(cache_dir, "completed.flag")
            if cache_graphs and osp.exists(done_flag):
                logger.info(f"Loading cached graphs from {cache_dir}")
                self.graphs = self._load_cached_graphs(cache_dir)
            else:
                if dist.get_rank() == 0:
                    logger.info("Building graphs for MD17 dataset...")
                    os.makedirs(cache_dir, exist_ok=True)
                    converter = build_graph_converter(build_graph_cfg)
                    structures = self._build_structures()
                    self.graphs = converter(structures)
                    if cache_graphs:
                        self._save_cached_graphs(cache_dir)
                        with open(done_flag, "w") as f:
                            f.write("done")
                if dist.is_initialized():
                    dist.barrier()
                if self.graphs is None:
                    self.graphs = self._load_cached_graphs(cache_dir)

    def _build_structures(self) -> list:
        """Convert all snapshots to pymatgen Structures."""
        if Structure is None:
            raise RuntimeError("pymatgen is required: pip install pymatgen")

        lattice = Lattice.from_parameters(
            self.box_size, self.box_size, self.box_size, 90, 90, 90
        )
        species = [_Z_TO_SYMBOL.get(z, str(z)) for z in self.atomic_numbers]

        structures = []
        for i in range(self.num_samples):
            coords = self.positions[i]  # [num_atoms, 3]
            struct = Structure(
                lattice,
                species,
                coords,
                coords_are_cartesian=True,
            )
            structures.append(struct)
        return structures

    def _save_cached_graphs(self, cache_dir: str) -> None:
        for i, g in enumerate(self.graphs):
            with open(osp.join(cache_dir, f"{i:08d}.pkl"), "wb") as f:
                pickle.dump(g, f)

    def _load_cached_graphs(self, cache_dir: str) -> list:
        files = sorted(
            f for f in os.listdir(cache_dir) if f.endswith(".pkl")
        )
        graphs = []
        for fn in files:
            with open(osp.join(cache_dir, fn), "rb") as f:
                graphs.append(pickle.load(f))
        return graphs

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = {}

        if self.graphs is not None:
            data["graph"] = self.graphs[idx]
        else:
            # Return raw data without graph (fallback)
            data["pos"] = self.positions[idx]
            data["atomic_numbers"] = self.atomic_numbers.copy()
            data["cell"] = np.eye(3, dtype="float32") * self.box_size
            data["natoms"] = len(self.atomic_numbers)
            data["pbc"] = np.array([False, False, False], dtype=bool)

        # Properties
        for pname in self.property_names:
            if pname == "energy":
                data["energy"] = self.energies[idx]
            elif pname == "forces":
                data["forces"] = self.forces[idx]
            else:
                data[pname] = self.energies[idx]

        return data
