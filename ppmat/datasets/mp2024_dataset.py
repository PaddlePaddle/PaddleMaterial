from __future__ import absolute_import
from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Dict
from typing import Literal
from typing import Optional

import numpy as np
from p_tqdm import p_map
from paddle.io import Dataset

from ppmat.datasets.collate_fn import Data
from ppmat.datasets.structure_converter import Structure2Graph
from ppmat.datasets.utils import build_structure_from_dict
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import logger


class MP2024Dataset(Dataset):
    """mp.2024.11.1 dataset."""

    def __init__(
        self,
        path: str,
        niggli: bool = True,
        primitive: bool = False,
        converter_cfg: Dict = None,
        transforms=None,
        num_cpus: Optional[int] = None,
        element_types: Literal["DEFAULT_ELEMENTS"] = "DEFAULT_ELEMENTS",
        filter_key: Optional[str] = None,
        cache: bool = True,
        **kwargs,
    ):

        self.path = path
        self.niggli = niggli
        self.primitive = primitive
        self.converter_cfg = converter_cfg
        self.transforms = transforms
        self.num_cpus = num_cpus
        self.element_types = element_types
        self.filter_key = filter_key
        self.cache = cache

        if cache:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "cached settings match your current settings."
            )

        self.txt_data = self.read_txt(path)
        if filter_key is not None:
            self.txt_data = self.filter_data(self.txt_data, filter_key)
        self.num_samples = len(self.txt_data)
        if element_types.upper() == "DEFAULT_ELEMENTS":
            self.element_types = DEFAULT_ELEMENTS
        else:
            raise ValueError("element_types must be 'DEFAULT_ELEMENTS'.")

        # when cache is True, load cached structures from cache file
        cache_path = osp.join(path.rsplit(".", 1)[0] + "_strucs.pkl")
        if self.cache and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.structures = pickle.load(f)
            logger.info(
                f"Load {len(self.structures)} cached structures from {cache_path}"
            )
        else:
            # build structures from dict
            self.structures = build_structure_from_dict(
                [data["structure"] for data in self.txt_data],
                niggli=niggli,
                primitive=primitive,
                num_cpus=num_cpus,
            )
            logger.info(f"Build {len(self.structures)} structures")
            if self.cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.structures, f)
                logger.info(
                    f"Save {len(self.structures)} built structures to {cache_path}"
                )
        # build graphs from structures
        if converter_cfg is not None:
            # load cached graphs from cache file
            graph_method = converter_cfg["method"]
            cache_path = osp.join(path.rsplit(".", 1)[0] + f"_{graph_method}_graphs")
            if osp.exists(cache_path):
                self.graphs = [
                    osp.join(cache_path, f"{i}.pkl")
                    for i in range(len(self.structures))
                ]
                logger.info(f"Load {len(self.graphs)} cached graphs from {cache_path}")
                assert len(self.graphs) == len(self.structures)
            else:
                # build graphs from structures
                self.converter = Structure2Graph(**self.converter_cfg)
                self.graphs = self.converter(self.structures)
                os.makedirs(cache_path, exist_ok=True)
                for i, graph in enumerate(self.graphs):
                    with open(os.path.join(cache_path, f"{i}.pkl"), "wb") as f:
                        pickle.dump(graph, f)

                self.graphs = [
                    osp.join(cache_path, f"{i}.pkl")
                    for i in range(len(self.structures))
                ]

                logger.info(f"Load {len(self.graphs)} cached graphs from {cache_path}")
                assert len(self.graphs) == len(self.structures)

        else:
            self.graphs = None

    def read_txt(self, path):

        data = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data.append(line.strip())
        data = p_map(eval, data, num_cpus=self.num_cpus)
        logger.info(f"Total number of samples loaded is {len(data)}.")
        return data

    def filter_data(self, data, filter_key):
        filtered_data = []
        for _, sample in enumerate(data):
            if sample[filter_key] is not None:
                filtered_data.append(sample)
        logger.info(
            f"Filtering done. Total number of samples left is {len(filtered_data)}. "
            f"{len(data)-len(filtered_data)} samples are removed."
        )
        return filtered_data

    def get_structure_array(self, structure):
        atom_types = np.array(
            [self.element_types.index(site.specie.symbol) for site in structure]
        )
        # get lattice parameters and matrix
        lattice_parameters = structure.lattice.parameters
        lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
        angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
        lattice = structure.lattice.matrix.astype("float32")

        structure_array = Data(
            {
                "frac_coords": structure.frac_coords.astype("float32"),
                "cart_coords": structure.cart_coords.astype("float32"),
                "atom_types": atom_types,
                "lattice": lattice.reshape(1, 3, 3),
                "lengths": lengths,
                "angles": angles,
                "num_atoms": np.array([tuple(atom_types.shape)[0]]),
            }
        )
        return structure_array

    def __getitem__(self, idx: int):
        """Get item at index idx."""
        data = {}
        # get graph
        if self.graphs is not None:
            if isinstance(self.graphs[idx], str):
                with open(self.graphs[idx], "rb") as f:
                    data["graph"] = pickle.load(f)
            else:
                data["graph"] = self.graphs[idx]
        else:
            structure = self.structures[idx]
            data["structure_array"] = self.get_structure_array(structure)

        # get formation energy per atom
        if "formation_energy_per_atom" in self.txt_data[idx]:
            data["formation_energy_per_atom"] = np.array(
                [self.txt_data[idx]["formation_energy_per_atom"]]
            ).astype("float32")
        # get band gap
        if "band_gap" not in self.txt_data[idx]:
            data["band_gap"] = np.array([self.txt_data[idx]["band_gap"]]).astype(
                "float32"
            )

        data["id"] = self.txt_data[idx]["material_id"]

        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return self.num_samples
