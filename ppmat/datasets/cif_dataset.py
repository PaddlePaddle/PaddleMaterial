from __future__ import absolute_import
from __future__ import annotations

import os.path as osp
import pickle
from typing import Dict
from typing import Literal
from typing import Optional

import numpy as np
import pandas as pd
from paddle.io import Dataset

from ppmat.datasets.collate_fn import Data
from ppmat.datasets.structure_converter import Structure2Graph
from ppmat.datasets.utils import build_structure_from_file
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import logger


class CIFDataset(Dataset):
    """cif dataset."""

    def __init__(
        self,
        csv_path: str,
        cif_path: str,
        niggli: bool = True,
        primitive: bool = False,
        converter_cfg: Dict = None,
        transforms=None,
        num_cpus: Optional[int] = None,
        element_types: Literal["DEFAULT_ELEMENTS"] = "DEFAULT_ELEMENTS",
        cache: bool = True,
        **kwargs,
    ):

        self.csv_path = csv_path
        self.cif_path = cif_path

        self.niggli = niggli
        self.primitive = primitive
        self.converter_cfg = converter_cfg
        self.transforms = transforms
        self.num_cpus = num_cpus
        self.cache = cache

        if cache:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "cached settings match your current settings."
            )

        self.property_data = self.read_csv(csv_path)
        self.num_samples = len(self.property_data["cif"])

        if element_types.upper() == "DEFAULT_ELEMENTS":
            self.element_types = DEFAULT_ELEMENTS
        else:
            raise ValueError("element_types must be 'DEFAULT_ELEMENTS'.")

        # when cache is True, load cached structures from cache file
        cache_path = osp.join(csv_path.rsplit(".", 1)[0] + "_strucs.pkl")
        if self.cache and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.structures = pickle.load(f)
            logger.info(
                f"Load {len(self.structures)} cached structures from {cache_path}"
            )
        else:
            # build structures from cif files
            cif_files = []
            for cif in self.property_data["cif"]:
                if not cif.endswith(".cif"):
                    cif += ".cif"
                cif_files.append(osp.join(cif_path, cif))
            self.structures = build_structure_from_file(
                cif_files,
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
            cache_path = osp.join(
                csv_path.rsplit(".", 1)[0] + f"_{graph_method}_graphs.pkl"
            )
            if self.cache and osp.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.graphs = pickle.load(f)
                logger.info(f"Load {len(self.graphs)} cached graphs from {cache_path}")
                assert len(self.graphs) == len(self.structures)
            else:
                # build graphs from structures
                self.converter = Structure2Graph(**self.converter_cfg)
                self.graphs = self.converter(self.structures)
                logger.info(f"Convert {len(self.graphs)} structures into graphs")
                if self.cache:
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.graphs, f)
                    logger.info(
                        f"Save {len(self.graphs)} converted graphs to {cache_path}"
                    )
        else:
            self.graphs = None

    def read_csv(self, path):
        data = pd.read_csv(path)
        logger.info(f"Read {len(data)} rows from {path}")
        data = {key: data[key].tolist() for key in data if "Unnamed" not in key}
        return data

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
            data["graph"] = self.graphs[idx]
        else:
            structure = self.structures[idx]
            data["structure_array"] = self.get_structure_array(structure)

        # get formation energy per atom
        if "formation_energy_per_atom" in self.property_data:
            data["formation_energy_per_atom"] = np.array(
                [self.property_data["formation_energy_per_atom"][idx]]
            ).astype("float32")
        # get band gap
        if "band_gap" in self.property_data:
            data["band_gap"] = np.array([self.property_data["band_gap"][idx]]).astype(
                "float32"
            )
        # use cif file name as id
        data["id"] = self.property_data["cif"][idx]

        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return self.num_samples
