import json
import os
import os.path as osp
import pickle
from typing import Callable
from typing import Dict
from typing import Literal
from typing import Optional

import numpy as np
import paddle
from p_tqdm import p_map
from pymatgen.core.structure import Structure

from ppmat.datasets.collate_fn import Data
from ppmat.datasets.structure_converter import Structure2Graph
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import ELEMENTS_94
from ppmat.utils import logger


def build_structure_from_dict(crystal_dict, num_cpus=None):
    """Build crystal from cif string."""

    def build_one(crystal_dict):
        crystal = Structure.from_dict(crystal_dict)
        return crystal

    if isinstance(crystal_dict, dict):
        return build_one(crystal_dict)
    elif isinstance(crystal_dict, list):
        canonical_crystal = p_map(build_one, crystal_dict, num_cpus=num_cpus)
        return canonical_crystal
    else:
        raise TypeError("crystal_dict must be str or list.")


class SturctureDataFromJsonl(paddle.io.Dataset):
    def __init__(
        self,
        path: str,
        converter_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        element_types: Literal["DEFAULT_ELEMENTS"] = "DEFAULT_ELEMENTS",
        cache: bool = False,
    ):
        super().__init__()
        self.path = path
        self.converter_cfg = converter_cfg
        self.transforms = transforms
        self.cache = cache

        if cache:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "cached settings match your current settings."
            )

        self.jsonl_data = self.read_jsonl(path)
        self.num_samples = len(self.jsonl_data)
        if element_types.upper() == "DEFAULT_ELEMENTS":
            self.element_types = DEFAULT_ELEMENTS
        elif element_types.upper() == "ELEMENTS_94":
            self.element_types = ELEMENTS_94
        else:
            raise ValueError("element_types must be 'DEFAULT_ELEMENTS'.")
        # when cache is True, load cached structures from cache file
        cache_path = osp.join(path.rsplit(".", 1)[0] + "_strucs_dev.pkl")
        if self.cache and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.structures = pickle.load(f)
            logger.info(
                f"Load {len(self.structures)} cached structures from {cache_path}"
            )
        else:
            # build structures from cif
            structure_dicts = [data["structure"] for data in self.jsonl_data]
            self.structures = build_structure_from_dict(structure_dicts)
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

    def read_jsonl(self, file_path):

        data_lines = []
        with open(file_path, "r") as f:
            for line in f:
                data_point = json.loads(line)
                data_lines.append(data_point)
        return data_lines

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

    def __getitem__(self, idx):
        data = {}
        if self.graphs is not None:
            # Obtain the graph from the cache, as this data is frequently utilized
            # for training property prediction models.
            if isinstance(self.graphs[idx], str):
                with open(self.graphs[idx], "rb") as f:
                    data["graph"] = pickle.load(f)
            else:
                data["graph"] = self.graphs[idx]
        else:
            structure = self.structures[idx]
            data["structure_array"] = self.get_structure_array(structure)

        if "formation_energy_per_atom" in self.jsonl_data[idx]:
            data["formation_energy_per_atom"] = np.array(
                [self.jsonl_data[idx]["formation_energy_per_atom"]]
            ).astype("float32")
        if "band_gap" in self.jsonl_data[idx]:
            data["band_gap"] = np.array([self.jsonl_data[idx]["band_gap"]]).astype(
                "float32"
            )
        if "energy" in self.jsonl_data[idx]:
            data["e"] = np.array(self.jsonl_data[idx]["energy"]).astype("float32")

        interatomic_properties = {}
        if self.jsonl_data[idx].get("forces", None) is not None:
            interatomic_properties["f"] = np.array(
                self.jsonl_data[idx]["forces"]
            ).astype("float32")
        if self.jsonl_data[idx].get("stress", None) is not None:
            interatomic_properties["stress"] = np.array(
                self.jsonl_data[idx]["stress"]
            ).astype("float32")
        if self.jsonl_data[idx].get("magmom", None) is not None:
            interatomic_properties["magmom"] = np.array(
                self.jsonl_data[idx]["magmom"]
            ).astype("float32")

        if interatomic_properties:
            data["interatomic_properties"] = Data(interatomic_properties)

        if "material_id" in self.jsonl_data[idx]:
            data["id"] = self.jsonl_data[idx]["material_id"]

        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples
