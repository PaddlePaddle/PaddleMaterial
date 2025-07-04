from typing import Callable
from typing import Dict
from typing import Literal
from typing import Optional

import numpy as np
import paddle

from ppmat.datasets.collate_fn import Data
from ppmat.datasets.structure_converter import Structure2Graph
from ppmat.datasets.utils import build_structure_from_array
from ppmat.utils import DEFAULT_ELEMENTS
from ppmat.utils import logger


class TensorDataset(paddle.io.Dataset):
    def __init__(
        self,
        crystal_list: list,
        niggli: bool = True,
        primitive: bool = False,
        converter_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        element_types: Literal["DEFAULT_ELEMENTS"] = "DEFAULT_ELEMENTS",
    ):
        super().__init__()
        self.crystal_list = crystal_list
        self.niggli = niggli
        self.primitive = primitive
        self.converter_cfg = converter_cfg
        self.transforms = transforms

        self.num_samples = len(crystal_list)
        if element_types.upper() == "DEFAULT_ELEMENTS":
            self.element_types = DEFAULT_ELEMENTS
        else:
            raise ValueError("element_types must be 'DEFAULT_ELEMENTS'.")

        # build structures from cif
        self.structures = build_structure_from_array(
            crystal_list, niggli=niggli, primitive=primitive
        )
        logger.info(f"Build {len(self.structures)} structures")

        # build graphs from structures
        if converter_cfg is not None:
            # build graphs from structures
            self.converter = Structure2Graph(**self.converter_cfg)
            self.graphs = self.converter(self.structures)
            logger.info(f"Convert {len(self.graphs)} structures into graphs")
        else:
            self.graphs = None

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
            data["graph"] = self.graphs[idx]
        else:
            structure = self.structures[idx]
            data["structure_array"] = self.get_structure_array(structure)
        data["idx"] = idx
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples
