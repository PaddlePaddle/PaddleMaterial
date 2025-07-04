from __future__ import absolute_import
from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
from paddle.io import Dataset

from ppmat.datasets.collate_fn import Data
from ppmat.utils import logger


class MPBaseDataset(Dataset):
    """Base dataset for Materials Project, this is a supper class for all datasets
        in Materials Project.

    Args:
        path (str): Path to data file.
        property_names (Optional[list[str]], optional):  Property names you want to
            use. Defaults to None.
        build_structure_cfg (Dict, optional): Config for building structure. Defaults
            to None.
        build_graph_cfg (Dict, optional): Config for building graph. Defaults to None.
        transforms (Optional[Callable], optional): The transforms. Defaults to None.
        cache_path (Optional[str], optional): If a cache_path is set, structures and
            graph will be read directly from this path; if the cache does not exist,
            the converted structures and graph will be saved to this path. Defaults
            to None.
        overwrite (bool, optional): Overwrite the existing cache file at the given
            path if it already exists. Defaults to False.
    """

    def __init__(
        self,
        path: str,
        property_names: Optional[list[str]] = None,
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,  # for compatibility
    ):
        self.path = path
        self.property_names = property_names if property_names is not None else []
        self.build_structure_cfg = build_structure_cfg
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms
        self.cache_path = cache_path
        self.overwrite = overwrite

        self.cache_exists = (
            True if self.cache_path is not None and osp.exists(cache_path) else False
        )

        self.row_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")
        self.property_data = self.read_property_data(self.row_data, self.property_names)

        if self.cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used in match your current settings."
            )

        # convert structures and graphs or load from cache file
        if self.cache_exists and not overwrite:
            # load data from cache file
            structure_cache_path = osp.join(cache_path, "structures")
            self.structures = [
                osp.join(structure_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]

            if build_graph_cfg is not None:
                graph_cache_path = osp.join(cache_path, "graphs")
                self.graphs = [
                    osp.join(graph_cache_path, f"{i:010d}.pkl")
                    for i in range(self.num_samples)
                ]
            else:
                self.graphs = None
        else:
            # convert strucutes and graphs
            self.structures = self.convert_to_structures(
                self.row_data, build_structure_cfg
            )
            if build_graph_cfg is not None:
                self.graphs = self.convert_to_graphs(self.structures, build_graph_cfg)
            else:
                self.graphs = None

        assert (
            len(self.structures) == self.num_samples
        ), "The number of structures must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

        # save to cache file
        if self.cache_path is not None:
            if (osp.exists(cache_path) and overwrite) or not osp.exists(cache_path):

                # save builded_structure_cfg and builded_graph_cfg to cache file
                os.makedirs(cache_path, exist_ok=True)
                self.save_to_cache(
                    osp.join(cache_path, "builded_structure_cfg.pkl"),
                    build_structure_cfg,
                )
                self.save_to_cache(
                    osp.join(cache_path, "builded_graph_cfg.pkl"), build_graph_cfg
                )

                # save structures to cache file
                structure_cache_path = osp.join(cache_path, "structures")
                os.makedirs(structure_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl"),
                        self.structures[i],
                    )
                logger.info(
                    f"Save {self.num_samples} structures to {structure_cache_path}"
                )

                # save graphs to cache file
                graph_cache_path = osp.join(cache_path, "graphs")
                os.makedirs(graph_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(graph_cache_path, f"{i:010d}.pkl"), self.graphs[i]
                    )
                logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")

    def read_data(self, path: str):
        raise NotImplementedError(
            f"{self.__class__.__name_}.read_data function is not implemented"
        )

    def read_property_data(self, data: Dict, property_names: list[str]):
        raise NotImplementedError(
            f"{self.__class__.__name_}.read_property_data function is not implemented"
        )

    def convert_to_structures(self, data: Dict, build_structure_cfg: Dict):
        raise NotImplementedError(
            f"{self.__class__.__name_}.convert_to_structures function is not "
            "implemented"
        )

    def convert_to_graphs(self, structures: list[Any], build_graph_cfg: Dict):
        raise NotImplementedError(
            f"{self.__class__.__name_}.convert_to_graphs function is not implemented"
        )

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

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
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph
        else:
            structure = self.structures[idx]
            if isinstance(structure, str):
                structure = self.load_from_cache(structure)
            data["structure_array"] = self.get_structure_array(structure)
        for property_name in self.property_names:
            if property_name in self.property_data:
                data[property_name] = np.array(
                    [self.property_data[property_name][idx]]
                ).astype("float32")
            else:
                raise KeyError(f"Property {property_name} not found.")

        data["id"] = (
            self.property_data["id"][idx] if "id" in self.property_data else idx
        )

        data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self):
        return self.num_samples
