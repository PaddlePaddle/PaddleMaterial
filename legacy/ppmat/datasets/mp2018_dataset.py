from __future__ import absolute_import
from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.mp_base_dataset import MPBaseDataset
from ppmat.datasets.structure_converter import Structure2Graph
from ppmat.utils.io import read_json


class MP2018Dataset(MPBaseDataset):
    """This class is designed for handling the MP2018 dataset.

    Args:
        path (str, optional): The path of the dataset. Defaults to
            "./data/mp18/mp.2018.6.1.json".
        property_names (Optional[list[str]], optional): Property names you want to use.
            Defaults to None.
        build_structure_cfg (Dict, optional): The configs for building the structure.
            Defaults to None.
        build_graph_cfg (Dict, optional): The configs for building the graph. Defaults
            to None.
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
        path: str = "./data/mp18/mp.2018.6.1.json",
        property_names: Optional[list[str]] = None,
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        **kwargs,  # for compatibility
    ):

        super().__init__(
            path,
            property_names,
            build_structure_cfg,
            build_graph_cfg,
            transforms,
            cache_path,
            overwrite,
            **kwargs,
        )

    def read_data(self, path: str):
        """Read the data from the given json path.

        Args:
            path (str): Path to the data.
        """
        json_data = read_json(path)
        num_samples = len(json_data["structure"])

        idxs = list(json_data["structure"])
        data = defaultdict(list)
        for key in json_data.keys():
            for idx in idxs:
                data[key].append(json_data[key][idx])

        return data, num_samples

    def read_property_data(self, data: Dict, property_names: list[str]):
        """Read the property data from the given data and property names.

        Args:
            data (Dict): Data that contains the property data.
            property_names (list[str]): Property names.
        """
        property_data = {}
        for property_name in property_names:
            if property_name not in data:
                raise ValueError(f"{property_name} not found in the data")
            property_data[property_name] = [
                data[property_name][i] for i in range(self.num_samples)
            ]
        return property_data

    def convert_to_structures(self, data: Dict, build_structure_cfg: Dict):
        """Convert the data to pymatgen structures.

        Args:
            data (Dict): Data dictionary.
            build_structure_cfg (Dict): Build structure configuration.
        """
        structures = BuildStructure(**build_structure_cfg)(data["structure"])
        return structures

    def convert_to_graphs(self, structures: list[Any], build_graph_cfg: Dict):
        """Convert the structure to graph.

        Args:
            structures (list[Any]): List of structures.
            build_graph_cfg (Dict): Build graph configuration.
        """
        if build_graph_cfg is None:
            return None
        converter = Structure2Graph(**build_graph_cfg)
        graphs = converter(structures)
        return graphs
