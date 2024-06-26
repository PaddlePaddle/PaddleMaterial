from __future__ import absolute_import
from __future__ import annotations

import abc
import hashlib
import json
import os
import traceback
from functools import partial
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np
import paddle
from tqdm import trange


class BaseDataset(object):
    def __init__(
        self,
        name,
        url=None,
        raw_dir=None,
        hash_key=(),
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        self._name = name
        self._url = url
        self._force_reload = force_reload
        self._verbose = verbose
        self._hash_key = hash_key
        self._hash = self._get_hash()
        self._transform = transform
        self._raw_dir = raw_dir
        self._load()

    @abc.abstractmethod
    def process(self):
        """Overwrite to realize your own logic of processing the input data."""
        pass

    def has_cache(self):
        """Overwrite to realize your own logic of
        deciding whether there exists a cached dataset.

        By default False.
        """
        return False

    def _load(self):
        """Entry point from __init__ to load the dataset.

        If cache exists:

          - Load the dataset from saved pgl graph and information files.
          - If loadin process fails, re-download and process the dataset.

        else:

          - Download the dataset if needed.
          - Process the dataset and build the pgl graph.
          - Save the processed dataset into files.
        """
        self.process()

    def _get_hash(self):
        """Compute the hash of the input tuple

        Example
        -------
        Assume `self._hash_key = (10, False, True)`

        >>> hash_value = self._get_hash()
        >>> hash_value
        'a770b222'
        """
        hash_func = hashlib.sha1()
        hash_func.update(str(self._hash_key).encode("utf-8"))
        return hash_func.hexdigest()[:8]

    def _get_hash_url_suffix(self):
        """Get the suffix based on the hash value of the url."""
        if self._url is None:
            return ""
        else:
            hash_func = hashlib.sha1()
            hash_func.update(str(self._url).encode("utf-8"))
            return "_" + hash_func.hexdigest()[:8]

    @property
    def url(self):
        """Get url to download the raw dataset."""
        return self._url

    @property
    def name(self):
        """Name of the dataset."""
        return self._name

    @property
    def raw_dir(self):
        """Raw file directory contains the input data folder."""
        return self._raw_dir

    @property
    def raw_path(self):
        """Directory contains the input data files.
        By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return os.path.join(self.raw_dir, self.name + self._get_hash_url_suffix())

    @property
    def verbose(self):
        """Whether to print information."""
        return self._verbose

    @property
    def hash(self):
        """Hash value for the dataset and the setting."""
        return self._hash

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Gets the data object at index."""
        pass

    @abc.abstractmethod
    def __len__(self):
        """The number of examples in the dataset."""
        pass

    def __repr__(self):
        return (
            f'Dataset("{self.name}", num_graphs={len(self)},'
            + f" save_path={self.save_path})"
        )


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


class MGLDataset(BaseDataset):
    """Create a dataset including pgl graphs."""

    def __init__(
        self,
        filename: str = "pgl_graph.bin",
        filename_lattice: str = "lattice.pt",
        filename_line_graph: str = "pgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.json",
        converter: (GraphConverter | None) = None,
        threebody_cutoff: (float | None) = None,
        directed_line_graph: bool = False,
        structures: (list | None) = None,
        labels: (dict[str, list] | None) = None,
        name: str = "MGLDataset",
        graph_labels: (list[int | float] | None) = None,
        clear_processed: bool = False,
        raw_dir: (str | None) = None,
    ):
        """
        Args:
            filename: file name for storing pgl graphs.
            filename_lattice: file name for storing lattice matrixs.
            filename_line_graph: file name for storing pgl line graphs.
            filename_state_attr: file name for storing state attributes.
            filename_labels: file name for storing labels.
            converter: pgl graph converter.
            threebody_cutoff: cutoff for three body.
            directed_line_graph (bool): Whether to create a directed line graph (CHGNet), or an
                undirected 3body line graph (M3GNet)
                Default: False (for M3GNet)
            structures: Pymatgen structure.
            labels: targets, as a dict of {name: list of values}.
            name: name of dataset.
            graph_labels: state attributes.
            clear_processed: Whether to clear the stored structures after processing into graphs. Structures
                are not really needed after the conversion to pgl graphs and can take a significant amount of memory.
                Setting this to True will delete the structures from memory.
            raw_dir : str specifying the directory that will store the downloaded data or the directory that already
                stores the input data.
        """
        self.filename = filename
        self.filename_lattice = filename_lattice
        self.filename_line_graph = filename_line_graph
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.converter = converter
        self.structures = structures or []
        self.labels = labels or {}
        for k, v in self.labels.items():
            self.labels[k] = v.tolist() if isinstance(v, np.ndarray) else v
        self.threebody_cutoff = threebody_cutoff
        self.directed_line_graph = directed_line_graph
        self.graph_labels = graph_labels
        self.clear_processed = clear_processed
        super().__init__(
            name=name,
            raw_dir=raw_dir,
            verbose=True,
            force_reload=True,
        )

    def has_cache(self) -> bool:
        """Check if the pgl_graph.bin exists or not."""
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
        ]
        return all(
            os.path.exists(os.path.join(self.save_path, f)) for f in files_to_check
        )

    def process(self):
        """Convert Pymatgen structure into pgl graphs."""
        num_graphs = len(self.structures)
        graphs, lattices, line_graphs, state_attrs = [], [], [], []
        not_use_idxs = []
        for idx in trange(num_graphs):
            structure = self.structures[idx]
            graph, lattice, state_attr = self.converter.get_graph(structure)
            if graph.num_edges == 0:
                not_use_idxs.append(idx)
                continue
            graphs.append(graph)
            lattices.append(lattice)
            state_attrs.append(state_attr)
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
        if self.graph_labels is not None:
            state_attrs = paddle.to_tensor(data=self.graph_labels).astype(dtype="int64")
        else:
            state_attrs = np.array(state_attrs, dtype="float32")
        if self.clear_processed:
            del self.structures
            self.structures = []
        self.graphs = graphs
        self.lattices = lattices
        self.state_attr = state_attrs
        for key, value in self.labels.items():
            new_value = []
            for idx in range(len(value)):
                if idx in not_use_idxs:
                    continue
                new_value.append(value[idx])
            self.labels[key] = new_value
        return self.graphs, self.lattices, self.state_attr

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        # items = [self.graphs[idx], self.lattices[idx], self.state_attr[idx],
        #     {k: paddle.to_tensor(data=v[idx], dtype='float32') for k,
        #     v in self.labels.items()}]
        items = [
            self.graphs[idx],
            self.lattices[idx],
            self.state_attr[idx],
            {k: np.array(v[idx], dtype="float32") for k, v in self.labels.items()},
        ]
        return tuple(items)

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)
