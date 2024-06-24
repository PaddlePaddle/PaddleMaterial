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


class DGLDataset(object):
    """The basic DGL dataset for creating graph datasets.
    This class defines a basic template class for DGL Dataset.
    The following steps will be executed automatically:

      1. Check whether there is a dataset cache on disk
         (already processed and stored on the disk) by
         invoking ``has_cache()``. If true, goto 5.
      2. Call ``download()`` to download the data if ``url`` is not None.
      3. Call ``process()`` to process the data.
      4. Call ``save()`` to save the processed dataset on disk and goto 6.
      5. Call ``load()`` to load the processed dataset from disk.
      6. Done.

    Users can overwite these functions with their
    own data processing logic.

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str
        Url to download the raw dataset. Default: None
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: same as raw_dir
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
        Default: (), the corresponding hash value is ``'f9065fa7'``.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    url : str
        The URL to download the dataset
    name : str
        The dataset name
    raw_dir : str
        Directory to store all the downloaded raw datasets.
    raw_path : str
        Path to the downloaded raw dataset folder. An alias for
        ``os.path.join(self.raw_dir, self.name)``.
    save_dir : str
        Directory to save all the processed datasets.
    save_path : str
        Path to the processed dataset folder. An alias for
        ``os.path.join(self.save_dir, self.name)``.
    verbose : bool
        Whether to print more runtime information.
    hash : str
        Hash value for the dataset and the setting.
    """

    def __init__(
        self,
        name,
        url=None,
        raw_dir=None,
        save_dir=None,
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
        if save_dir is None:
            self._save_dir = self._raw_dir
        else:
            self._save_dir = save_dir
        self._load()

    def download(self):
        """Overwite to realize your own logic of downloading data.

        It is recommended to download the to the :obj:`self.raw_dir`
        folder. Can be ignored if the dataset is
        already in :obj:`self.raw_dir`.
        """
        pass

    def save(self):
        """Overwite to realize your own logic of
        saving the processed dataset into files.

        It is recommended to use ``dgl.data.utils.save_graphs``
        to save dgl graph into files and use
        ``dgl.data.utils.save_info`` to save extra
        information into files.
        """
        pass

    def load(self):
        """Overwite to realize your own logic of
        loading the saved dataset from files.

        It is recommended to use ``dgl.data.utils.load_graphs``
        to load dgl graph from files and use
        ``dgl.data.utils.load_info`` to load extra information
        into python dict object.
        """
        pass

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

          - Load the dataset from saved dgl graph and information files.
          - If loadin process fails, re-download and process the dataset.

        else:

          - Download the dataset if needed.
          - Process the dataset and build the dgl graph.
          - Save the processed dataset into files.
        """
        load_flag = not self._force_reload and self.has_cache()
        if load_flag:
            try:
                self.load()
                if self.verbose:
                    print("Done loading data from cached files.")
            except KeyboardInterrupt:
                raise
            except:
                load_flag = False
                if self.verbose:
                    print(traceback.format_exc())
                    print("Loading from cache failed, re-processing.")
        if not load_flag:
            self.process()
            # self.save()
            if self.verbose:
                print("Done saving data into cached files.")

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
    def save_dir(self):
        """Directory to save the processed dataset."""
        return self._save_dir

    @property
    def save_path(self):
        """Path to save the processed dataset."""
        return os.path.join(self.save_dir, self.name + self._get_hash_url_suffix())

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


def compute_pair_vector_and_distance(g: dgl.DGLGraph):
    """Calculate bond vectors and distances using dgl graphs.

    Args:
    g: DGL graph

    Returns:
    bond_vec (torch.tensor): bond distance between two atoms
    bond_dist (torch.tensor): vector from src node to dst node
    """
    dst_pos = g.node_feat["pos"][g.edges[:, 1]] + g.edge_feat["pbc_offshift"]
    src_pos = g.node_feat["pos"][g.edges[:, 0]]
    bond_vec = dst_pos - src_pos
    bond_dist = np.linalg.norm(bond_vec, axis=1)
    return bond_vec, bond_dist


class MGLDataset(DGLDataset):
    """Create a dataset including dgl graphs."""

    def __init__(
        self,
        filename: str = "dgl_graph.bin",
        filename_lattice: str = "lattice.pt",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.json",
        include_line_graph: bool = False,
        converter: (GraphConverter | None) = None,
        threebody_cutoff: (float | None) = None,
        directed_line_graph: bool = False,
        structures: (list | None) = None,
        labels: (dict[str, list] | None) = None,
        name: str = "MGLDataset",
        graph_labels: (list[int | float] | None) = None,
        clear_processed: bool = False,
        save_cache: bool = True,
        raw_dir: (str | None) = None,
        save_dir: (str | None) = None,
    ):
        """
        Args:
            filename: file name for storing dgl graphs.
            filename_lattice: file name for storing lattice matrixs.
            filename_line_graph: file name for storing dgl line graphs.
            filename_state_attr: file name for storing state attributes.
            filename_labels: file name for storing labels.
            include_line_graph: whether to include line graphs.
            converter: dgl graph converter.
            threebody_cutoff: cutoff for three body.
            directed_line_graph (bool): Whether to create a directed line graph (CHGNet), or an
                undirected 3body line graph (M3GNet)
                Default: False (for M3GNet)
            structures: Pymatgen structure.
            labels: targets, as a dict of {name: list of values}.
            name: name of dataset.
            graph_labels: state attributes.
            clear_processed: Whether to clear the stored structures after processing into graphs. Structures
                are not really needed after the conversion to DGL graphs and can take a significant amount of memory.
                Setting this to True will delete the structures from memory.
            save_cache: whether to save the processed dataset. The dataset can be reloaded from save_dir
                Default: True
            raw_dir : str specifying the directory that will store the downloaded data or the directory that already
                stores the input data.
                Default: ~/.dgl/
            save_dir : directory to save the processed dataset. Default: same as raw_dir.
        """
        self.filename = filename
        self.filename_lattice = filename_lattice
        self.filename_line_graph = filename_line_graph
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.include_line_graph = include_line_graph
        self.converter = converter
        self.structures = structures or []
        self.labels = labels or {}
        for k, v in self.labels.items():
            self.labels[k] = v.tolist() if isinstance(v, np.ndarray) else v
        self.threebody_cutoff = threebody_cutoff
        self.directed_line_graph = directed_line_graph
        self.graph_labels = graph_labels
        self.clear_processed = clear_processed
        self.save_cache = save_cache
        super().__init__(
            name=name,
            raw_dir=raw_dir,
            save_dir=save_dir,
            verbose=True,
            force_reload=True,
        )

    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not."""
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
        ]
        if self.include_line_graph:
            files_to_check.append(self.filename_line_graph)
        return all(
            os.path.exists(os.path.join(self.save_path, f)) for f in files_to_check
        )

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
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
            if self.include_line_graph:
                line_graph = create_line_graph(
                    graph, self.threebody_cutoff, directed=self.directed_line_graph
                )
                for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                    line_graph.node_feat.pop(name)
                line_graphs.append(line_graph)
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
        if self.include_line_graph:
            self.line_graphs = line_graphs
            return (self.graphs, self.lattices, self.line_graphs, self.state_attr)

        for key, value in self.labels.items():
            new_value = []
            for idx in range(len(value)):
                if idx in not_use_idxs:
                    continue
                new_value.append(value[idx])
            self.labels[key] = new_value
        return self.graphs, self.lattices, self.state_attr

    def save(self):
        """Save dgl graphs and labels to self.save_path."""
        if self.save_cache is False:
            return
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.labels:
            with open(os.path.join(self.save_path, self.filename_labels), "w") as file:
                json.dump(self.labels, file)
        save_graphs(os.path.join(self.save_path, self.filename), self.graphs)
        paddle.save(
            obj=self.lattices, path=os.path.join(self.save_path, self.filename_lattice)
        )
        paddle.save(
            obj=self.state_attr,
            path=os.path.join(self.save_path, self.filename_state_attr),
        )
        if self.include_line_graph:
            save_graphs(
                os.path.join(self.save_path, self.filename_line_graph), self.line_graphs
            )

    def load(self):
        """Load dgl graphs from files."""
        self.graphs, _ = load_graphs(os.path.join(self.save_path, self.filename))
        self.lattices = paddle.load(
            path=os.path.join(self.save_path, self.filename_lattice)
        )
        if self.include_line_graph:
            self.line_graphs, _ = load_graphs(
                os.path.join(self.save_path, self.filename_line_graph)
            )
        self.state_attr = paddle.load(
            path=os.path.join(self.save_path, self.filename_state_attr)
        )
        with open(os.path.join(self.save_path, self.filename_labels)) as f:
            self.labels = json.load(f)

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
        if self.include_line_graph:
            items.insert(2, self.line_graphs[idx])
        return tuple(items)

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)
