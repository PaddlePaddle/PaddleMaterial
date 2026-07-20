"""Historical-equivalent crystal graph conversion for GMTNet.

This converter reproduces the graph-construction portion of the historical
GMTNet ``atoms2graphs`` path for ordered pymatgen structures. It converts the
structure to JARVIS ``Atoms``, constructs canonicalized periodic neighbors,
and returns a PaddleMaterials geometric ``Data`` object containing ``x``,
``edge_index``, and ``edge_attr``.

The first version intentionally serves only the frozen GMTNet graph contract.
Callers must provide ``equivalent_atoms`` with one entry per structure atom.
It does not generate feature masks, matrix-equality constraints, labels, data
splits, or batches.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import paddle
from jarvis.core.specie import get_node_attributes
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from ppmat.datasets.geometric_data_type.data import Data


class GMTNetGraphConverter:
    """Convert an ordered pymatgen structure into the GMTNet graph format.

    Args:
        cutoff: Initial periodic-neighbor cutoff in Angstrom.
        max_neighbors: Minimum number of neighbors used to select each site.
        atom_features: JARVIS node-feature representation. Only ``"cgcnn"``
            is supported by this historical-equivalence implementation.
        use_canonize: Whether to canonicalize periodic edge representations.
        reduce_cell: Historical ``reduce`` argument. The current GMTNet
            baseline requires ``False``.

    The returned :class:`~ppmat.datasets.geometric_data_type.data.Data` has
    ``x`` with shape ``[num_nodes, 92]``, ``edge_index`` with shape
    ``[2, num_edges]``, and ``edge_attr`` with shape ``[num_edges, 3]``.
    """

    def __init__(
        self,
        cutoff: float = 4.0,
        max_neighbors: int = 16,
        atom_features: str = "cgcnn",
        use_canonize: bool = True,
        reduce_cell: bool = False,
    ):
        if atom_features != "cgcnn":
            raise ValueError(
                "GMTNetGraphConverter only supports atom_features='cgcnn'."
            )
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive.")
        if reduce_cell:
            raise ValueError(
                "GMTNetGraphConverter v1 supports only reduce_cell=False."
            )

        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.atom_features = atom_features
        self.use_canonize = bool(use_canonize)
        self.reduce_cell = bool(reduce_cell)
        self._adaptor = JarvisAtomsAdaptor()

    @staticmethod
    def _canonize_edge(src_id, dst_id, src_image, dst_image):
        if dst_id < src_id:
            src_id, dst_id = dst_id, src_id
            src_image, dst_image = dst_image, src_image

        if not np.array_equal(src_image, (0, 0, 0)):
            shift = src_image
            src_image = tuple(np.subtract(src_image, shift))
            dst_image = tuple(np.subtract(dst_image, shift))

        if src_image != (0, 0, 0):
            raise RuntimeError("Canonical edge source image must be (0, 0, 0).")
        return src_id, dst_id, src_image, dst_image

    def _build_edges(self, atoms, cutoff: float):
        all_neighbors = atoms.get_all_neighbors(r=cutoff)
        if len(all_neighbors) == 0:
            raise RuntimeError("Structure contains no atoms.")

        min_neighbors = min(len(neighbor_list) for neighbor_list in all_neighbors)
        if min_neighbors < self.max_neighbors:
            lattice = atoms.lattice
            expanded_cutoff = (
                max(lattice.a, lattice.b, lattice.c)
                if cutoff < max(lattice.a, lattice.b, lattice.c)
                else 2.0 * cutoff
            )
            return self._build_edges(atoms, expanded_cutoff)

        edges = defaultdict(set)
        for site_index, neighbor_list in enumerate(all_neighbors):
            neighbor_list = sorted(neighbor_list, key=lambda neighbor: neighbor[2])
            distances = np.asarray([neighbor[2] for neighbor in neighbor_list])
            neighbor_ids = np.asarray([neighbor[1] for neighbor in neighbor_list])
            images = np.asarray([neighbor[3] for neighbor in neighbor_list])

            max_distance = distances[self.max_neighbors - 1]
            selected = distances <= max_distance
            neighbor_ids = neighbor_ids[selected]
            images = images[selected]

            for destination_index, image in zip(neighbor_ids, images):
                source_index, destination_index, _, destination_image = (
                    self._canonize_edge(
                        site_index,
                        int(destination_index),
                        (0, 0, 0),
                        tuple(image),
                    )
                )
                if self.use_canonize:
                    edges[(source_index, destination_index)].add(destination_image)
                else:
                    edges[(site_index, int(destination_index))].add(tuple(image))
        return edges

    @staticmethod
    def _build_undirected_edge_data(atoms, edges):
        sources = []
        destinations = []
        displacements = []
        for (source_index, destination_index), images in edges.items():
            for destination_image in images:
                destination_coordinate = (
                    atoms.frac_coords[destination_index] + destination_image
                )
                displacement = atoms.lattice.cart_coords(
                    destination_coordinate - atoms.frac_coords[source_index]
                )
                for source, destination, vector in (
                    (source_index, destination_index, displacement),
                    (destination_index, source_index, -displacement),
                ):
                    sources.append(source)
                    destinations.append(destination)
                    displacements.append(vector)

        if not sources:
            raise RuntimeError("Graph construction produced no edges.")

        edge_index = np.asarray([sources, destinations], dtype=np.int64)
        edge_attr = np.asarray(displacements, dtype=np.float32)
        return edge_index, edge_attr

    def __call__(self, structure, equivalent_atoms: Sequence[int]):
        """Return the historical-equivalent GMTNet graph for ``structure``.

        Args:
            structure: Ordered pymatgen ``Structure`` accepted by
                :class:`pymatgen.io.jarvis.JarvisAtomsAdaptor`.
            equivalent_atoms: Per-atom symmetry-equivalence identifiers. They
                are required to preserve the historical call contract even
                though the frozen ``reduce_cell=False`` path does not reduce
                nodes by equivalence classes.
        """
        if structure is None:
            raise ValueError("structure must be provided.")
        if equivalent_atoms is None:
            raise ValueError("equivalent_atoms must be provided.")

        atoms = self._adaptor.get_atoms(structure)
        if len(equivalent_atoms) != len(atoms.elements):
            raise ValueError(
                "equivalent_atoms length must equal the number of structure atoms."
            )

        edges = self._build_edges(atoms, self.cutoff)
        edge_index, edge_attr = self._build_undirected_edge_data(atoms, edges)

        node_features = np.asarray(
            [
                list(get_node_attributes(element, atom_features=self.atom_features))
                for element in atoms.elements
            ],
            dtype=np.float32,
        )
        graph = Data(
            x=paddle.to_tensor(node_features, dtype="float32"),
            edge_index=paddle.to_tensor(edge_index, dtype="int64"),
            edge_attr=paddle.to_tensor(edge_attr, dtype="float32"),
        )
        required_fields = ("x", "edge_index", "edge_attr")
        if any(getattr(graph, field, None) is None for field in required_fields):
            raise RuntimeError("GMTNet graph is missing required fields.")
        return graph
