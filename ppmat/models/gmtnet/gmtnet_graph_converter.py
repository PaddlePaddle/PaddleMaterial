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
from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np
import paddle
import spglib
from jarvis.core.specie import get_node_attributes
from pymatgen.core import Structure
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from ppmat.datasets.geometric_data_type.data import Data
from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.io import CartesianTensor


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
        self._irreps_output = o3.Irreps(
            "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o"
        )
        self._cartesian_converter = CartesianTensor("ij")
        self._symprec = 1e-5

    @staticmethod
    def load_structure_from_cif(cif_path) -> Structure:
        """Load a CIF without parser-side coordinate idealization for GMTNet."""
        try:
            return Structure.from_file(
                cif_path,
                primitive=False,
                sort=False,
                merge_tol=0.0,
                frac_tolerance=0.0,
            )
        except Exception as error:
            raise ValueError(
                f"GMTNet precision-preserving CIF parsing failed for {cif_path!s}."
            ) from error

    @staticmethod
    def _dataset_value(dataset, name: str):
        try:
            return getattr(dataset, name)
        except AttributeError:
            return dataset[name]

    @staticmethod
    def _unique_rotations(rotations: np.ndarray) -> np.ndarray:
        unique_rotations = []
        seen = set()
        for rotation in rotations:
            key = tuple(rotation.reshape(-1).tolist())
            if key not in seen:
                seen.add(key)
                unique_rotations.append(rotation)
        return np.asarray(unique_rotations, dtype=np.float32)

    def _symmetry_dataset(self, structure: Structure):
        dataset = spglib.get_symmetry_dataset(
            (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers),
            symprec=self._symprec,
        )
        if dataset is None:
            raise ValueError("spglib could not determine the structure symmetry.")
        return dataset

    def _prediction_constraints(self, structure: Structure, symmetry_dataset):
        rotations = self._unique_rotations(
            np.asarray(self._dataset_value(symmetry_dataset, "rotations"))
        )
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float32).T
        transformed_rotations = lattice @ rotations @ np.linalg.inv(lattice)
        representations = self._irreps_output.D_from_matrix(
            paddle.to_tensor(transformed_rotations, dtype="float32")
        )
        average = representations.sum(axis=0) / representations.shape[0]
        feature_mask = average * (average > 1e-5).astype("float32")
        mask = paddle.concat([
            paddle.arange(8, dtype="float32") + 10.0,
            (paddle.arange(24, dtype="float32") + 18.0) * 100.0,
        ])
        feature_total = representations.sum(axis=0) @ mask
        selected = feature_total[[0, 2, 3, 4, 8, 9, 10, 11, 12]]
        ideal_matrix = self._cartesian_converter.to_cartesian(selected).numpy()
        flattened = ideal_matrix.reshape(-1)
        matrix_equal = np.abs(flattened[:, None] - flattened[None, :]) < (
            0.0001 * np.abs(flattened[:, None] + flattened[None, :]) / 2.0
        )
        return feature_mask, matrix_equal

    def _build_prediction_input_from_structure(self, structure: Structure):
        symmetry_dataset = self._symmetry_dataset(structure)
        equivalent_atoms = np.asarray(
            self._dataset_value(symmetry_dataset, "equivalent_atoms"), dtype=np.int32
        )
        feature_mask, matrix_equal = self._prediction_constraints(
            structure, symmetry_dataset
        )
        return {
            "graph": self(structure, equivalent_atoms),
            "feature_mask": feature_mask.unsqueeze(0),
            "matrix_equal": paddle.to_tensor(matrix_equal, dtype="bool").unsqueeze(0),
        }

    def build_prediction_input(self, values):
        """Return GMTNet prediction Mappings while preserving Mapping inputs."""
        if isinstance(values, Mapping):
            return values
        if isinstance(values, Structure):
            return self._build_prediction_input_from_structure(values)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError("structures must be a pymatgen Structure or a sequence of Structures.")
        if not values:
            raise ValueError("structures must not be empty.")
        if all(isinstance(value, Mapping) for value in values):
            return values
        if not all(isinstance(value, Structure) for value in values):
            raise TypeError("structures must contain only pymatgen Structure objects.")
        return [self._build_prediction_input_from_structure(value) for value in values]

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
