"""Structure-to-Mapping adapter for label-free GMTNet prediction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import paddle
import spglib
from pymatgen.core import Structure

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.io import CartesianTensor


class GMTNetPredictorInputAdapter:
    """Build GMTNet Mapping inputs from one or more pymatgen structures.

    The adapter reconstructs the symmetry-derived inference constraints used by
    the canonical GMTNet data path.  It intentionally never creates labels.

    Args:
        graph_converter: Configured :class:`GMTNetGraphConverter` instance.
        symprec: Symmetry tolerance passed to spglib.
    """

    def __init__(self, graph_converter, symprec: float = 1e-5):
        if graph_converter is None:
            raise ValueError("GMTNet prediction requires a graph_converter.")
        if symprec <= 0:
            raise ValueError("symprec must be positive.")
        self.graph_converter = graph_converter
        self.symprec = float(symprec)
        self.irreps_output = o3.Irreps(
            "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o"
        )
        self.cartesian_converter = CartesianTensor("ij")

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
            symprec=self.symprec,
        )
        if dataset is None:
            raise ValueError("spglib could not determine the structure symmetry.")
        return dataset

    def _constraints(self, structure: Structure, symmetry_dataset):
        rotations = self._unique_rotations(
            np.asarray(self._dataset_value(symmetry_dataset, "rotations"))
        )
        lattice = np.asarray(structure.lattice.matrix, dtype=np.float32).T
        transformed_rotations = lattice @ rotations @ np.linalg.inv(lattice)
        representations = self.irreps_output.D_from_matrix(
            paddle.to_tensor(transformed_rotations, dtype="float32")
        )
        average = representations.sum(axis=0) / representations.shape[0]
        feature_mask = average * (average > 1e-5).astype("float32")

        mask = paddle.concat(
            [
                paddle.arange(8, dtype="float32") + 10.0,
                (paddle.arange(24, dtype="float32") + 18.0) * 100.0,
            ]
        )
        feature_total = representations.sum(axis=0) @ mask
        selected = feature_total[[0, 2, 3, 4, 8, 9, 10, 11, 12]]
        ideal_matrix = self.cartesian_converter.to_cartesian(selected).numpy()
        flattened = ideal_matrix.reshape(-1)
        matrix_equal = np.abs(flattened[:, None] - flattened[None, :]) < (
            0.0001 * np.abs(flattened[:, None] + flattened[None, :]) / 2.0
        )
        return feature_mask, matrix_equal

    def _from_structure(self, structure: Structure):
        symmetry_dataset = self._symmetry_dataset(structure)
        equivalent_atoms = np.asarray(
            self._dataset_value(symmetry_dataset, "equivalent_atoms"), dtype=np.int32
        )
        feature_mask, matrix_equal = self._constraints(structure, symmetry_dataset)
        return {
            "graph": self.graph_converter(structure, equivalent_atoms),
            "feature_mask": feature_mask.unsqueeze(0),
            "matrix_equal": paddle.to_tensor(matrix_equal, dtype="bool").unsqueeze(0),
        }

    def __call__(self, structures):
        """Return one Mapping or an ordered list of label-free Mappings."""
        if isinstance(structures, Structure):
            return self._from_structure(structures)
        if not isinstance(structures, Sequence) or isinstance(structures, (str, bytes)):
            raise TypeError("structures must be a pymatgen Structure or a sequence of Structures.")
        if not structures:
            raise ValueError("structures must not be empty.")
        if not all(isinstance(structure, Structure) for structure in structures):
            raise TypeError("Every item in structures must be a pymatgen Structure.")
        return [self._from_structure(structure) for structure in structures]
