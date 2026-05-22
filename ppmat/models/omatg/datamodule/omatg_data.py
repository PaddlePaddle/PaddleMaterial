# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the specific language governing permissions and
# limitations under the License.

"""OMATGData class for representing batches of crystal structures.
"""

from typing import Optional, Any
import paddle
from paddle.io import Dataset

from ppmat.datasets.omatg.structure import Structure


class OMATGData:
    """Representation of single/batch crystal structures.
    Batch format: n_atoms(batch_size,), species(total_atoms,), cell(batch,3,3),
    pos(total_atoms,3), pos_is_fractional(batch_size,), ptr(batch+1,).
    """

    def __init__(self, structure: Optional[Structure] = None) -> None:
        if structure is None:
            self.n_atoms = None
            self.species = None
            self.cell = None
            self.pos = None
            self.pos_is_fractional = None
            self.property_dict = None
            self.batch = None
            self.ptr = None
        else:
            self._from_structure(structure)

    def _from_structure(self, structure: Structure) -> None:
        """Initialize from a single Structure object."""
        self.n_atoms = paddle.to_tensor([len(structure.atomic_numbers)], dtype="int64")
        self.species = structure.atomic_numbers
        self.cell = structure.cell.unsqueeze(0)  # Shape: (1, 3, 3)
        self.pos = structure.pos
        self.pos_is_fractional = paddle.to_tensor(
            [structure.pos_is_fractional], dtype="bool"
        )
        self.property_dict = [structure.property_dict]
        self.batch = paddle.zeros_like(self.species, dtype="int64")
        self.ptr = paddle.to_tensor([0, len(self.species)], dtype="int64")

    @classmethod
    def from_batch(
        cls, structures: list[Structure], concatenate: bool = True
    ) -> "OMATGData":
        """
        Create OMATGData from a list of Structure objects.

        :param structures: List of Structure objects.
        :param concatenate: If True, concatenate into batch; else return list.
        """
        if not concatenate:
            return [cls(s) for s in structures]

        if len(structures) == 0:
            return cls()

        if len(structures) == 1:
            return cls(structures[0])

        # Concatenate multiple structures
        cells = []
        species_list = []
        pos_list = []
        pos_is_fractional_list = []
        n_atoms_list = []
        batch_indices = []
        properties_list = []

        for i, struct in enumerate(structures):
            n_atoms_list.append(len(struct.atomic_numbers))
            species_list.append(struct.atomic_numbers)
            cells.append(struct.cell)
            pos_list.append(struct.pos)
            pos_is_fractional_list.append(
                paddle.to_tensor([struct.pos_is_fractional], dtype="bool")
            )
            batch_indices.append(
                paddle.full([len(struct.atomic_numbers)], i, dtype="int64")
            )
            properties_list.append(struct.property_dict)

        data = cls()
        data.n_atoms = paddle.to_tensor(n_atoms_list, dtype="int64")
        data.species = paddle.concat(species_list)
        data.cell = paddle.stack(cells)  # (batch, 3, 3)
        data.pos = paddle.concat(pos_list)
        data.pos_is_fractional = paddle.concat(pos_is_fractional_list)
        data.batch = paddle.concat(batch_indices)
        data.ptr = paddle.concat(
            [
                paddle.to_tensor([0], dtype="int64"),
                paddle.cumsum(data.n_atoms, axis=0).cast("int64"),
            ]
        )
        data.property_dict = properties_list

        return data

    @property
    def num_graphs(self) -> int:
        """Return the number of structures in the batch."""
        if self.n_atoms is None:
            return 0
        return len(self.n_atoms)

    @property
    def num_atoms(self) -> int:
        """Return the total number of atoms across all structures."""
        if self.species is None:
            return 0
        return len(self.species)

    def get_graph(self, idx: int) -> Structure:
        """
        Get a single Structure from the batch by index.

        :param idx: Graph index (0 to num_graphs-1).
        :return: Structure object.
        """
        if idx < 0 or idx >= self.num_graphs:
            raise IndexError(f"Index {idx} out of range for batch of {self.num_graphs}")

        start = int(self.ptr[idx])
        end = int(self.ptr[idx + 1])

        species = self.species[start:end]
        pos = self.pos[start:end]
        cell = self.cell[idx]
        pos_is_fractional = bool(self.pos_is_fractional[idx])

        if self.property_dict and idx < len(self.property_dict):
            prop = self.property_dict[idx]
        else:
            prop = {}

        return Structure(
            cell=cell,
            atomic_numbers=species,
            pos=pos,
            property_dict=prop,
            metadata={},
            pos_is_fractional=pos_is_fractional,
        )

    def slice(self, idx: int) -> slice:
        """Return a slice object for accessing atoms of structure idx."""
        start = int(self.ptr[idx])
        end = int(self.ptr[idx + 1])
        return slice(start, end)

    def to_dict(self) -> dict[str, Any]:
        """Return data as a dictionary."""
        return {
            "n_atoms": self.n_atoms,
            "species": self.species,
            "cell": self.cell,
            "pos": self.pos,
            "pos_is_fractional": self.pos_is_fractional,
            "batch": self.batch,
            "ptr": self.ptr,
            "property_dict": self.property_dict,
        }

    @classmethod
    def _from_dict(cls, data_dict: dict[str, Any]) -> "OMATGData":
        """Create OMATGData from a dictionary (internal use)."""
        data = cls()
        data.n_atoms = data_dict.get("n_atoms")
        data.species = data_dict.get("species")
        data.cell = data_dict.get("cell")
        data.pos = data_dict.get("pos")
        data.pos_is_fractional = data_dict.get("pos_is_fractional")
        data.batch = data_dict.get("batch")
        data.ptr = data_dict.get("ptr")
        data.property_dict = data_dict.get("property_dict")
        return data

    def clone(self) -> "OMATGData":
        """Create a deep copy of this OMATGData object."""
        data = OMATGData()
        if self.n_atoms is not None:
            data.n_atoms = self.n_atoms.clone()
        if self.species is not None:
            data.species = self.species.clone()
        if self.cell is not None:
            data.cell = self.cell.clone()
        if self.pos is not None:
            data.pos = self.pos.clone()
        if self.pos_is_fractional is not None:
            data.pos_is_fractional = self.pos_is_fractional.clone()
        if self.batch is not None:
            data.batch = self.batch.clone()
        if self.ptr is not None:
            data.ptr = self.ptr.clone()
        data.property_dict = self.property_dict
        return data

    def set_field(self, field_name: str, value: paddle.Tensor) -> None:
        """
        Set a field value in this OMATGData object.

        :param field_name: Name of the field to set.
        :param value: New value for the field.
        """
        if field_name == "n_atoms":
            self.n_atoms = value
        elif field_name == "species":
            self.species = value
        elif field_name == "cell":
            self.cell = value
        elif field_name == "pos":
            self.pos = value
        elif field_name == "pos_is_fractional":
            self.pos_is_fractional = value
        elif field_name == "batch":
            self.batch = value
        elif field_name == "ptr":
            self.ptr = value
        else:
            raise ValueError(f"Unknown field: {field_name}")

    def get_field(self, field_name: str) -> paddle.Tensor:
        """
        Get a field value from this OMATGData object.

        :param field_name: Name of the field to get.
        :return: Field value.
        """
        if field_name == "n_atoms":
            return self.n_atoms
        elif field_name == "species":
            return self.species
        elif field_name == "cell":
            return self.cell
        elif field_name == "pos":
            return self.pos
        elif field_name == "pos_is_fractional":
            return self.pos_is_fractional
        elif field_name == "batch":
            return self.batch
        elif field_name == "ptr":
            return self.ptr
        else:
            raise ValueError(f"Unknown field: {field_name}")

    def __repr__(self) -> str:
        return (
            f"OMATGData(num_graphs={self.num_graphs}, "
            f"num_atoms={self.num_atoms}, "
            f"cell={self.cell.shape if self.cell is not None else None})"
        )


class OMATGDataset(Dataset):
    """
    Dataset class for OMatG crystal structures.

    :param structures: List of Structure objects or path to LMDB database.
    :param property_keys: Properties to include.
    """

    def __init__(
        self,
        structures: list[Structure] | None = None,
        property_keys: list[str] | None = None,
    ) -> None:
        self._structures = structures or []
        self._property_keys = property_keys or []

    def __len__(self) -> int:
        return len(self._structures)

    def __getitem__(self, idx: int) -> OMATGData:
        return OMATGData(self._structures[idx])

    def add_structure(self, structure: Structure) -> None:
        self._structures.append(structure)

    @classmethod
    def from_lmdb(cls, lmdb_path: str, property_keys: list[str] | None = None) -> "OMATGDataset":
        import lmdb
        import pickle

        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        structures = []
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    key_str = key.decode("ascii") if isinstance(key, bytes) else key
                    if key_str.startswith("__"):
                        continue
                    data = pickle.loads(value)
                    cell = paddle.to_tensor(data["cell"], dtype="float32")
                    atomic_numbers = paddle.to_tensor(data["atomic_numbers"], dtype="int64")
                    pos = paddle.to_tensor(data["pos"], dtype="float32")
                    structures.append(
                        Structure(cell=cell, atomic_numbers=atomic_numbers, pos=pos)
                    )
        finally:
            env.close()

        return cls(structures=structures, property_keys=property_keys)
