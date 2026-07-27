# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
from configparser import ConfigParser
from typing import Dict

import numpy as np
import paddle
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.datasets.deeph_graph import _load_orbital_types
from ppmat.models import build_graph_converter
from ppmat.predictor.base import BasePredictor


class DeepHPredictor(BasePredictor):
    """Predict DeepH Hamiltonian blocks and export them in DeepH npz format."""

    @staticmethod
    def load_orbital_config(config_path):
        config = ConfigParser()
        if not config.read(config_path):
            raise FileNotFoundError(f"Can not read DeepH config: {config_path}")
        return json.loads(config.get("basic", "orbital"))

    @staticmethod
    def _load_processed_structure(processed_dir):
        lattice = np.loadtxt(os.path.join(processed_dir, "lat.dat")).T
        atomic_numbers = (
            np.loadtxt(os.path.join(processed_dir, "element.dat"))
            .astype(np.int64)
            .reshape(-1)
        )
        cart_coords = np.loadtxt(os.path.join(processed_dir, "site_positions.dat")).T
        atoms = Atoms(
            numbers=atomic_numbers,
            positions=cart_coords,
            cell=lattice,
            pbc=True,
        )
        return AseAtomsAdaptor.get_structure(atoms)

    def build_inference_batch(
        self,
        processed_dir,
        species,
        num_l,
        dtype="float32",
    ):
        """Build a label-free DeepH batch from overlap-derived local coordinates."""
        structure = self._load_processed_structure(processed_dir)
        np_dtype = np.dtype(dtype).type
        graph_converter = build_graph_converter(
            {
                "__class_name__": "DeepHGraphConverter",
                "__init_params__": {
                    "radius": -1.0,
                    "max_num_nbr": 0,
                    "default_dtype": np_dtype,
                    "interface": "npz_rc_only",
                    "num_l": num_l,
                    "create_from_DFT": True,
                    "if_lcmp_graph": True,
                    "separate_onsite": False,
                    "target": "hamiltonian",
                },
            }
        )
        graph = graph_converter(structure, processed_dir)

        species_to_index = {
            int(atomic_number): index for index, atomic_number in enumerate(species)
        }
        atomic_numbers = np.asarray(structure.atomic_numbers, dtype=np.int64)
        unknown_species = sorted(set(atomic_numbers) - set(species_to_index))
        if unknown_species:
            raise ValueError(
                "DeepH inference structure contains species missing from the model: "
                f"{unknown_species}"
            )
        atom_features = np.asarray(
            [species_to_index[int(number)] for number in atomic_numbers],
            dtype=np.int64,
        )
        subgraph = graph.subgraph_dict

        return {
            "x": atom_features,
            "edge_index": np.asarray(graph.edge_index, dtype=np.int64),
            "edge_attr": np.asarray(graph.edge_attr, dtype=np_dtype),
            "batch": np.zeros(atom_features.shape[0], dtype=np.int64),
            "pos": np.asarray(structure.cart_coords, dtype=np_dtype),
            "sub_atom_idx": np.asarray(
                subgraph["subgraph_atom_idx"], dtype=np.int64
            ),
            "sub_edge_idx": np.asarray(
                subgraph["subgraph_edge_idx"], dtype=np.int64
            ),
            "sub_edge_ang": np.asarray(
                subgraph["subgraph_edge_ang"], dtype=np_dtype
            ),
            "sub_index": np.asarray(subgraph["subgraph_index"], dtype=np.int64),
            "structure_lattice": np.asarray(
                structure.lattice.matrix, dtype=np_dtype
            ).reshape(1, 3, 3),
            "structure_frac_coords": np.asarray(
                structure.frac_coords, dtype=np_dtype
            ),
            "structure_atomic_numbers": atomic_numbers,
            "structure_folder": os.path.abspath(processed_dir),
        }

    @staticmethod
    def _to_numpy(data):
        if paddle.is_tensor(data):
            return data.numpy()
        return np.asarray(data)

    @staticmethod
    def _get_structure_folder(batch) -> str:
        folder = batch.get("structure_folder")
        if isinstance(folder, (list, tuple)):
            if len(folder) != 1:
                raise ValueError(
                    "DeepH Hamiltonian export currently expects batch_size=1."
                )
            folder = folder[0]
        if not isinstance(folder, str):
            raise TypeError("DeepH Hamiltonian export requires `structure_folder`.")
        return folder

    def build_hamiltonian_npz(
        self,
        pred,
        batch,
        orbital,
        spinful: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Convert DeepH edge predictions to rh.npz-compatible blocks."""
        if spinful:
            raise NotImplementedError(
                "Spinful DeepH Hamiltonian export is not enabled."
            )

        folder = self._get_structure_folder(batch)
        atom_num_orbital = _load_orbital_types(
            os.path.join(folder, "orbital_types.dat")
        )

        pred = self._to_numpy(pred)
        edge_index = self._to_numpy(batch["edge_index"]).astype(np.int64)
        if edge_index.ndim == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.T
        edge_attr = self._to_numpy(batch["edge_attr"])
        atomic_numbers = self._to_numpy(batch["structure_atomic_numbers"]).astype(
            np.int64
        )
        lattice = self._to_numpy(batch["structure_lattice"]).reshape(-1, 3, 3)[0]
        inv_lattice = np.linalg.inv(lattice)

        hamiltonian = {}
        lattice_shifts = np.rint(
            edge_attr[:, 4:7] @ inv_lattice - edge_attr[:, 7:10] @ inv_lattice
        ).astype(int)

        for edge_id in range(edge_attr.shape[0]):
            atom_i, atom_j = edge_index[:, edge_id]
            block = np.zeros(
                (atom_num_orbital[atom_i], atom_num_orbital[atom_j]),
                dtype=pred.dtype,
            )
            atomic_number_i = int(atomic_numbers[atom_i])
            atomic_number_j = int(atomic_numbers[atom_j])

            for out_idx, orbital_dict in enumerate(orbital):
                for n_m_str, orbital_pair in orbital_dict.items():
                    condition_i, condition_j = map(int, n_m_str.split())
                    if (
                        atomic_number_i == condition_i
                        and atomic_number_j == condition_j
                    ):
                        orbital_i, orbital_j = orbital_pair
                        block[orbital_i, orbital_j] = pred[edge_id, out_idx]

            key = json.dumps(
                [
                    int(lattice_shifts[edge_id, 0]),
                    int(lattice_shifts[edge_id, 1]),
                    int(lattice_shifts[edge_id, 2]),
                    int(atom_i) + 1,
                    int(atom_j) + 1,
                ]
            )
            hamiltonian[key] = block
        return hamiltonian

    def save_hamiltonian_npz(
        self,
        path,
        pred,
        batch,
        orbital,
        spinful: bool = False,
    ) -> None:
        """Save DeepH predictions as an rh.npz-compatible Hamiltonian file."""
        hamiltonian = self.build_hamiltonian_npz(
            pred=pred,
            batch=batch,
            orbital=orbital,
            spinful=spinful,
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **hamiltonian)

    def predict_hamiltonian(
        self,
        processed_dir,
        output_path,
        orbital,
        species,
        num_l,
        dtype="float32",
    ):
        """Run Paddle DeepH inference and save predicted Hamiltonian blocks."""
        batch = self.build_inference_batch(
            processed_dir=processed_dir,
            species=species,
            num_l=num_l,
            dtype=dtype,
        )
        if self.eval_with_no_grad:
            with paddle.no_grad():
                prediction = self.model.predict(batch)
        else:
            prediction = self.model.predict(batch)
        prediction = self.post_process(prediction)
        pred = prediction[self.model.target_name]
        self.save_hamiltonian_npz(
            output_path,
            pred=pred,
            batch=batch,
            orbital=orbital,
            spinful=False,
        )
        return output_path
