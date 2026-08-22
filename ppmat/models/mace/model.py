# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""MACE interatomic potential for PaddleMaterials train/predict APIs."""

from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import paddle.nn as nn

from .layers import EquivariantLayer


class MACE(nn.Layer):
    """MACE model compatible with PaddleMaterials Trainer / Predictor.

    Keeps a message-passing energy backbone and exposes the same
    ``forward`` / ``predict`` interface as CHGNet / MatterSim, returning
    ``energy_per_atom``, ``force``, and ``stress``.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_basis: int = 8,
        r_max: float = 6.0,
        num_elements: int = 89,
        property_names: Optional[Sequence[str]] = None,
        loss_type: str = "smooth_l1_loss",
        huber_loss_delta: float = 0.1,
        loss_weights_dict: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_basis = num_basis
        self.r_max = r_max
        self.num_elements = num_elements
        self.atomic_numbers = tuple(
            list(range(1, 84)) + list(range(89, 96))
        )
        if len(self.atomic_numbers) != num_elements:
            self.atomic_numbers = tuple(range(1, num_elements + 1))
        self._atomic_number_to_index = {
            atomic_number: index
            for index, atomic_number in enumerate(self.atomic_numbers)
        }

        support_property_names = ["energy_per_atom", "force", "stress"]
        support_loss_weights = {
            "energy_per_atom": 1.0,
            "force": 1.0,
            "stress": 0.1,
        }
        if property_names is None:
            property_names = support_property_names
        for name in property_names:
            assert name in support_property_names, (
                f"{name} is not supported; choose from: {support_property_names}"
            )
        self.property_names = list(property_names)
        if loss_weights_dict is None:
            loss_weights_dict = {
                key: support_loss_weights[key] for key in self.property_names
            }
        self.loss_weights_dict = loss_weights_dict

        # Atomic-number embedding (map Z to [0, num_elements))
        self.embedding = nn.Embedding(num_elements, hidden_dim)
        self.layers = nn.LayerList(
            [EquivariantLayer(hidden_dim, num_basis, r_max) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, 1)

        if loss_type == "mse_loss":
            self.loss_fn = nn.MSELoss()
        elif loss_type in ("smooth_l1_loss", "huber_loss"):
            self.loss_fn = nn.SmoothL1Loss(delta=huber_loss_delta)
        elif loss_type == "l1_loss":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def _atom_type_to_index(self, atom_types: paddle.Tensor) -> paddle.Tensor:
        """Map atomic numbers Z to embedding indices."""
        atomic_numbers = (
            paddle.cast(atom_types, dtype="int64")
            .reshape([-1])
            .cpu()
            .numpy()
            .tolist()
        )
        try:
            indices = [self._atomic_number_to_index[int(z)] for z in atomic_numbers]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported atomic number {exc.args[0]}; supported atomic numbers are "
                f"{self.atomic_numbers}."
            ) from exc
        return paddle.to_tensor(indices, dtype="int64")

    def _ensure_graph_tensor(self, graph):
        """Ensure graph features are paddle Tensors."""
        if hasattr(graph, "tensor") and callable(graph.tensor):
            # Skip conversion when features are already paddle Tensors
            sample = graph.node_feat.get("atom_types")
            if sample is not None and not isinstance(sample, paddle.Tensor):
                graph = graph.tensor()
        return graph

    def _compute_bond_dist(self, graph, positions, lattice):
        """Recompute bond lengths from coordinates (and PBC offsets).

        Graph-builder ``bond_dist`` is a constant and has no coordinate
        gradients. Keep FindPointsInSpheres edge topology and only replace
        the distance computation for autodiff of forces/stresses.
        """
        edges = graph.edges.astype("int64")
        src = edges[:, 0]
        dst = edges[:, 1]
        pbc_offset = None
        if hasattr(graph, "edge_feat") and graph.edge_feat is not None:
            pbc_offset = graph.edge_feat.get("pbc_offset", None)

        if pbc_offset is not None and lattice is not None:
            pbc_offset = pbc_offset.astype("float32")
            # Prefer graph_edge_id; otherwise use the source node's graph id
            if hasattr(graph, "graph_edge_id"):
                edge_batch = graph.graph_edge_id
            else:
                edge_batch = paddle.gather(graph.graph_node_id, src)
            cell_e = paddle.gather(lattice, edge_batch)
            offset = paddle.einsum("bi,bij->bj", pbc_offset, cell_e)
            bond_vec = positions[dst] + offset - positions[src]
        else:
            bond_vec = positions[dst] - positions[src]

        bond_dist = paddle.linalg.norm(bond_vec, axis=-1)
        # Avoid zero distances that break RBF / gradients
        bond_dist = paddle.clip(bond_dist, min=1e-8)
        return bond_dist, src, dst

    def _compute_energy_force_stress(self, graph):
        """Compute energy, forces, and stress from a graph."""
        graph = self._ensure_graph_tensor(graph)

        atom_types = graph.node_feat["atom_types"]
        positions = graph.node_feat["cart_coords"].astype("float32")
        num_atoms = graph.node_feat["num_atoms"].astype("float32").reshape([-1])

        # Forces require gradients w.r.t. positions
        if "force" in self.property_names:
            positions.stop_gradient = False

        lattice = graph.node_feat.get("lattice", None)
        if lattice is not None:
            lattice = lattice.astype("float32")
            if lattice.ndim == 2:
                lattice = lattice.unsqueeze(0)

        strain = None
        if "stress" in self.property_names and lattice is not None:
            # Strain tensor for stress autodiff
            strain = paddle.zeros_like(lattice)
            strain.stop_gradient = False
            lattice = paddle.matmul(
                lattice, paddle.eye(3, dtype=lattice.dtype) + strain
            )
            # Update Cartesian coordinates under small-strain approximation
            node_id = graph.graph_node_id
            strain_atoms = paddle.gather(strain, node_id)
            positions = paddle.einsum(
                "bi,bij->bj",
                positions,
                paddle.eye(3, dtype=positions.dtype) + strain_atoms,
            )

        indices = self._atom_type_to_index(atom_types)
        x = self.embedding(indices)

        # Differentiable bond lengths instead of constant graph bond_dist
        bond_dist, src, dst = self._compute_bond_dist(graph, positions, lattice)

        edge_indices = (src, dst)
        for layer in self.layers:
            x = layer(x, edge_indices, bond_dist)

        atomic_energies = self.output(x).squeeze(-1)  # [N]

        # Aggregate total energy per graph (index_add, same idea as CHGNet)
        node_id = graph.graph_node_id.astype("int32")
        n_graph = int(num_atoms.shape[0])
        energy_buf = paddle.zeros([n_graph, 1], dtype=atomic_energies.dtype)
        energy_buf.stop_gradient = False
        total_energy = energy_buf.index_add(
            axis=0,
            index=node_id,
            value=atomic_energies.reshape([-1, 1]),
        ).reshape([-1])
        energy_per_atom = total_energy / paddle.clip(num_atoms, min=1.0)

        forces = None
        if "force" in self.property_names:
            # Match CHGNet: create_graph=True in train for force-loss backprop.
            # Fall back when higher-order grads are unavailable.
            try:
                force_grad = paddle.grad(
                    outputs=[total_energy.sum()],
                    inputs=[positions],
                    create_graph=self.training,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                force_grad = paddle.grad(
                    outputs=[total_energy.sum()],
                    inputs=[positions],
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            if force_grad is None:
                forces = paddle.zeros_like(positions)
            else:
                forces = -force_grad

        stress = None
        if "stress" in self.property_names and strain is not None:
            volume = paddle.linalg.det(lattice)
            try:
                stress_grad = paddle.grad(
                    outputs=[total_energy.sum()],
                    inputs=[strain],
                    create_graph=self.training,
                    retain_graph=self.training,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                stress_grad = paddle.grad(
                    outputs=[total_energy.sum()],
                    inputs=[strain],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
            if stress_grad is None:
                stress = paddle.zeros_like(lattice)
            else:
                # Convert to GPa (1 eV/Angstrom^3 ≈ 160.21766208 GPa)
                stress = stress_grad / volume.reshape([-1, 1, 1]) / 160.21766208

        return energy_per_atom, forces, stress

    def _forward(self, data):
        """Compute standardized predictions from a batched graph."""
        energy, force, stress = self._compute_energy_force_stress(data["graph"])

        pred_dict = {}
        if "energy_per_atom" in self.property_names:
            pred_dict["energy_per_atom"] = energy
        if "force" in self.property_names and force is not None:
            pred_dict["force"] = force
        if "stress" in self.property_names and stress is not None:
            pred_dict["stress"] = stress
        return pred_dict

    def forward(self, data, return_loss=True, return_prediction=True):
        """Return standardized predictions and optional training losses."""
        assert return_loss or return_prediction

        pred_dict = self._forward(data)

        loss_dict = {}
        if return_loss:
            loss = paddle.to_tensor(0.0, dtype="float32")
            for property_name, pred in pred_dict.items():
                if property_name not in data:
                    continue
                label = data[property_name]
                # Flatten and drop NaNs
                pred_flat = pred.reshape([-1])
                label_flat = label.reshape([-1]).astype(pred_flat.dtype)
                valid = ~paddle.isnan(label_flat)
                if int(valid.astype("int64").sum()) == 0:
                    continue
                loss_property = self.loss_fn(pred_flat[valid], label_flat[valid])
                loss_dict[property_name] = loss_property
                loss = loss + loss_property * self.loss_weights_dict[property_name]
            loss_dict["loss"] = loss

        prediction = pred_dict if return_prediction else {}
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    def _prediction_to_numpy(self, prediction):
        """Convert predictions to numpy for Predictor CSV export."""
        out = {}
        for key, value in prediction.items():
            if isinstance(value, paddle.Tensor):
                value = value.numpy()
            if key == "energy_per_atom" and isinstance(value, np.ndarray):
                value = value.reshape(-1)
                if value.size == 1:
                    value = float(value[0])
            if key == "stress" and isinstance(value, np.ndarray) and value.ndim == 3:
                value = value[0]
            out[key] = value
        return out

    def predict(self, graphs):
        """Inference API aligned with CHGNet / MatterSim."""
        if isinstance(graphs, list):
            results = []
            for graph in graphs:
                result = self.forward(
                    {"graph": graph},
                    return_loss=False,
                    return_prediction=True,
                )
                results.append(self._prediction_to_numpy(result["pred_dict"]))
            return results

        result = self.forward(
            {"graph": graphs},
            return_loss=False,
            return_prediction=True,
        )
        return self._prediction_to_numpy(result["pred_dict"])
