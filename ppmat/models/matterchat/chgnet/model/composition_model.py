# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import collections
from typing import TYPE_CHECKING, Sequence

import numpy as np
import paddle
from pymatgen.core import Structure
from paddle import Tensor, nn

from ppmat.models.matterchat.chgnet.model.functions import GatedMLP, find_activation

if TYPE_CHECKING:
    from ppmat.models.matterchat.chgnet.graph.crystalgraph import CrystalGraph


class CompositionModel(nn.Layer):
    """A simple FC model that takes in a chemical composition (no structure info)
    and outputs energy.
    """

    def __init__(
        self,
        atom_fea_dim: int = 64,
        activation: str = "silu",
        is_intensive: bool = True,
        max_num_elements: int = 94,
    ) -> None:
        """Initialize a CompositionModel."""
        super().__init__()
        self.is_intensive = is_intensive
        self.max_num_elements = max_num_elements
        self.fc1 = nn.Linear(max_num_elements, atom_fea_dim)
        self.activation = find_activation(activation)
        self.gated_mlp = GatedMLP(
            input_dim=atom_fea_dim,
            output_dim=atom_fea_dim,
            hidden_dim=atom_fea_dim,
            activation=activation,
        )
        self.fc2 = nn.Linear(atom_fea_dim, 1)

    def _get_energy(self, composition_feas: Tensor) -> Tensor:
        """Predict the energy given composition encoding."""
        composition_feas = self.activation(self.fc1(composition_feas))
        composition_feas = composition_feas + self.gated_mlp(composition_feas)
        return self.fc2(composition_feas).reshape([-1])

    def forward(self, graphs: list[CrystalGraph]) -> Tensor:
        """Get the energy of a list of CrystalGraphs as Tensor."""
        composition_feas = self._assemble_graphs(graphs)
        return self._get_energy(composition_feas)

    def _assemble_graphs(self, graphs: list[CrystalGraph]):
        """Assemble a list of graphs into one-hot composition encodings."""
        composition_feas = []
        for _graph_idx, graph in enumerate(graphs):
            composition_fea = paddle.bincount(
                graph.atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                n_atom = graph.atomic_number.shape[0]
                composition_fea = composition_fea.cast(paddle.float32) / n_atom
            composition_feas.append(composition_fea)
        return paddle.stack(composition_feas, axis=0)


class AtomRef(nn.Layer):
    """A linear regression for elemental energy."""

    def __init__(self, is_intensive: bool = True, max_num_elements: int = 94) -> None:
        """Initialize an AtomRef model."""
        super().__init__()
        self.is_intensive = is_intensive
        self.max_num_elements = max_num_elements
        self.fc = nn.Linear(max_num_elements, 1, bias_attr=False)
        self.fitted = False

    def forward(self, graphs: list[CrystalGraph]):
        """Get the energy of a list of CrystalGraphs."""
        assert self.fitted is True, "composition model need to be fitted first!"
        composition_feas = self._assemble_graphs(graphs)
        return self._get_energy(composition_feas)

    def _get_energy(self, composition_feas: Tensor) -> Tensor:
        """Predict the energy given composition encoding."""
        return self.fc(composition_feas).reshape([-1])

    def fit(
        self,
        structures_or_graphs: Sequence[Structure | CrystalGraph],
        energies: Sequence[float],
    ) -> None:
        """Fit the model to a list of crystals and energies."""
        num_data = len(energies)
        composition_feas = paddle.zeros([num_data, self.max_num_elements])
        e = paddle.zeros([num_data])
        for index, (structure, energy) in enumerate(
            zip(structures_or_graphs, energies)
        ):
            if isinstance(structure, Structure):
                atomic_number = paddle.to_tensor(
                    [i.specie.Z for i in structure], dtype="int32"
                )
            else:
                atomic_number = structure.atomic_number
            composition_fea = paddle.bincount(
                atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                composition_fea = composition_fea.cast(paddle.float32) / atomic_number.shape[0]
            composition_feas[index, :] = composition_fea
            e[index] = energy

        self.feature_matrix = composition_feas.detach().numpy()
        self.energies = e.detach().numpy()
        state_dict = collections.OrderedDict()
        weight = (
            np.linalg.pinv(self.feature_matrix.T @ self.feature_matrix)
            @ self.feature_matrix.T
            @ self.energies
        )
        state_dict["weight"] = paddle.to_tensor(weight).reshape([94, 1])
        self.fc.load_state_dict(state_dict)
        self.fitted = True

    def _assemble_graphs(self, graphs: list[CrystalGraph]):
        """Assemble a list of graphs into one-hot composition encodings."""
        composition_feas = []
        for _graph_idx, graph in enumerate(graphs):
            composition_fea = paddle.bincount(
                graph.atomic_number - 1, minlength=self.max_num_elements
            )
            if self.is_intensive:
                n_atom = graph.atomic_number.shape[0]
                composition_fea = composition_fea.cast(paddle.float32) / n_atom
            composition_feas.append(composition_fea)
        return paddle.stack(composition_feas, axis=0).cast("float32")

    def initialize_from(self, dataset: str):
        """Initialize pre-fitted weights from a dataset."""
        if dataset in ["MPtrj", "MPtrj_e"]:
            self.initialize_from_MPtrj()
        elif dataset in ["MPF"]:
            self.initialize_from_MPF()
        else:
            raise NotImplementedError(f"{dataset=} not supported yet")

    def initialize_from_MPtrj(self):
        """Initialize pre-fitted weights from MPtrj dataset."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(
            [
                -3.4431,
                -0.1279,
                -2.8300,
                -3.4737,
                -7.4946,
                -8.2354,
                -8.1611,
                -8.3861,
                -5.7498,
                -0.0236,
                -1.7406,
                -1.6788,
                -4.2833,
                -6.2002,
                -6.1315,
                -5.8405,
                -3.8795,
                -0.0703,
                -1.5668,
                -3.4451,
                -7.0549,
                -9.1465,
                -9.2594,
                -9.3514,
                -8.9843,
                -8.0228,
                -6.4955,
                -5.6057,
                -3.4002,
                -0.9217,
                -3.2499,
                -4.9164,
                -4.7810,
                -5.0191,
                -3.3316,
                0.5130,
                -1.4043,
                -3.2175,
                -7.4994,
                -9.3816,
                -10.4386,
                -9.9539,
                -7.9555,
                -8.5440,
                -7.3245,
                -5.2771,
                -1.9014,
                -0.4034,
                -2.6002,
                -4.0054,
                -4.1156,
                -3.9928,
                -2.7003,
                2.2170,
                -1.9671,
                -3.7180,
                -6.8133,
                -7.3502,
                -6.0712,
                -6.1699,
                -5.1471,
                -6.1925,
                -11.5829,
                -15.8841,
                -5.9994,
                -6.0798,
                -5.9513,
                -6.0400,
                -5.9773,
                -2.5091,
                -6.0767,
                -10.6666,
                -11.8761,
                -11.8491,
                -10.7397,
                -9.6100,
                -8.4755,
                -6.2070,
                -3.0337,
                0.4726,
                -1.6425,
                -3.1295,
                -3.3328,
                -0.1221,
                -0.3448,
                -0.4364,
                -0.1661,
                -0.3680,
                -4.1869,
                -8.4233,
                -10.0467,
                -12.0953,
                -12.5228,
                -14.2530,
            ]
        ).reshape([94, 1])
        self.fc.load_state_dict(state_dict)
        self.is_intensive = True
        self.fitted = True

    def initialize_from_MPF(self):
        """Initialize pre-fitted weights from MPF dataset."""
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(
            [
                -3.4654e00,
                -6.2617e-01,
                -3.4622e00,
                -4.7758e00,
                -8.0362e00,
                -8.4038e00,
                -7.7681e00,
                -7.3892e00,
                -4.9472e00,
                -5.4833e00,
                -2.4783e00,
                -2.0202e00,
                -5.1548e00,
                -7.9121e00,
                -6.9135e00,
                -4.6228e00,
                -3.0155e00,
                -2.1285e00,
                -2.3174e00,
                -4.7595e00,
                -8.1742e00,
                -1.1421e01,
                -8.9229e00,
                -8.4901e00,
                -8.1664e00,
                -6.5826e00,
                -5.2614e00,
                -4.4844e00,
                -3.2737e00,
                -1.3498e00,
                -3.6264e00,
                -4.6727e00,
                -4.1316e00,
                -3.6755e00,
                -2.8030e00,
                6.4728e00,
                -2.2469e00,
                -4.2510e00,
                -1.0245e01,
                -1.1666e01,
                -1.1802e01,
                -8.6551e00,
                -9.3641e00,
                -7.5716e00,
                -5.6990e00,
                -4.9716e00,
                -1.8871e00,
                -6.7951e-01,
                -2.7488e00,
                -3.7945e00,
                -3.3883e00,
                -2.5588e00,
                -1.9621e00,
                9.9793e00,
                -2.5566e00,
                -4.8803e00,
                -8.8604e00,
                -9.0537e00,
                -7.9431e00,
                -8.1259e00,
                -6.3212e00,
                -8.3025e00,
                -1.2289e01,
                -1.7310e01,
                -7.5512e00,
                -8.1959e00,
                -8.3493e00,
                -7.2591e00,
                -8.4170e00,
                -3.3873e00,
                -7.6823e00,
                -1.2630e01,
                -1.3626e01,
                -9.5299e00,
                -1.1840e01,
                -9.7990e00,
                -7.5561e00,
                -5.4690e00,
                -2.6508e00,
                4.1746e-01,
                -2.3255e00,
                -3.4830e00,
                -3.1808e00,
                -1.6934e-02,
                -3.6191e-02,
                -1.0842e-02,
                1.3170e-02,
                -6.5371e-02,
                -5.4892e00,
                -1.0335e01,
                -1.1130e01,
                -1.4312e01,
                -1.4700e01,
                -1.5473e01,
            ]
        ).reshape([94, 1])
        self.fc.load_state_dict(state_dict)
        self.is_intensive = False
        self.fitted = True

    def initialize_from_numpy(self, file_name):
        """Initialize pre-fitted weights from numpy file."""
        atom_ref_np = np.load(file_name)
        state_dict = collections.OrderedDict()
        state_dict["weight"] = paddle.to_tensor(atom_ref_np).reshape([94, 1])
        self.fc.load_state_dict(state_dict)
        self.is_intensive = False
        self.fitted = True
