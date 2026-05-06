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

"""
SGEQUI 训练包装器：将 EquivariantDiffusionModel 适配到 BaseTrainer 的训练流程。
"""

from __future__ import annotations

from typing import Dict, Any

import paddle
import paddle.nn as nn

from ppmat.models.sgequidiff.diffusion_model import (
    EquivariantDiffusionModel,
    EquivariantDiffusionModelConfig,
)
from ppmat.models.sgequidiff.non_equivariant_drift_modules import (
    GNNConfig,
    CSPNetConfig,
)


class SGEQUITrainingWrapper(nn.Layer):
    """
    将 EquivariantDiffusionModel 适配到 BaseTrainer 的训练流程。
    """

    def __init__(
        self,
        model: EquivariantDiffusionModel = None,
        num_wn_lattice_translations: int = 5,
        noise_scheduler_num_monte_carlo_samples: int = 10_000,
        num_timesteps: int = 1000,
        sigma_min: float = 0.002,
        sigma_max: float = 0.5,
        time_emb_dim: int = 256,
        model_type: str = "mlp",
        num_plane_wave_freqs: int = 64,
        subsample_group_operations: bool = False,
        gnn_config: Any = None,
        cspnet_config: Any = None,
    ):
        super().__init__()

        if model is not None:
            self.diffusion_model = model
        else:
            from ppmat.models.sgequidiff.global_vars import embedding_tools
            if embedding_tools is None:
                from ppmat.models.sgequidiff.embedding_utils import set_global_embedding_tools
                set_global_embedding_tools(element_embedding_json_path="cgcnn_atom_init.json")

            final_gnn_config = gnn_config
            if gnn_config is not None and isinstance(gnn_config, dict):
                final_gnn_config = GNNConfig(**gnn_config)

            final_cspnet_config = cspnet_config
            if cspnet_config is not None and isinstance(cspnet_config, dict):
                final_cspnet_config = CSPNetConfig(**cspnet_config)

            config = EquivariantDiffusionModelConfig(
                num_wn_lattice_translations=num_wn_lattice_translations,
                noise_scheduler_num_monte_carlo_samples=noise_scheduler_num_monte_carlo_samples,
                num_timesteps=num_timesteps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                time_emb_dim=time_emb_dim,
                model_type=model_type,
                num_plane_wave_freqs=num_plane_wave_freqs,
                subsample_group_operations=subsample_group_operations,
                gnn_config=final_gnn_config,
                cspnet_config=final_cspnet_config,
            )
            self.diffusion_model = EquivariantDiffusionModel(config)

        self.add_sublayer("diffusion_model", self.diffusion_model)

    def forward(self, batch_data: Dict) -> Dict:
        """
        训练入口，调用 compute_loss() 并返回 BaseTrainer 期望的格式。
        """
        from ppmat.models.sgequidiff.data_utils import lattice_params_to_matrix_paddle

        space_group_indices = batch_data["space_group_indices"]
        lattice_lengths = batch_data["lattice_lengths"]
        lattice_angles = batch_data["lattice_angles"]
        lattice_matrices = batch_data.get(
            "lattice_matrices",
            lattice_params_to_matrix_paddle(lattice_lengths, lattice_angles),
        )
        padded_element_indices = batch_data["element_indices"]
        padded_wyckoff_indices = batch_data["wyckoff_indices"]
        padded_wyckoff_shape_indices = batch_data["wyckoff_shape_indices"]
        padded_frac_coords = batch_data["frac_coords"]
        atoms_mask = batch_data["atoms_mask"]

        flat_mask = atoms_mask.reshape([-1])
        selected_indices = paddle.nonzero(flat_mask).reshape([-1])

        batch_size = padded_element_indices.shape[0]
        max_atoms = padded_element_indices.shape[1]
        selected_xtal_indices = paddle.floor_divide(selected_indices, max_atoms)
        n_atoms_per_xtal = paddle.zeros([batch_size], dtype="int64")
        n_atoms_per_xtal = paddle.scatter_nd_add(
            n_atoms_per_xtal,
            selected_xtal_indices.reshape([-1, 1]),
            paddle.ones_like(selected_xtal_indices, dtype="int64"),
        )

        element_indices = paddle.gather(
            padded_element_indices.reshape([-1]), selected_indices
        )
        wyckoff_indices = paddle.gather(
            padded_wyckoff_indices.reshape([-1]), selected_indices
        )
        wyckoff_shape_indices = paddle.gather(
            padded_wyckoff_shape_indices.reshape([-1]), selected_indices
        )
        asu_frac_coords = paddle.gather(
            padded_frac_coords.reshape([-1, 3]), selected_indices, axis=0
        )

        loss = self.diffusion_model.compute_loss(
            asu_frac_coords=asu_frac_coords,
            element_indices=element_indices,
            wyckoff_indices=wyckoff_indices,
            space_group_indices=space_group_indices,
            n_atoms_per_xtal=n_atoms_per_xtal,
            wyckoff_shape_indices=wyckoff_shape_indices,
            lattice_matrices=lattice_matrices,
            lattice_lengths=lattice_lengths,
            lattice_angles=lattice_angles,
        )
        return {"loss_dict": {"loss": loss}}


def get_grad_norm(parameters, norm_type: float = 2.0) -> paddle.Tensor:
    """计算参数的梯度范数。"""
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return paddle.to_tensor(0.0)

    norm_type = float(norm_type)

    total_norm = paddle.norm(
        paddle.stack([p.grad.detach().norm(norm_type) for p in parameters]),
        norm_type,
    )

    return total_norm
