"""主扩散模型：EquivariantDiffusionModel、NoiseScheduler。"""
import dataclasses
import math
import os
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from tqdm import tqdm

import ppmat.models.sgequidiff.global_vars as global_vars
from ppmat.models.sgequidiff.constants import MAX_WYCKOFF_SITES
from ppmat.models.sgequidiff.crystal_utils import uniformly_sample_point_in_asu_wyckoff_site
from ppmat.models.sgequidiff.diffusion_utils import (
    get_wyckoff_shape_hull_equations,
    wrap_frac_coords_into_asu,
    d_log_p_asu_wrapped_normal,
    get_space_group_ops_and_conventional_atoms,
    get_wyckoff_projected_gaussian_noise,
)
from ppmat.models.sgequidiff.non_equivariant_drift_modules import (
    FourierTimeEmbeddings,
    TorusMLP,
    GNN,
    GNNConfig,
    CSPNet,
    CSPNetConfig,
)
from ppmat.models.sgequidiff.scatter_utils import safe_scatter as paddle_scatter


@dataclasses.dataclass
class EquivariantDiffusionModelConfig:
    """扩散模型超参数配置。"""
    num_wn_lattice_translations: int = 5
    noise_scheduler_num_monte_carlo_samples: int = 10_000
    num_timesteps: int = 1000
    sigma_min: float = 0.002
    sigma_max: float = 0.5
    time_emb_dim: int = 256
    model_type: str = "mlp"  # ["mlp", "gnn", "cspnet"]
    num_plane_wave_freqs: int = 64
    subsample_group_operations: bool = False

    gnn_config: Optional[GNNConfig] = None
    cspnet_config: Optional[CSPNetConfig] = None

class EquivariantDiffusionModel(nn.Layer):
    """空间群等变扩散模型，在非对称单元中建模晶体坐标。"""

    def __init__(self, config: EquivariantDiffusionModelConfig):
        self.validate_config(config)
        super().__init__()
        self.config = config
        self.num_wn_lattice_translations = config.num_wn_lattice_translations

        self.noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            num_lattice_translations=config.num_wn_lattice_translations,
            num_monte_carlo_samples=config.noise_scheduler_num_monte_carlo_samples,
        )
        self.time_embedder = FourierTimeEmbeddings(dim=config.time_emb_dim)
        self.non_equivariant_drift_model = self.get_non_equivariant_drift_module(
            config, self.time_embedder
        )

        (
            padded_hull_equations,
            mask_padded_hull_equations,
        ) = get_wyckoff_shape_hull_equations()

        self.register_buffer("padded_hull_equations", padded_hull_equations)
        self.register_buffer("padded_hull_equations_mask", mask_padded_hull_equations)

        self._wyckoff_shape_decomposition_dict = None

    @property
    def wyckoff_shape_decomposition_dict(self) -> dict:
        """懒加载 wyckoff_shape_decomposition.pkl。"""
        if self._wyckoff_shape_decomposition_dict is None:
            import pickle
            from ppmat.models.sgequidiff.global_vars import SHAPE_DECOMP_DICT_PATH, _ensure_wyckoff_shape_decomp
            _ensure_wyckoff_shape_decomp()
            with open(str(SHAPE_DECOMP_DICT_PATH), "rb") as f:
                self._wyckoff_shape_decomposition_dict = pickle.load(f)
        return self._wyckoff_shape_decomposition_dict

    def compute_loss(
        self,
        asu_frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        wyckoff_shape_indices: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
    ) -> paddle.Tensor:
        """计算训练损失（score matching）。"""
        sampled_timesteps = self.noise_scheduler.uniform_sample_timestep(
            batch_size=space_group_indices.shape[0]
        )
        time_embeddings = self.time_embedder(sampled_timesteps.cast(paddle.float32))

        sampled_timesteps = sampled_timesteps.repeat_interleave(n_atoms_per_xtal, axis=0)
        time_embeddings = time_embeddings.repeat_interleave(n_atoms_per_xtal, axis=0)
        sigmas = self.noise_scheduler.sigmas[sampled_timesteps].detach()

        _sg_per_asu_atom = space_group_indices.repeat_interleave(n_atoms_per_xtal, axis=0)

        with paddle.no_grad():
            projected_noise = get_wyckoff_projected_gaussian_noise(
                space_group_indices,
                wyckoff_indices,
                wyckoff_shape_indices,
                n_atoms_per_xtal,
                sigmas.unsqueeze(1),
            )
            noisy_asu_frac_coords = asu_frac_coords.detach() + projected_noise

            (
                noisy_asu_frac_coords,
                wyckoff_shape_indices,
            ) = wrap_frac_coords_into_asu(
                noisy_asu_frac_coords,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                self.padded_hull_equations[_sg_per_asu_atom, wyckoff_indices].clone(),
                self.padded_hull_equations_mask[_sg_per_asu_atom, wyckoff_indices].clone(),
            )

            (
                _,
                _,
                _,
                map_conventional_to_asu_atom,
                conventional_wyckoff_indices,
                conventional_element_indices,
                conventional_frac_coords,
                unique_non_overlapping_atom_indices,
            ) = get_space_group_ops_and_conventional_atoms(
                asu_frac_coords,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
            )
            map_unique_conventional_to_asu = map_conventional_to_asu_atom[
                unique_non_overlapping_atom_indices
            ]

            ground_truth_scores = d_log_p_asu_wrapped_normal(
                noisy_asu_frac_coords,
                conventional_frac_coords,
                map_unique_conventional_to_asu,
                self.num_wn_lattice_translations,
                sigmas,
            )

        # Avoid Paddle's AssignOutGradNode issue with shared computation graphs
        noisy_asu_frac_coords = noisy_asu_frac_coords.detach()
        wyckoff_shape_indices = wyckoff_shape_indices.detach()
        ground_truth_scores = ground_truth_scores.detach()
        _sg_per_asu_atom = _sg_per_asu_atom.detach()
        sampled_timesteps = sampled_timesteps.detach()

        predicted_scores = self.predict_equivariant_vectors(
            time_embeddings,
            noisy_asu_frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
            lattice_matrices=lattice_matrices,
            lattice_lengths=lattice_lengths,
            lattice_angles=lattice_angles,
        )

        score_norms = self.noise_scheduler.sigma_norms[
            _sg_per_asu_atom, wyckoff_indices, sampled_timesteps
        ]
        loss = F.mse_loss(
            predicted_scores,
            ground_truth_scores / (score_norms.unsqueeze(1) + 1e-8),
        )
        return loss

    @paddle.no_grad()
    def sample(
        self,
        wyckoff_indices: paddle.Tensor,
        element_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        snr: float = 0.4,
        max_step_size: float = 1e6,
    ) -> paddle.Tensor:
        """Variance-exploding Predictor-Corrector SDE 采样。"""
        num_asu_atoms: int = wyckoff_indices.shape[0]
        num_crystals: int = space_group_indices.shape[0]
        time_start: int = self.noise_scheduler.num_timesteps

        space_group_indices_per_atom = space_group_indices.repeat_interleave(n_atoms_per_xtal, axis=0)
        map_atom_to_xtal = paddle.arange(num_crystals).repeat_interleave(n_atoms_per_xtal, axis=0)

        (
            x_T,
            wyckoff_shape_indices,
        ) = uniformly_sample_point_in_asu_wyckoff_site(
            space_group_numbers=[
                str(1 + int(sg_idx)) for sg_idx in space_group_indices_per_atom.tolist()
            ],
            wyckoff_letters=[
                chr(97 + int(idx)) if int(idx) <= 25 else chr(39 + int(idx))
                for idx in wyckoff_indices.tolist()
            ],
            dictionary_of_wyckoffs_in_asu=global_vars.asu_wyckoff_dict,
            dictionary_of_wyckoff_shape_decompositions=self.wyckoff_shape_decomposition_dict,
            n_samples_per_wyckoff=1,
            return_sampled_wyckoff_shape_indices=True,
        )
        x_T = x_T.squeeze(1)
        wyckoff_shape_indices = wyckoff_shape_indices.squeeze(1)

        wyckoff_dims = global_vars.wyckoff_dimension_tensor[
            space_group_indices_per_atom, wyckoff_indices
        ]
        noise_projection_matrices = global_vars.noise_projection_matrices[
            space_group_indices_per_atom,
            wyckoff_indices,
            wyckoff_shape_indices,
        ]

        x_t_plus_1 = x_T
        for t in tqdm(range(time_start - 1, 0, -1)):
            time_embeddings = self.time_embedder(
                paddle.to_tensor([t], dtype=paddle.float32)
            ).expand([num_asu_atoms, -1])

            sigma_norm_t_plus_1 = self.noise_scheduler.sigma_norms[
                space_group_indices_per_atom, wyckoff_indices, t + 1
            ]
            sigma_norm_t = self.noise_scheduler.sigma_norms[
                space_group_indices_per_atom, wyckoff_indices, t
            ]

            predicted_score = sigma_norm_t_plus_1.unsqueeze(1) * self.predict_equivariant_vectors(
                time_embeddings,
                x_t_plus_1,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
            )
            predicted_score = paddle.bmm(
                predicted_score.unsqueeze(1), noise_projection_matrices
            ).reshape([-1, 3])

            sigma_sq_diff = (
                self.noise_scheduler.sigmas[t + 1] ** 2
                - self.noise_scheduler.sigmas[t] ** 2
            )
            noise = get_wyckoff_projected_gaussian_noise(
                space_group_indices, wyckoff_indices, wyckoff_shape_indices,
                n_atoms_per_xtal, 1.0,
            )
            x_t = x_t_plus_1 + sigma_sq_diff * predicted_score
            x_t = x_t + paddle.sqrt(sigma_sq_diff) * noise

            predicted_score = sigma_norm_t.unsqueeze(1) * self.predict_equivariant_vectors(
                time_embeddings,
                x_t,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
            )
            predicted_score = paddle.bmm(
                predicted_score.unsqueeze(1), noise_projection_matrices
            ).reshape([-1, 3])

            noise = get_wyckoff_projected_gaussian_noise(
                space_group_indices, wyckoff_indices, wyckoff_shape_indices,
                n_atoms_per_xtal, 1.0,
            )
            noise_norm = ((noise ** 2).sum(axis=-1)).sqrt().mean()
            grad_norm = ((predicted_score ** 2).sum(axis=-1)).sqrt().mean()
            step_size = 2 * (snr * noise_norm / (grad_norm + 1e-12)) ** 2

            step_size = paddle.where(noise == 0.0, paddle.zeros_like(noise), step_size * paddle.ones_like(noise))
            step_size = paddle.nan_to_num(step_size, nan=0.0, posinf=max_step_size, neginf=-max_step_size)

            x_t = x_t + step_size * predicted_score + paddle.sqrt(2 * step_size) * noise

            x_t = self.project_point_onto_wyckoff_shape(
                x_t,
                space_group_indices_per_atom,
                wyckoff_indices,
                wyckoff_shape_indices,
                wyckoff_dims,
            )
            x_t_plus_1 = x_t

        x_final = x_t_plus_1 % 1.0
        x_final, _ = wrap_frac_coords_into_asu(
            x_final,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
            self.padded_hull_equations[space_group_indices_per_atom, wyckoff_indices],
            self.padded_hull_equations_mask[space_group_indices_per_atom, wyckoff_indices],
        )
        return x_final.unsqueeze(0)

    def forward(
        self,
        t: paddle.Tensor,
        states: tuple,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        noise_projection_matrices: paddle.Tensor,
        divergence_type: str = "hutchinson",
    ) -> tuple:
        """ODE 函数 (velocity, divergence)，用于 RK4 积分。"""
        frac_coords = states[0]
        n_asu_atoms = frac_coords.shape[0]
        t_val = float(t.item())

        sg_per_atom = space_group_indices.repeat_interleave(n_atoms_per_xtal, axis=0)
        sigma_norm_t = self.noise_scheduler.get_interpolated_sigma_norm_t(
            t, sg_per_atom, wyckoff_indices
        ).unsqueeze(1)

        frac_coords_stop = frac_coords.detach()
        frac_coords_stop.stop_gradient = False

        time_embeddings = self.time_embedder(
            t.reshape([1])
        ).detach().expand([n_asu_atoms, -1])

        d_sigma_sq_dt = self.noise_scheduler.d_sigma_sq_dt(t_val)

        dx_dt = (
            0.5 * d_sigma_sq_dt * sigma_norm_t
            * self.predict_equivariant_vectors(
                time_embeddings,
                frac_coords_stop,
                element_indices,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                lattice_matrices=lattice_matrices,
                lattice_lengths=lattice_lengths,
                lattice_angles=lattice_angles,
                differentiate_graph_construction=True,
            )
        )

        dx_dt = paddle.bmm(dx_dt.unsqueeze(1), noise_projection_matrices).squeeze(1)

        if divergence_type == "full":
            divergence = paddle.zeros([n_asu_atoms], dtype=paddle.float32)
            for d in range(3):
                grad_d = paddle.grad(
                    outputs=[dx_dt[:, d].sum()],
                    inputs=[frac_coords_stop],
                    create_graph=False,
                    retain_graph=(d < 2),
                )[0]
                if grad_d is not None:
                    divergence += grad_d[:, d]
        else:
            z = paddle.randn_like(frac_coords_stop)
            grad = paddle.grad(
                outputs=[(dx_dt * z.detach()).sum()],
                inputs=[frac_coords_stop],
                create_graph=False,
                retain_graph=False,
            )[0]
            divergence = (grad * z.detach()).sum(axis=-1) if grad is not None else paddle.zeros([n_asu_atoms])

        dx_dt = dx_dt.detach()
        divergence = divergence.detach()
        return dx_dt, divergence

    def get_log_likelihood(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        divergence_type: str = "hutchinson",
        num_ode_solver_steps: int = 500,
    ) -> paddle.Tensor:
        """通过概率流 ODE 计算对数似然（RK4 数值积分）。"""
        sg_per_asu_atom = space_group_indices.repeat_interleave(n_atoms_per_xtal, axis=0)

        with paddle.no_grad():
            frac_coords, wyckoff_shape_indices = wrap_frac_coords_into_asu(
                frac_coords,
                wyckoff_indices,
                space_group_indices,
                n_atoms_per_xtal,
                self.padded_hull_equations[sg_per_asu_atom, wyckoff_indices],
                self.padded_hull_equations_mask[sg_per_asu_atom, wyckoff_indices],
            )

        noise_projection_matrices = global_vars.noise_projection_matrices[
            sg_per_asu_atom, wyckoff_indices, wyckoff_shape_indices
        ]

        T = float(self.noise_scheduler.num_timesteps)
        t0 = 0.0
        dt = (T - t0) / num_ode_solver_steps

        x = frac_coords.clone()
        logp = paddle.zeros([frac_coords.shape[0]], dtype=paddle.float32)

        for step in range(num_ode_solver_steps):
            t_val = t0 + step * dt
            t = paddle.to_tensor([t_val], dtype=paddle.float32)

            def _f(x_in, t_in):
                return self.forward(
                    t_in, (x_in, None),
                    element_indices, wyckoff_indices, space_group_indices, n_atoms_per_xtal,
                    lattice_matrices, lattice_lengths, lattice_angles,
                    noise_projection_matrices, divergence_type,
                )

            k1_x, k1_l = _f(x, t)
            k2_x, k2_l = _f(
                x + 0.5 * dt * k1_x,
                t + paddle.to_tensor([0.5 * dt], dtype=paddle.float32),
            )
            k3_x, k3_l = _f(
                x + 0.5 * dt * k2_x,
                t + paddle.to_tensor([0.5 * dt], dtype=paddle.float32),
            )
            k4_x, k4_l = _f(
                x + dt * k3_x,
                t + paddle.to_tensor([dt], dtype=paddle.float32),
            )

            x = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            logp = logp + (dt / 6.0) * (k1_l + 2.0 * k2_l + 2.0 * k3_l + k4_l)
            x = x.detach()
            logp = logp.detach()

        logp_T = self.uniform_prior_log_prob(wyckoff_indices, space_group_indices, n_atoms_per_xtal)
        return logp + logp_T

    def predict_equivariant_vectors(
        self,
        time_embeddings: paddle.Tensor,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: Optional[paddle.Tensor] = None,
        lattice_lengths: Optional[paddle.Tensor] = None,
        lattice_angles: Optional[paddle.Tensor] = None,
        differentiate_graph_construction: bool = False,
    ) -> paddle.Tensor:
        """等变向量场：mean(A^-1 @ f(Ax + t))。"""
        frac_coords = frac_coords % 1.0

        (
            A_inv_ops,
            t_ops,
            inverse_indices,
            map_conventional_to_asu_atom,
            conventional_wyckoff_indices,
            conventional_element_indices,
            frac_coords_of_conv_atoms,
            unique_non_overlapping_atom_indices,
        ) = get_space_group_ops_and_conventional_atoms(
            frac_coords,
            element_indices,
            wyckoff_indices,
            space_group_indices,
            n_atoms_per_xtal,
        )

        map_asu_atom_to_xtal = paddle.arange(space_group_indices.shape[0]).repeat_interleave(
            n_atoms_per_xtal, axis=0
        )
        map_unique_conventional_to_asu = map_conventional_to_asu_atom[
            unique_non_overlapping_atom_indices
        ]

        n_conv_atoms_per_xtal = paddle_scatter(
            src=paddle.ones([conventional_wyckoff_indices.shape[0]], dtype=paddle.int64),
            index=map_asu_atom_to_xtal[map_unique_conventional_to_asu],
            dim=0,
            dim_size=space_group_indices.shape[0],
            reduce="sum",
        )

        non_equivariant_output = self.non_equivariant_drift_model(
            frac_coords=frac_coords_of_conv_atoms,
            element_indices=conventional_element_indices,
            n_atoms_per_xtal=n_conv_atoms_per_xtal,
            lattice_matrices=lattice_matrices,
            lattice_lengths=lattice_lengths,
            lattice_angles=lattice_angles,
            time_embeddings=time_embeddings[map_unique_conventional_to_asu],
        )

        src = paddle.bmm(
            non_equivariant_output[inverse_indices].unsqueeze(1),
            A_inv_ops,
        ).squeeze(1)

        vector_field = paddle_scatter(
            src=src,
            index=map_conventional_to_asu_atom,
            dim=0,
            dim_size=frac_coords.shape[0],
            reduce="mean",
        )
        return vector_field

    @paddle.no_grad()
    def uniform_prior_log_prob(
        self,
        wyckoff_indices: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
    ) -> paddle.Tensor:
        """均匀先验的对数概率。"""
        sg_per_atom = space_group_indices.repeat_interleave(n_atoms_per_xtal, axis=0)
        wyckoff_volumes = global_vars.wyckoff_shape_volumes[
            sg_per_atom, wyckoff_indices
        ].sum(axis=-1)
        return paddle.where(
            wyckoff_volumes == 0.0,
            paddle.zeros_like(wyckoff_volumes),
            paddle.log(1.0 / wyckoff_volumes),
        )

    @staticmethod
    def validate_config(config: EquivariantDiffusionModelConfig) -> None:
        assert (
            config.sigma_min > 0.0
            and isinstance(config.time_emb_dim, int)
            and config.time_emb_dim > 0
            and config.model_type in ["mlp", "gnn", "cspnet"]
        )

    @staticmethod
    def get_non_equivariant_drift_module(
        config: EquivariantDiffusionModelConfig,
        time_embedder: FourierTimeEmbeddings,
    ) -> nn.Layer:
        """根据配置创建非等变漂移模块。"""
        if config.model_type == "mlp":
            return TorusMLP(time_embedder, config.num_plane_wave_freqs)
        elif config.model_type == "gnn":
            return GNN(config.gnn_config, time_embedder)
        elif config.model_type == "cspnet":
            return CSPNet(config.cspnet_config, time_embedder)
        else:
            raise AttributeError(f"Unknown model_type: {config.model_type}")

    @staticmethod
    def project_point_onto_wyckoff_shape(
        points: paddle.Tensor,
        space_group_indices_repeated: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
        wyckoff_shape_indices: paddle.Tensor,
        wyckoff_dims: paddle.Tensor,
    ) -> paddle.Tensor:
        """将点投影到 Wyckoff 子空间（1D->线, 2D->平面）。"""
        def project_to_lines(pts, p0s, dirs):
            dirs = dirs / (dirs.norm(axis=-1, keepdim=True) + 1e-12)
            v = pts - p0s
            t = (v * dirs).sum(axis=-1, keepdim=True)
            return p0s + t * dirs

        def project_to_planes(pts, p0s, normals):
            normals = normals / (normals.norm(axis=-1, keepdim=True) + 1e-12)
            v = pts - p0s
            dist = (v * normals).sum(axis=-1, keepdim=True)
            return pts - dist * normals

        wyckoff_dim_is_1 = (wyckoff_dims == 1)
        wyckoff_dim_is_2 = (wyckoff_dims == 2)

        if wyckoff_dim_is_1.any():
            mask = wyckoff_dim_is_1
            pts_1d = project_to_lines(
                points[mask],
                global_vars.point_per_1d_wyckoff_line[
                    space_group_indices_repeated[mask],
                    wyckoff_indices[mask],
                    wyckoff_shape_indices[mask],
                ],
                global_vars.line_directions_of_1d_wyckoffs[
                    space_group_indices_repeated[mask],
                    wyckoff_indices[mask],
                    wyckoff_shape_indices[mask],
                ],
            )
            mask_idx = paddle.where(mask)[0]
            for i, idx in enumerate(mask_idx.tolist()):
                points[idx] = pts_1d[i]

        if wyckoff_dim_is_2.any():
            mask = wyckoff_dim_is_2
            pts_2d = project_to_planes(
                points[mask],
                global_vars.point_per_2d_wyckoff_plane[
                    space_group_indices_repeated[mask],
                    wyckoff_indices[mask],
                    wyckoff_shape_indices[mask],
                ],
                global_vars.plane_normals_of_2d_wyckoffs[
                    space_group_indices_repeated[mask],
                    wyckoff_indices[mask],
                    wyckoff_shape_indices[mask],
                ],
            )
            mask_idx = paddle.where(mask)[0]
            for i, idx in enumerate(mask_idx.tolist()):
                points[idx] = pts_2d[i]

        return points

class NoiseScheduler(nn.Layer):
    """指数噪声调度器。"""

    def __init__(
        self,
        num_timesteps: int,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        sigma_norm_type: str = "asu_wrapped",
        num_lattice_translations: int = 5,
        num_monte_carlo_samples: int = 10_000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_lattice_translations = num_lattice_translations

        sigmas = paddle.to_tensor(
            np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), num_timesteps)),
            dtype=paddle.float32,
        )

        if sigma_norm_type == "unwrapped":
            _sigma_norms = self._sigma_norm_unwrapped(sigmas)
        elif sigma_norm_type == "asu_wrapped":
            _sigma_norms = self._sigma_norm_asu_wrapped(sigmas, num_monte_carlo_samples)
        else:
            raise AttributeError(f"Unknown sigma_norm_type: {sigma_norm_type}")

        self.register_buffer(
            "sigmas",
            paddle.concat([paddle.zeros([1]), sigmas], axis=0),
        )
        self.register_buffer(
            "sigma_norms",
            paddle.concat(
                [paddle.ones([230, MAX_WYCKOFF_SITES, 1]), _sigma_norms],
                axis=-1,
            ),
        )

    def uniform_sample_timestep(self, batch_size: int) -> paddle.Tensor:
        """在 [1, num_timesteps] 均匀采样整数时间步。"""
        return paddle.randint(
            low=1,
            high=self.num_timesteps + 1,
            shape=[batch_size],
            dtype=paddle.int64,
        )

    def d_sigma_sq_dt(self, t: float) -> float:
        """d(sigma^2)/dt，用于 ODE 采样。"""
        t = max(t, 1e-5)
        T = float(self.num_timesteps)
        sigma_ratio = self.sigma_max / self.sigma_min
        return 2 * (self.sigma_min / (T - 1)) * math.log(sigma_ratio) * (sigma_ratio ** ((t - 1) / (T - 1)))

    @paddle.no_grad()
    def _sigma_norm_unwrapped(self, sigmas: paddle.Tensor) -> paddle.Tensor:
        """返回 1/sigma。"""
        sigma_norms = 1.0 / sigmas
        return sigma_norms[None, None, :].expand([230, MAX_WYCKOFF_SITES, -1])

    @paddle.no_grad()
    def _sigma_norm_asu_wrapped(
        self,
        sigmas: paddle.Tensor,
        num_monte_carlo_samples: int = 10_000,
    ) -> paddle.Tensor:
        """用 Monte Carlo 估计 ASU 包裹正态分布的期望分数 L2 范数。"""
        from ppmat.models.sgequidiff.global_vars import DATA_DIRECTORY, _ensure_wyckoff_shape_decomp
        from ppmat.models.sgequidiff.crystal_utils import uniformly_sample_point_in_asu_wyckoff_site

        num_timesteps = sigmas.shape[0]
        num_lattice_translations = self.num_lattice_translations

        cache_filename = os.path.join(
            str(DATA_DIRECTORY),
            f"expected_score_norms_minSigma{float(sigmas[0]):0.3f}"
            f"_maxSigma{float(sigmas[-1]):0.3f}"
            f"_T{num_timesteps}_{num_monte_carlo_samples}MCsamples"
            f"_{num_lattice_translations}LatticeTranslations.pdparams",
        )
        print(f"Checking cache: {cache_filename}")
        print(f"Cache exists: {os.path.exists(cache_filename)}")
        if os.path.exists(cache_filename):
            print(f"Loading sigma norms from: {cache_filename}")
            sigma_norms = paddle.load(cache_filename)
            print(f"Successfully loaded sigma_norms with shape: {sigma_norms.shape}")
            return sigma_norms
        else:
            print(f"Cache not found, computing sigma_norms...")

        _ensure_wyckoff_shape_decomp()
        import pickle
        with open(global_vars.SHAPE_DECOMP_DICT_PATH, "rb") as f:
            wyckoff_shape_decomp_dict = pickle.load(f)

        asu_wyckoff_dict = global_vars.asu_wyckoff_dict
        sigma_norms = paddle.zeros([230, MAX_WYCKOFF_SITES, num_timesteps], dtype=paddle.float32)

        for sg_num in tqdm(range(230, 0, -1)):
            sg_dict = asu_wyckoff_dict[str(sg_num)]
            wyckoff_letters = sg_dict["ordered_wyckoff_letters"]

            (
                x0s,
                wsi,
            ) = uniformly_sample_point_in_asu_wyckoff_site(
                space_group_numbers=[str(sg_num)] * len(wyckoff_letters),
                wyckoff_letters=wyckoff_letters,
                dictionary_of_wyckoffs_in_asu=asu_wyckoff_dict,
                dictionary_of_wyckoff_shape_decompositions=wyckoff_shape_decomp_dict,
                n_samples_per_wyckoff=num_monte_carlo_samples,
                return_sampled_wyckoff_shape_indices=True,
            )

            space_group_idx = paddle.to_tensor([sg_num - 1], dtype=paddle.int64)

            for i, letter in enumerate(wyckoff_letters):
                _wyckoff_idx = paddle.to_tensor([i], dtype=paddle.int64)
                _wyckoff_idx_expanded = _wyckoff_idx.expand([num_monte_carlo_samples])

                (
                    _,
                    _,
                    _,
                    map_conv_to_asu,
                    conv_wp_idxs,
                    _,
                    orbited_x,
                    unique_indices,
                ) = get_space_group_ops_and_conventional_atoms(
                    x0s[i],
                    paddle.zeros_like(_wyckoff_idx_expanded),
                    _wyckoff_idx_expanded,
                    space_group_idx,
                    n_atoms_per_xtal=paddle.to_tensor(
                        [num_monte_carlo_samples], dtype=paddle.int64
                    ),
                )
                map_unique_conv_to_asu = map_conv_to_asu[unique_indices]

                _chunks = min(200, num_timesteps)
                for sigma_idxs in paddle.chunk(
                    paddle.arange(num_timesteps), chunks=_chunks
                ):
                    batch_sigmas = sigmas[sigma_idxs]

                    norms = []
                    for t_idx, sigma in enumerate(batch_sigmas.tolist()):
                        noise = paddle.randn([num_monte_carlo_samples, 3]) * sigma
                        xts = x0s[i] + noise

                        scores = d_log_p_asu_wrapped_normal(
                            xts,
                            orbited_x,
                            map_unique_conv_to_asu,
                            num_lattice_translations,
                            sigma,
                        )
                        norm_t = ((scores ** 2).sum(axis=-1)).sqrt().mean()
                        norms.append(float(norm_t.item()))

                    sigma_norms[sg_num - 1, i, sigma_idxs] = paddle.to_tensor(
                        norms, dtype=paddle.float32
                    )

        paddle.save(sigma_norms, cache_filename)
        print(f"Saved sigma norms to: {cache_filename}")
        return sigma_norms

    @paddle.no_grad()
    def get_interpolated_sigma_norm_t(
        self,
        t: paddle.Tensor,
        space_group_indices: paddle.Tensor,
        wyckoff_indices: paddle.Tensor,
    ) -> paddle.Tensor:
        """在实数时间 t 处插值 sigma_norm。"""
        t_val = float(t.item())
        assert 0.0 <= t_val <= self.num_timesteps

        t_below = int(math.floor(t_val))
        t_above = int(math.ceil(t_val))
        if t_below == t_above:
            if t_below == 0:
                t_below, t_above = 0, 1
            elif t_below == self.num_timesteps:
                t_above = self.num_timesteps
                t_below = self.num_timesteps - 1
            else:
                t_above = t_below + 1

        p = (t_val - t_below) / max(t_above - t_below, 1)
        sn_below = self.sigma_norms[space_group_indices, wyckoff_indices, t_below]
        sn_above = self.sigma_norms[space_group_indices, wyckoff_indices, t_above]
        return (1 - p) * sn_below + p * sn_above
