# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lattice sampler: SpaceGroupEncoder, TelescopingDiscreteLatticeSampler。"""
import dataclasses
import math
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Categorical

import ppmat.models.sgequidiff.global_vars as global_vars
from ppmat.models.sgequidiff.data_utils import lattice_transform_and_log_prob_mask
from ppmat.models.sgequidiff.non_equivariant_drift_modules import FourierLinear, Swish


class SpaceGroupEncoder(nn.Layer):
    def __init__(self, hidden_channels: int = 128, space_group_embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_vars.embedding_tools.space_group_embedding_length, hidden_channels),
            Swish(),
            nn.Linear(hidden_channels, space_group_embedding_dim),
            Swish(),
        )

    def forward(self, space_group_indices):
        return self.net(global_vars.embedding_tools.get_space_group_embedding(space_group_indices))

@dataclasses.dataclass
class LatticeSamplerConfig:
    input_dimension: int
    hidden_dimension: int
    num_hidden_layers: int = 1
    use_fourier_features: Optional[bool] = True
    max_fourier_frequency: Optional[float] = 64.0
    num_fourier_frequencies: Optional[int] = 16
    lattice_lengths_transform: str = "identity"
    model_type: str = "telescoping_discrete"
    min_lattice_length: float = 2.0
    max_lattice_length: float = 133.0
    min_lattice_angle: float = 60.0
    max_lattice_angle: float = 135.0
    gradient_attenuation_factor: float = 1.0
    n_bins: int = 100
    n_telescopes: int = 2
    lattice_length_bin_embedder_fourier_scale: float = 10.0
    lattice_angle_bin_embedder_fourier_scale: float = 10.0
    lattice_length_embedder_fourier_scale: float = 20.0
    lattice_angle_embedder_fourier_scale: float = 1.0
    lattice_param_dim: int = 112
    n_emb_layers: int = 4

class TelescopingDiscreteLatticeSampler(nn.Layer):
    def __init__(self, config: LatticeSamplerConfig):
        super().__init__()
        self.config = config
        self.MAX_LATTICE_LENGTH = config.max_lattice_length
        self.MIN_LATTICE_LENGTH = config.min_lattice_length
        self.MAX_LATTICE_ANGLE = config.max_lattice_angle
        self.MIN_LATTICE_ANGLE = config.min_lattice_angle
        self.length_transform = lambda x: x
        self.inv_length_transform = lambda x: x

        bravais_data = [lattice_transform_and_log_prob_mask(sg) for sg in range(1, 231)]
        bravais_length_transforms = paddle.stack([d[0] for d in bravais_data], axis=0)
        bravais_angle_transforms = paddle.stack([d[1] for d in bravais_data], axis=0)
        bravais_angle_offsets = paddle.stack([d[2] for d in bravais_data], axis=0)
        bravais_log_prob_masks = paddle.stack([d[3] for d in bravais_data], axis=0)
        self.register_buffer("bravais_length_transforms", bravais_length_transforms)
        self.register_buffer("bravais_angle_transforms", bravais_angle_transforms)
        self.register_buffer("bravais_angle_offsets", bravais_angle_offsets)
        self.register_buffer("bravais_log_prob_masks", bravais_log_prob_masks)

        self.n_bins = self.config.n_bins
        self.n_telescopes = self.config.n_telescopes
        self.min_bin_edge = -4.0
        self.max_bin_edge = 4.0

        self.space_group_encoder = SpaceGroupEncoder(
            hidden_channels=256,
            space_group_embedding_dim=self.config.input_dimension,
        )

        self.lattice_param_dim = self.config.lattice_param_dim

        self.lattice_length_embedder = nn.Sequential(
            FourierLinear(
                input_dim=1,
                num_fourier_frequencies=128,
                scale=self.config.lattice_length_embedder_fourier_scale,
                num_layers=self.config.n_emb_layers,
                output_dim=512,
                use_bias=True,
            ),
            nn.Linear(512, self.lattice_param_dim),
            nn.Silu(),
        )

        self.lattice_angle_embedder = nn.Sequential(
            FourierLinear(
                input_dim=1,
                num_fourier_frequencies=128,
                scale=self.config.lattice_angle_embedder_fourier_scale,
                num_layers=self.config.n_emb_layers,
                output_dim=256,
                use_bias=True,
            ),
            nn.Linear(256, self.lattice_param_dim),
            nn.Silu(),
        )

        self.length_bin_embedder = nn.Sequential(
            FourierLinear(
                input_dim=2,
                num_fourier_frequencies=128,
                scale=self.config.lattice_length_bin_embedder_fourier_scale,
                output_dim=256,
                num_layers=self.config.n_emb_layers,
                use_bias=True,
            ),
            nn.Linear(256, self.config.hidden_dimension),
            nn.Silu(),
        )

        self.angle_bin_embedder = nn.Sequential(
            FourierLinear(
                input_dim=2,
                num_fourier_frequencies=128,
                scale=self.config.lattice_angle_bin_embedder_fourier_scale,
                output_dim=256,
                num_layers=self.config.n_emb_layers,
                use_bias=True,
            ),
            nn.Linear(256, self.config.hidden_dimension),
            nn.Silu(),
        )

        self.bin_conditioning_info_dim = (
            self.config.input_dimension + 6 * self.lattice_param_dim + 6
        )

        self.bin_logit_head = nn.Sequential(
            nn.Linear(
                self.config.hidden_dimension + self.bin_conditioning_info_dim,
                self.config.hidden_dimension,
            ),
            nn.Silu(),
            nn.Linear(self.config.hidden_dimension, 6),
        )

        self.register_buffer("grid_pts", paddle.linspace(0, 1, self.n_bins + 1))


        angle_offsets = self.bravais_angle_offsets.clone()
        unconstrained_angle_mask = (angle_offsets == 0.0)
        angle_offsets[unconstrained_angle_mask] = self.MIN_LATTICE_ANGLE
        _normed_params = self._get_normed_lattice_parameters(
            self.MIN_LATTICE_LENGTH * paddle.ones_like(self.bravais_angle_offsets),
            angle_offsets,
            self.min_bin_edge,
            self.max_bin_edge,
            norm_gamma_separately=False,
        )
        _discretized_normed_params = self.get_discretized_normed_lattice_params(
            _normed_params, angle_offsets[:, :2]
        )
        _discretized_normed_angle_offsets = _discretized_normed_params[:, 3:]
        _discretized_normed_angle_offsets[unconstrained_angle_mask] = 0.0
        self.register_buffer(
            "discretized_normed_bravais_angle_offsets",
            _discretized_normed_angle_offsets,
        )

    @paddle.no_grad()
    def _get_normed_lattice_parameters(
        self,
        lattice_lengths,
        lattice_angles,
        min_normed_param: float = -1.0,
        max_normed_param: float = 1.0,
        norm_gamma_separately: bool = True,
    ):
        normed_param_range = max_normed_param - min_normed_param
        normed_lattice_lengths = (
            normed_param_range
            * (
                (self.length_transform(lattice_lengths) - self.length_transform(self.MIN_LATTICE_LENGTH))
                / (self.length_transform(self.MAX_LATTICE_LENGTH) - self.length_transform(self.MIN_LATTICE_LENGTH))
            )
            + min_normed_param
        )

        if norm_gamma_separately:
            min_gamma_angle, max_gamma_angle = self.get_valid_gamma_angle_interval(
                alpha_and_beta_angles=lattice_angles[:, :2]
            )
        else:
            min_gamma_angle = self.MIN_LATTICE_ANGLE
            max_gamma_angle = self.MAX_LATTICE_ANGLE

        normed_lattice_angles = paddle.concat(
            [
                normed_param_range
                * (
                    (lattice_angles[:, :2] - self.MIN_LATTICE_ANGLE)
                    / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
                )
                + min_normed_param,
                normed_param_range
                * (
                    (lattice_angles[:, -1] - min_gamma_angle)
                    / (max_gamma_angle - min_gamma_angle)
                ).unsqueeze(-1)
                + min_normed_param,
            ],
            axis=1,
        )
        return paddle.concat([normed_lattice_lengths, normed_lattice_angles], axis=1)

    def get_valid_gamma_angle_interval(self, alpha_and_beta_angles):
        cos_alpha = paddle.cos(alpha_and_beta_angles[:, 0] * math.pi / 180.0)
        cos_beta = paddle.cos(alpha_and_beta_angles[:, 1] * math.pi / 180.0)
        cos_alpha_sq = cos_alpha ** 2
        cos_beta_sq = cos_beta ** 2
        term1 = cos_alpha * cos_beta
        inner = 4 * cos_alpha_sq * cos_beta_sq - 4 * (cos_alpha_sq + cos_beta_sq - 1)
        inner = paddle.clip(inner, min=0.0)
        term2 = 0.5 * paddle.sqrt(inner)
        gamma_min = paddle.acos(paddle.clip(term1 + term2, min=-1.0, max=1.0)) * 180.0 / math.pi
        gamma_max = paddle.acos(paddle.clip(term1 - term2, min=-1.0, max=1.0)) * 180.0 / math.pi
        return (
            paddle.clip(gamma_min, min=self.MIN_LATTICE_ANGLE, max=self.MAX_LATTICE_ANGLE),
            paddle.clip(gamma_max, min=self.MIN_LATTICE_ANGLE, max=self.MAX_LATTICE_ANGLE),
        )

    @paddle.no_grad()
    def get_discretized_normed_lattice_params(self, normed_lattice_parameters, raw_alpha_and_beta_angles):
        batch_size = normed_lattice_parameters.shape[0]
        _batch_idxs = paddle.arange(batch_size)
        discretized_normed_lattice_parameters = paddle.zeros_like(normed_lattice_parameters)
        for i in range(6):
            min_bin_edge = self.min_bin_edge * paddle.ones([batch_size])
            max_bin_edge = self.max_bin_edge * paddle.ones([batch_size])
            x = normed_lattice_parameters[:, i]
            for j in range(self.n_telescopes):
                bin_edges = min_bin_edge.unsqueeze(-1) + (max_bin_edge - min_bin_edge).unsqueeze(-1) * self.grid_pts.unsqueeze(0)
                bins = paddle.stack([bin_edges[:, :-1], bin_edges[:, 1:]], axis=-1)
                normalized_x = (x - min_bin_edge) / (max_bin_edge - min_bin_edge)
                bin_idxs = paddle.bucketize(normalized_x, self.grid_pts[1:])
                bin_idxs = paddle.clip(bin_idxs, max=self.n_bins - 1)
                chosen_bins = bins[_batch_idxs, bin_idxs]
                min_bin_edge = chosen_bins[:, 0]
                max_bin_edge = chosen_bins[:, 1]
            discretized_normed_lattice_parameters[:, i] = paddle.mean(chosen_bins, axis=-1)

        min_gamma, max_gamma = self.get_valid_gamma_angle_interval(raw_alpha_and_beta_angles)
        min_normed_gamma = (self.max_bin_edge - self.min_bin_edge) * (
            (min_gamma - self.MIN_LATTICE_ANGLE) / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
        ) + self.min_bin_edge
        max_normed_gamma = (self.max_bin_edge - self.min_bin_edge) * (
            (max_gamma - self.MIN_LATTICE_ANGLE) / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
        ) + self.min_bin_edge
        gammas_lt_min = discretized_normed_lattice_parameters[:, -1] < min_normed_gamma
        gammas_gt_max = discretized_normed_lattice_parameters[:, -1] > max_normed_gamma

        if paddle.any(gammas_lt_min | gammas_gt_max):
            bin_edges = (self.max_bin_edge - self.min_bin_edge) * paddle.linspace(0, 1, self.n_telescopes * self.n_bins + 1) + self.min_bin_edge
            bin_midpoints = paddle.mean(paddle.stack([bin_edges[:-1], bin_edges[1:]], axis=-1), axis=-1)
            if paddle.any(gammas_lt_min):
                valid_bin_indices = paddle.argmax(
                    (bin_midpoints.unsqueeze(0) > min_normed_gamma[gammas_lt_min].unsqueeze(-1)).cast("int32"), axis=-1)
                discretized_normed_lattice_parameters[:, -1][gammas_lt_min] = bin_midpoints[valid_bin_indices]
            if paddle.any(gammas_gt_max):
                valid_bin_indices = paddle.argmax(
                    (bin_midpoints.unsqueeze(0) < max_normed_gamma[gammas_gt_max].unsqueeze(-1)).cast("int32"), axis=-1)
                discretized_normed_lattice_parameters[:, -1][gammas_gt_max] = bin_midpoints[valid_bin_indices]
        return discretized_normed_lattice_parameters

    @paddle.no_grad()
    def forward(self, space_group_indices):
        """
        Sample 6 lattice parameters constrained by the Bravais lattices.

        Args:
            space_group_indices: (batch_size,) int64, 0-indexed

        Returns:
            lengths: (batch_size, 3)
            angles: (batch_size, 3)
            log_pfs: (batch_size,)
            regularizer: (batch_size,)
        """
        batch_size = space_group_indices.shape[0]
        sg_features = self.space_group_encoder(space_group_indices)

        num_angles = 3
        num_lengths = 3
        num_lattice_parameters = num_angles + num_lengths

        lattice_params_transform = paddle.concat(
            (
                paddle.to_tensor(
                    [self.length_transform(self.MAX_LATTICE_LENGTH)
                     - self.length_transform(self.MIN_LATTICE_LENGTH)]
                ).expand([batch_size, num_lengths]),
                paddle.to_tensor(
                    [self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE]
                ).expand([batch_size, num_angles]),
            ),
            axis=1,
        )

        lattice_params_offset = paddle.concat(
            (
                paddle.to_tensor(
                    [self.length_transform(self.MIN_LATTICE_LENGTH)]
                ).expand([batch_size, num_lengths]),
                paddle.to_tensor(
                    [self.MIN_LATTICE_ANGLE]
                ).expand([batch_size, num_angles]),
            ),
            axis=1,
        )

        normed_lattice_parameters = paddle.zeros([batch_size, num_lattice_parameters])
        current_lattice_embedding = paddle.zeros(
            [batch_size, 6 * self.lattice_param_dim]
        )
        log_pfs = []
        lattice_mask = paddle.zeros([6])
        regularizer = paddle.zeros([batch_size])
        alpha_and_beta_angles = None
        bravais_length_transforms = self.bravais_length_transforms[space_group_indices]
        bravais_angle_transforms = self.bravais_angle_transforms[space_group_indices]
        discretized_normed_bravais_angle_offsets = (
            self.discretized_normed_bravais_angle_offsets[space_group_indices]
        )

        for i in range(num_lattice_parameters):
            if i > 0:
                if i < 4:
                    current_lattice_embedding[
                        :, self.lattice_param_dim * (i - 1):self.lattice_param_dim * i
                    ] = self.lattice_length_embedder(
                        normed_lattice_parameters[:, i - 1].unsqueeze(-1)
                    )
                else:
                    current_lattice_embedding[
                        :, self.lattice_param_dim * (i - 1):self.lattice_param_dim * i
                    ] = self.lattice_angle_embedder(
                        normed_lattice_parameters[:, i - 1].unsqueeze(-1)
                    )

            current_state_features = paddle.concat(
                [
                    sg_features,
                    current_lattice_embedding,
                    lattice_mask.unsqueeze(0).expand([batch_size, 6])
                ], axis=1
            )[:, None, :].expand([batch_size, self.n_bins, -1])

            if i == 5:
                alpha_and_beta_angles = (
                    (normed_lattice_parameters[:, 3:5] - self.min_bin_edge)
                    / (self.max_bin_edge - self.min_bin_edge)
                ) * lattice_params_transform[:, 3:5] + lattice_params_offset[:, 3:5]

            sample, log_prob = self._sample_and_log_prob(
                lattice_param_index=i,
                bin_embedder=self.length_bin_embedder if i < 3 else self.angle_bin_embedder,
                batch_size=batch_size,
                z=current_state_features,
                alpha_and_beta_angles=alpha_and_beta_angles,
                enforce_gamma_bounds=(i == (num_lattice_parameters - 1)),
            )

            normed_lattice_parameters[:, i] = sample
            lattice_mask[i] = 1.0
            log_pfs.append(log_prob)

            # Apply Bravais lattice constraints
            if i < 3:
                normed_lattice_parameters[:, :3] = paddle.bmm(
                    normed_lattice_parameters[:, :3].unsqueeze(1),
                    bravais_length_transforms
                ).squeeze(1)
            else:
                normed_lattice_parameters[:, 3:] = paddle.bmm(
                    normed_lattice_parameters[:, 3:].unsqueeze(1),
                    bravais_angle_transforms
                ).squeeze(1) + discretized_normed_bravais_angle_offsets

        log_pfs = paddle.stack(log_pfs, axis=1)

        lattice_parameters = (
            (normed_lattice_parameters - self.min_bin_edge)
            / (self.max_bin_edge - self.min_bin_edge)
        ) * lattice_params_transform + lattice_params_offset

        lengths = self.inv_length_transform(lattice_parameters[:, :num_lengths])
        angles = lattice_parameters[:, num_lengths:]

        log_pf_masks = self.bravais_log_prob_masks[space_group_indices]
        log_pfs = (log_pfs * log_pf_masks).sum(axis=1)

        if self.config.gradient_attenuation_factor != 1.0:
            log_pfs_detach = log_pfs.detach()
            log_pfs = (
                self.config.gradient_attenuation_factor * log_pfs
                - self.config.gradient_attenuation_factor * log_pfs_detach
                + log_pfs_detach
            )

        return lengths, angles, log_pfs, regularizer

    def _sample_and_log_prob(
        self,
        lattice_param_index: int,
        bin_embedder,
        batch_size: int,
        z=None,
        x=None,
        alpha_and_beta_angles=None,
        enforce_gamma_bounds: bool = False,
    ):
        assert 0 <= lattice_param_index < 6
        if z is None:
            z = paddle.zeros([batch_size, self.n_bins, self.bin_conditioning_info_dim])

        if enforce_gamma_bounds:
            assert alpha_and_beta_angles is not None
            min_gamma, max_gamma = self.get_valid_gamma_angle_interval(
                alpha_and_beta_angles
            )
            min_gamma = min_gamma.unsqueeze(-1)
            max_gamma = max_gamma.unsqueeze(-1)

            min_normed_gamma = (self.max_bin_edge - self.min_bin_edge) * (
                (min_gamma - self.MIN_LATTICE_ANGLE)
                / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
            ) + self.min_bin_edge
            max_normed_gamma = (self.max_bin_edge - self.min_bin_edge) * (
                (max_gamma - self.MIN_LATTICE_ANGLE)
                / (self.MAX_LATTICE_ANGLE - self.MIN_LATTICE_ANGLE)
            ) + self.min_bin_edge

        _batch_idxs = paddle.arange(batch_size)
        min_bin_edge = self.min_bin_edge * paddle.ones([batch_size])
        max_bin_edge = self.max_bin_edge * paddle.ones([batch_size])
        log_probs = paddle.zeros([batch_size])

        for j in range(self.n_telescopes):
            bin_edges = (
                min_bin_edge.unsqueeze(-1) + (max_bin_edge - min_bin_edge).unsqueeze(-1)
                * self.grid_pts.unsqueeze(0).expand([batch_size, -1])
            )
            bins = paddle.stack([bin_edges[:, :-1], bin_edges[:, 1:]], axis=-1)

            # bin_embedder: (batch_size, n_bins, 2) -> (batch_size, n_bins, hidden_dim)
            bin_emb = bin_embedder(bins.reshape([-1, 2])).reshape(
                [batch_size, self.n_bins, -1]
            )

            bin_logits = self.bin_logit_head(
                paddle.concat([bin_emb, z], axis=-1)
            )[:, :, lattice_param_index]

            if enforce_gamma_bounds:
                if j < self.n_telescopes - 1:
                    epsilon = (bin_edges[0, 1] - bin_edges[0, 0]) / self.n_bins
                    zero_prob_bins_mask = (
                        (bins[:, :, 1] - epsilon < min_normed_gamma)
                        | (bins[:, :, 0] + epsilon > max_normed_gamma)
                    )
                else:
                    bin_midpoints = paddle.mean(bins, axis=-1)
                    zero_prob_bins_mask = (
                        (bin_midpoints < min_normed_gamma)
                        | (bin_midpoints > max_normed_gamma)
                    )

                if paddle.any(paddle.all(zero_prob_bins_mask, axis=-1)):
                    raise NotImplementedError("All bins were invalid")

                bin_logits = bin_logits + zero_prob_bins_mask.cast("float32") * (
                    paddle.where(zero_prob_bins_mask, paddle.to_tensor(-float('inf')), paddle.to_tensor(0.0))
                )

            if x is None:
                dist = Categorical(logits=bin_logits)
                bin_idxs = dist.sample([1]).squeeze(0)
            else:
                normalized_x = (x - min_bin_edge) / (max_bin_edge - min_bin_edge)
                bin_idxs = paddle.bucketize(normalized_x, self.grid_pts[1:])

            log_probs = log_probs + paddle.take_along_axis(
                F.log_softmax(bin_logits, axis=-1), bin_idxs.unsqueeze(-1), axis=-1
            ).squeeze(-1)

            chosen_bins = bins[_batch_idxs, bin_idxs]
            min_bin_edge = chosen_bins[:, 0]
            max_bin_edge = chosen_bins[:, 1]

        samples = paddle.mean(chosen_bins, axis=-1)
        return samples, log_probs