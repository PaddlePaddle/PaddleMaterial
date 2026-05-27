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

"""
Crystal diffusion controllers for MiAD.
"""

import os

import numpy as np
import paddle

from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.miad.diffusion_utils import TimeDistribution
from ppmat.models.miad.diffusion_utils import mean_interleave
from ppmat.models.miad.frac_diffusion import PFM
from ppmat.models.miad.frac_diffusion import WrappedNormal
from ppmat.models.miad.lattice_diffusion import DDPM
from ppmat.models.miad.lattice_diffusion import FM
from ppmat.models.miad.lattice_diffusion import FM_LenAng
from ppmat.models.miad.type_diffusion import D3PM
from ppmat.models.miad.type_diffusion import DDPM_onehot


def parse_num_atoms_to_per_crystal(num_atoms_data):
    if num_atoms_data is None:
        return None
    if hasattr(num_atoms_data, "numpy"):
        num_atoms_np = num_atoms_data.numpy().flatten()
    elif hasattr(num_atoms_data, "reshape"):
        num_atoms_np = num_atoms_data.reshape(-1)
    else:
        num_atoms_np = np.array(num_atoms_data).flatten()
    return paddle.to_tensor(num_atoms_np.astype("int64")), num_atoms_np


def parse_batch(batch):
    if "batch" in batch:
        return {
            "num_atoms": batch["batch"].num_atoms,
            "batch_idx": batch["batch"].batch,
            "atom_types": batch["batch"].atom_types,
            "batch_size": batch["batch_size"],
        }
    if "batch_idx" in batch:
        return {
            "num_atoms": batch["num_atoms"],
            "batch_idx": batch["batch_idx"],
            "atom_types": batch["atom_types"],
            "batch_size": batch.get("batch_size", len(batch["num_atoms"])),
        }
    raise ValueError("Cannot determine batch format: missing 'batch' or 'batch_idx'")


def init_diffusion(diffusion_config, logger):
    if hasattr(diffusion_config, "default_config") and diffusion_config.default_config:
        diffusion_config = _apply_default_config(diffusion_config.default_config)

    switch = {
        "Default": CrystalGen,
        "DiffCSP": DiffCSP,
    }
    method = diffusion_config.method
    if method in switch:
        return switch[method](diffusion_config, logger)
    raise NotImplementedError(f"Diffusion method '{method}' not implemented")


def _apply_default_config(config_name):
    from ppmat.models.miad.default_configs import DEFAULT_DIFFUSION_CONFIGS

    if config_name in DEFAULT_DIFFUSION_CONFIGS:
        return DEFAULT_DIFFUSION_CONFIGS[config_name]
    raise KeyError(f"Unknown default config: '{config_name}'")


class CrystalGen:
    """Standard crystal generation with single-step reverse sampling."""

    def __init__(self, diffusion_config, logger):
        self.config = diffusion_config
        self.logger = logger
        self.cont_time = self.config.cont_time
        self.num_steps = self.config.num_steps
        self.time_distribution = TimeDistribution(self.num_steps, self.cont_time)
        time_embed_dim = getattr(self.config, "time_embed_dim", 256)
        self.time_embedding = SinusoidalTimeEmbeddings(time_embed_dim)

        # Lattice diffusion
        switch_lat = {
            "ddpm": DDPM,
            "fm": FM,
            "fm_lenang": FM_LenAng,
        }
        self.lat_diffusion = switch_lat[self.config.lat_diffusion.method](self.config)

        # Fractional coordinate diffusion
        switch_frac = {
            "wrapped_normal": WrappedNormal,
            "pfm": PFM,
        }
        self.frac_diffusion = switch_frac[self.config.frac_diffusion.method](
            self.config
        )

        # Atom type diffusion (only for generation tasks)
        self.gen_type = "gen" in self.config.task
        if self.gen_type:
            switch_type = {
                "ddpm_onehot": DDPM_onehot,
                "d3pm": D3PM,
            }
            self.type_diffusion = switch_type[self.config.type_diffusion.method](
                self.config
            )
        else:
            self.type_diffusion = None

    def forward_step_sample(self, x0, t, batch):
        l0, f0, a0 = x0
        lt = self.lat_diffusion.forward_step_sample(l0, t[0], batch)
        ft = self.frac_diffusion.forward_step_sample(f0, t[1], batch)
        at = (
            self.type_diffusion.forward_step_sample(a0, t[1], batch)
            if self.gen_type
            else a0
        )
        return [lt, ft, at]

    def reverse_step_sample(self, xt, t, model, batch):
        lt, ft, at = xt
        batch["prediction"] = self.model_prediction(xt, t, model, batch)
        l_pred, f_pred, a_pred = batch["prediction"]
        lt_1 = self.lat_diffusion.reverse_step_sample(l_pred, lt, t[0], batch)
        ft_1 = self.frac_diffusion.reverse_step_sample(f_pred, ft, t[1], batch)
        at_1 = (
            self.type_diffusion.reverse_step_sample(a_pred, at, t[1], batch)
            if self.gen_type
            else at
        )
        return [lt_1, ft_1, at_1]

    def _get_batch_info(self, batch):
        try:
            return parse_batch(batch)
        except ValueError:
            return self._normalize_batch(batch)

    def _normalize_batch(self, batch):
        if "structure_array" in batch:
            num_atoms_data = batch["structure_array"].get("num_atoms", None)
        else:
            num_atoms_data = batch.get("num_atoms", None)

        parsed = parse_num_atoms_to_per_crystal(num_atoms_data)
        if parsed is None:
            raise ValueError("Cannot determine num_atoms from batch")
        num_atoms, num_atoms_np = parsed
        batch_size = len(num_atoms_np)

        batch_idx_np = np.concatenate(
            [np.full(int(n), i) for i, n in enumerate(num_atoms_np)]
        )
        batch_idx = paddle.to_tensor(batch_idx_np.astype("int64"))
        atom_types = paddle.zeros([int(num_atoms_np.sum())], dtype="int64")

        return {
            "num_atoms": num_atoms,
            "batch_idx": batch_idx,
            "atom_types": atom_types,
            "batch_size": batch_size,
        }

    def prior_sample(self, batch):
        batch_info = self._get_batch_info(batch)
        return [
            self.lat_diffusion.prior_sample(batch),
            self.frac_diffusion.prior_sample(batch),
            (
                self.type_diffusion.prior_sample(batch)
                if self.gen_type
                else batch_info["atom_types"]
            ),
        ]

    def model_prediction(self, xt, t, model, batch):
        batch_info = self._get_batch_info(batch)
        lt, ft, at = xt
        nn_pred = model(
            t[0],
            self.time_embedding(1000 * (t[0] / self.num_steps) + 1),
            at,
            ft,
            lt,
            batch_info["num_atoms"],
            batch_info["batch_idx"],
        )
        l_pred = nn_pred[0]
        f_pred = nn_pred[1]
        a_pred = nn_pred[2] if self.gen_type else None
        return [l_pred, f_pred, a_pred]

    def get_x0_prediction(self, pred, xt, t, batch):
        l_pred, f_pred, a_pred = pred
        lt, ft, at = xt
        return [
            self.lat_diffusion.get_x0_prediction(l_pred, lt, t[0], batch),
            self.frac_diffusion.get_x0_prediction(f_pred, ft, t[1], batch),
            (
                self.type_diffusion.get_x0_prediction(
                    a_pred, at, t[1], batch, x0_format="disc"
                )
                if self.gen_type
                else at.clone().detach()
            ),
        ]

    def train_step(self, batch, model, mode):
        modifications = os.environ.get("MODIFICATIONS_FIELD", "")

        batch["t"] = self.time_distribution.sample(batch, mode)
        batch["xt"] = self.forward_step_sample(batch["x0"], batch["t"], batch)
        batch["prediction"] = self.model_prediction(
            batch["xt"], batch["t"], model, batch
        )

        loss_lat = self.lat_diffusion.loss(batch)
        loss_frac = self.frac_diffusion.loss(batch)

        if "insert_latloss_" in modifications:
            lat_coef = float(
                modifications.split("insert_latloss_")[1].split("_coef")[0]
            )
            loss_lat = loss_lat * lat_coef
        if "insert_fracloss_" in modifications:
            frac_coef = float(
                modifications.split("insert_fracloss_")[1].split("_coef")[0]
            )
            loss_frac = loss_frac * frac_coef

        loss_lat_val = loss_lat.mean()
        loss_frac_val = loss_frac.mean()
        batch["loss"] = loss_lat_val + loss_frac_val

        if self.gen_type:
            loss_type = self.type_diffusion.loss(batch)
            if "insert_typeloss_" in modifications:
                type_coef = float(
                    modifications.split("insert_typeloss_")[1].split("_coef")[0]
                )
                loss_type = loss_type * type_coef
            loss_type_val = loss_type.mean()
            batch["loss"] = batch["loss"] + loss_type_val

        if self.logger is not None:
            batch_info = self._get_batch_info(batch)
            t_log = paddle.clip(batch["t"][0].clone().cast("int64"), 0, 999)

            self.logger.add(
                f"loss:lattice:{mode}", loss_lat_val.item(), stack_after_epoch=True
            )
            loss_lat_t = paddle.zeros([self.num_steps], dtype=loss_lat.dtype)
            loss_lat_t[t_log] = loss_lat.clone()
            self.logger.add(
                f"loss:lattice4time:{mode}", loss_lat_t, stack_after_epoch=True
            )

            self.logger.add(
                f"loss:coord:{mode}", loss_frac_val.item(), stack_after_epoch=True
            )
            loss_frac_t = paddle.zeros([self.num_steps], dtype=loss_frac.dtype)
            loss_frac_t[t_log] = mean_interleave(
                loss_frac.clone(), batch_info["num_atoms"]
            )
            self.logger.add(
                f"loss:coord4time:{mode}", loss_frac_t, stack_after_epoch=True
            )

            if self.gen_type:
                self.logger.add(
                    f"loss:type:{mode}",
                    loss_type_val.item(),
                    stack_after_epoch=True,
                )
                loss_type_t = paddle.zeros([self.num_steps], dtype=loss_type.dtype)
                loss_type_t[t_log] = mean_interleave(
                    loss_type.clone(), batch_info["num_atoms"]
                )
                self.logger.add(
                    f"loss:type4time:{mode}", loss_type_t, stack_after_epoch=True
                )

        return batch

    def sampling_procedure(self, model, batch, progress_printer):
        batch["xt"] = self.prior_sample(batch)
        reverse_time_iterator = self.time_distribution.reverse_time_iterator(
            batch, start_from=self.num_steps - 1
        )
        for t_vector in reverse_time_iterator:
            batch["t"] = self.time_distribution.to_cuda(t_vector, batch)
            t_value = batch["t"][0][0].item()
            progress_printer(t_value)
            batch["xt"] = self.reverse_step_sample(
                batch["xt"], batch["t"], model, batch
            )
        batch["xt"] = self.output_transform(batch["xt"], batch)
        batch["x0_prediction"] = batch["xt"]
        return batch

    def output_transform(self, x0, batch):
        return [
            self.lat_diffusion.output_transform(x0[0], batch),
            self.frac_diffusion.output_transform(x0[1], batch),
            (
                self.type_diffusion.output_transform(x0[2], batch)
                if self.gen_type
                else x0[2]
            ),
        ]


class DiffCSP(CrystalGen):
    """DiffCSP with two-step reverse sampling."""

    def reverse_step_sample(self, xt, t, model, batch):
        lt, ft, at = xt

        # Step 1: only use fractional prediction
        _, f_pred, _ = self.model_prediction(xt, t, model, batch)
        ft_05 = self.frac_diffusion.reverse_step_sample_part_1(f_pred, ft, t[1], batch)
        xt_05 = [lt, ft_05, at]

        # Step 2: full prediction with corrected fractional coords
        l_pred, f_pred, a_pred = self.model_prediction(xt_05, t, model, batch)
        lt_1 = self.lat_diffusion.reverse_step_sample(l_pred, lt, t[0], batch)
        ft_1 = self.frac_diffusion.reverse_step_sample_part_2(
            f_pred, ft_05, t[1], batch
        )
        at_1 = (
            self.type_diffusion.reverse_step_sample(a_pred, at, t[1], batch)
            if self.gen_type
            else at
        )
        return [lt_1, ft_1, at_1]
