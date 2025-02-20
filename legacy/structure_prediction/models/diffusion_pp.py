from copy import deepcopy as dc

import paddle
import paddle.nn as nn
from models import initializer
from models.cspppnet import CSPPPNet
from models.gemnet.utils import scatter
from models.lattice import CrystalFamily
from models.noise_schedule import BetaScheduler
from models.noise_schedule import SigmaScheduler
from models.noise_schedule import d_log_p_wrapped_normal
from models.time_embedding import SinusoidalTimeEmbeddings
from models.time_embedding import uniform_sample_t
from tqdm import tqdm
from utils.crystal import lattice_params_to_matrix_paddle

MAX_ATOMIC_NUM = 100


class CSPDiffusionPP(paddle.nn.Layer):
    def __init__(
        self,
        decoder_cfg,
        beta_scheduler_cfg,
        sigma_scheduler_cfg,
        time_dim,
        cost_lattice,
        cost_coord,
        pretrained=None,
        num_classes=100,
        **kwargs,
    ) -> None:
        super().__init__()

        self.decoder = CSPPPNet(**decoder_cfg)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_cfg)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_cfg)

        self.time_dim = time_dim
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord
        self.num_classes = num_classes

        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.cost_lattice < 1e-05
        self.keep_coords = self.cost_coord < 1e-05
        self.device = paddle.device.get_device()
        self.pretrained = pretrained
        self.apply(self._init_weights)
        if self.pretrained is not None:
            self.set_dict(paddle.load(self.pretrained))

        self.crystal_family = CrystalFamily()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def forward(self, batch):

        batch_size = batch["num_graphs"]
        times = uniform_sample_t(batch_size, self.beta_scheduler.timesteps)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = paddle.sqrt(x=alphas_cumprod)
        c1 = paddle.sqrt(x=1.0 - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_paddle(batch["lengths"], batch["angles"])
        lattices = self.crystal_family.de_so3(lattices)

        frac_coords = batch["frac_coords"]

        rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
        sigmas_per_atom = sigmas.repeat_interleave(repeats=batch["num_atoms"])[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(
            repeats=batch["num_atoms"]
        )[:, None]

        rand_x_anchor = rand_x[batch["anchor_index"]]
        rand_x_anchor = (
            batch["ops_inv"][batch["anchor_index"]] @ rand_x_anchor.unsqueeze(axis=-1)
        ).squeeze(axis=-1)
        rand_x_anchor = paddle.cast(rand_x_anchor, dtype="float32")
        rand_x = (batch["ops"][:, :3, :3] @ rand_x_anchor.unsqueeze(axis=-1)).squeeze(
            axis=-1
        )
        rand_x = paddle.cast(rand_x, dtype="float32")
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0

        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            ori_crys_fam, batch["spacegroup"]
        )
        rand_crys_fam = paddle.randn(shape=ori_crys_fam.shape, dtype=ori_crys_fam.dtype)

        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            rand_crys_fam, batch["spacegroup"]
        )
        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            input_crys_fam, batch["spacegroup"]
        )

        pred_crys_fam, pred_x = self.decoder(
            time_emb,
            batch["atom_types"],
            input_frac_coords,
            input_crys_fam,
            batch["num_atoms"],
            batch["batch"],
        )
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            pred_crys_fam, batch["spacegroup"]
        )

        pred_x_proj = paddle.einsum("bij, bj-> bi", batch["ops_inv"], pred_x)

        tar_x_anchor = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x_anchor, sigmas_per_atom
        ) / paddle.sqrt(x=sigmas_norm_per_atom)

        loss_lattice = paddle.nn.functional.mse_loss(
            input=pred_crys_fam, label=rand_crys_fam
        )
        loss_coord = paddle.nn.functional.mse_loss(
            input=pred_x_proj, label=tar_x_anchor
        )
        loss = self.cost_lattice * loss_lattice + self.cost_coord * loss_coord

        return {"loss": loss, "loss_lattice": loss_lattice, "loss_coord": loss_coord}

    @paddle.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-05):

        batch_size = batch["num_graphs"]

        x_T = paddle.rand(shape=[batch["num_nodes"], 3]).to(self.device)
        crys_fam_T = paddle.randn([batch_size, 6]).to(self.device)
        crys_fam_T = self.crystal_family.proj_k_to_spacegroup(
            crys_fam_T, batch["spacegroup"]
        )

        if diff_ratio < 1:
            time_start = int(self.beta_scheduler.timesteps * diff_ratio)
            lattices = lattice_params_to_matrix_paddle(
                batch["lengths"], batch["angles"]
            )
            lattices = self.crystal_family.de_so3(lattices)
            ori_crys_fam = self.crystal_family.m2v(lattices)
            ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                ori_crys_fam, batch["spacegroup"]
            )

            frac_coords = batch["frac_coords"]

            rand_crys_fam, rand_x = paddle.randn(
                shape=ori_crys_fam.shape, dtype=ori_crys_fam.dtype
            ), paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)

            alphas_cumprod = self.beta_scheduler.alphas_cumprod[time_start]

            c0 = paddle.sqrt(x=alphas_cumprod)
            c1 = paddle.sqrt(x=1.0 - alphas_cumprod)

            sigmas = self.sigma_scheduler.sigmas[time_start]

            rand_x_anchor = rand_x[batch["anchor_index"]]
            rand_x_anchor = (
                batch["ops"][batch["anchor_index"], :3, :3]
                @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)
            rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(axis=-1)).squeeze(
                axis=-1
            )

            crys_fam_T = c0 * ori_crys_fam + c1 * rand_crys_fam
            x_T = (frac_coords + sigmas * rand_x) % 1.0

        else:
            time_start = self.beta_scheduler.timesteps - 1

        l_T = self.crystal_family.v2m(crys_fam_T)

        x_T_all = paddle.concat(
            x=[
                x_T[batch["anchor_index"]],
                paddle.ones(shape=[batch["ops"].shape[0], 1]).to(x_T.place),
            ],
            axis=-1,
        ).unsqueeze(axis=-1)

        x_T = (batch["ops"] @ x_T_all).squeeze(axis=-1)[:, :3] % 1.0

        traj = {
            time_start: {
                "num_atoms": batch["num_atoms"],
                "atom_types": batch["atom_types"],
                "frac_coords": x_T % 1.0,
                "lattices": l_T,
                "crys_fam": crys_fam_T,
            }
        }

        for t in tqdm(range(time_start, 0, -1)):

            times = paddle.full(shape=(batch_size,), fill_value=t)
            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            alphas_cumprod_next = self.beta_scheduler.alphas_cumprod[t - 1]

            alphas_cumprod_next = paddle.sqrt(x=alphas_cumprod_next)

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / paddle.sqrt(x=alphas)
            c1 = (1 - alphas) / paddle.sqrt(x=1 - alphas_cumprod)

            x_t = traj[t]["frac_coords"]
            crys_fam_t = traj[t]["crys_fam"]

            # Corrector

            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )

            step_size = step_lr / (sigma_norm * self.sigma_scheduler.sigma_begin**2)
            std_x = paddle.sqrt(x=2 * step_size)

            rand_x_anchor = rand_x[batch["anchor_index"]]
            rand_x_anchor = (
                batch["ops_inv"][batch["anchor_index"]]
                @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)
            rand_x = (
                batch["ops"][:, :3, :3] @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)

            pred_crys_fam, pred_x = self.decoder(
                time_emb,
                batch["atom_types"],
                x_t,
                crys_fam_t,
                batch["num_atoms"],
                batch["batch"],
            )

            pred_x = pred_x * paddle.sqrt(x=sigma_norm)

            pred_x_proj = paddle.einsum("bij, bj-> bi", batch["ops_inv"], pred_x)
            pred_x_anchor = scatter(
                pred_x_proj, batch["anchor_index"], dim=0, reduce="mean"
            )[batch["anchor_index"]]

            pred_x = (
                batch["ops"][:, :3, :3] @ pred_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            crys_fam_t_minus_05 = crys_fam_t

            frac_coords_all = paddle.concat(
                x=[
                    x_t_minus_05[batch["anchor_index"]],
                    paddle.ones(shape=[batch["ops"].shape[0], 1]).to(
                        x_t_minus_05.place
                    ),
                ],
                axis=-1,
            ).unsqueeze(axis=-1)

            x_t_minus_05 = (batch["ops"] @ frac_coords_all).squeeze(axis=-1)[
                :, :3
            ] % 1.0

            # Predictor

            rand_crys_fam = paddle.randn(shape=crys_fam_T.shape, dtype=crys_fam_T.dtype)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                rand_crys_fam, batch["spacegroup"]
            )
            ori_crys_fam = crys_fam_t_minus_05
            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = paddle.sqrt(
                x=adjacent_sigma_x**2
                * (sigma_x**2 - adjacent_sigma_x**2)
                / sigma_x**2
            )

            rand_x_anchor = rand_x[batch["anchor_index"]]
            rand_x_anchor = (
                batch["ops_inv"][batch["anchor_index"]]
                @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)
            rand_x = (
                batch["ops"][:, :3, :3] @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)

            pred_crys_fam, pred_x = self.decoder(
                time_emb,
                batch["atom_types"],
                x_t_minus_05,
                crys_fam_t,
                batch["num_atoms"],
                batch["batch"],
            )

            pred_x = pred_x * paddle.sqrt(x=sigma_norm)

            crys_fam_t_minus_1 = (
                c0 * (ori_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            )
            crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(
                crys_fam_t_minus_1, batch["spacegroup"]
            )

            pred_x_proj = paddle.einsum("bij, bj-> bi", batch["ops_inv"], pred_x)
            pred_x_anchor = scatter(
                pred_x_proj, batch["anchor_index"], dim=0, reduce="mean"
            )[batch["anchor_index"]]
            pred_x = (
                batch["ops"][:, :3, :3] @ pred_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)

            frac_coords_all = paddle.concat(
                x=[
                    x_t_minus_1[batch["anchor_index"]],
                    paddle.ones(shape=[batch["ops"].shape[0], 1]).to(x_t_minus_1.place),
                ],
                axis=-1,
            ).unsqueeze(axis=-1)

            x_t_minus_1 = (batch["ops"] @ frac_coords_all).squeeze(axis=-1)[:, :3] % 1.0

            traj[t - 1] = {
                "num_atoms": batch["num_atoms"],
                "atom_types": batch["atom_types"],
                "frac_coords": x_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
                "crys_fam": crys_fam_t_minus_1,
            }

        traj_stack = {
            "num_atoms": batch["num_atoms"],
            "atom_types": batch["atom_types"],
            "all_frac_coords": paddle.stack(
                x=[traj[i]["frac_coords"] for i in range(time_start, -1, -1)]
            ),
            "all_lattices": paddle.stack(
                x=[traj[i]["lattices"] for i in range(time_start, -1, -1)]
            ),
        }
        return traj[0], traj_stack


class CSPDiffusionPPWithType(paddle.nn.Layer):
    def __init__(
        self,
        decoder_cfg,
        beta_scheduler_cfg,
        sigma_scheduler_cfg,
        time_dim,
        cost_lattice,
        cost_coord,
        cost_type,
        pretrained=None,
        num_classes=100,
        **kwargs,
    ) -> None:
        super().__init__()

        self.decoder = CSPPPNet(**decoder_cfg)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_cfg)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_cfg)

        self.time_dim = time_dim
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord
        self.cost_type = cost_type
        self.num_classes = num_classes

        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.cost_lattice < 1e-05
        self.keep_coords = self.cost_coord < 1e-05
        self.device = paddle.device.get_device()
        self.pretrained = pretrained
        self.apply(self._init_weights)
        if self.pretrained is not None:
            self.set_dict(paddle.load(self.pretrained))

        self.crystal_family = CrystalFamily()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def forward(self, batch):

        batch_size = batch["num_graphs"]
        times = uniform_sample_t(batch_size, self.beta_scheduler.timesteps)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = paddle.sqrt(x=alphas_cumprod)
        c1 = paddle.sqrt(x=1.0 - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_paddle(batch["lengths"], batch["angles"])
        lattices = self.crystal_family.de_so3(lattices)

        frac_coords = batch["frac_coords"]

        rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
        sigmas_per_atom = sigmas.repeat_interleave(repeats=batch["num_atoms"])[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(
            repeats=batch["num_atoms"]
        )[:, None]

        rand_x_anchor = rand_x[batch["anchor_index"]]
        rand_x_anchor = (
            batch["ops_inv"][batch["anchor_index"]] @ rand_x_anchor.unsqueeze(axis=-1)
        ).squeeze(axis=-1)
        rand_x_anchor = paddle.cast(rand_x_anchor, dtype="float32")
        rand_x = (batch["ops"][:, :3, :3] @ rand_x_anchor.unsqueeze(axis=-1)).squeeze(
            axis=-1
        )
        rand_x = paddle.cast(rand_x, dtype="float32")
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0

        ori_crys_fam = self.crystal_family.m2v(lattices)
        ori_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            ori_crys_fam, batch["spacegroup"]
        )
        rand_crys_fam = paddle.randn(shape=ori_crys_fam.shape, dtype=ori_crys_fam.dtype)

        rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            rand_crys_fam, batch["spacegroup"]
        )
        input_crys_fam = c0[:, None] * ori_crys_fam + c1[:, None] * rand_crys_fam
        input_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            input_crys_fam, batch["spacegroup"]
        )

        gt_atom_types_onehot = (
            paddle.nn.functional.one_hot(
                num_classes=self.num_classes, x=batch["atom_types"] - 1
            )
            .astype("int64")
            .astype(dtype="float32")
        )
        rand_t = paddle.randn(
            shape=gt_atom_types_onehot.shape, dtype=gt_atom_types_onehot.dtype
        )[batch["anchor_index"]]
        atom_type_probs = (
            c0.repeat_interleave(repeats=batch["num_atoms"])[:, None]
            * gt_atom_types_onehot
            + c1.repeat_interleave(repeats=batch["num_atoms"])[:, None] * rand_t
        )[batch["anchor_index"]]

        pred_crys_fam, pred_x, pred_t = self.decoder(
            time_emb,
            atom_type_probs,
            input_frac_coords,
            input_crys_fam,
            batch["num_atoms"],
            batch["batch"],
        )
        pred_crys_fam = self.crystal_family.proj_k_to_spacegroup(
            pred_crys_fam, batch["spacegroup"]
        )

        pred_x_proj = paddle.einsum("bij, bj-> bi", batch["ops_inv"], pred_x)

        tar_x_anchor = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x_anchor, sigmas_per_atom
        ) / paddle.sqrt(x=sigmas_norm_per_atom)

        loss_lattice = paddle.nn.functional.mse_loss(
            input=pred_crys_fam, label=rand_crys_fam
        )
        loss_coord = paddle.nn.functional.mse_loss(
            input=pred_x_proj, label=tar_x_anchor
        )
        loss_type = paddle.nn.functional.mse_loss(input=pred_t, label=rand_t)
        loss = (
            self.cost_lattice * loss_lattice
            + self.cost_coord * loss_coord
            + self.cost_type * loss_type
        )

        return {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
            "loss_type": loss_type,
        }

    @paddle.no_grad()
    def sample(self, batch, diff_ratio=1.0, step_lr=1e-05):

        batch_size = batch["num_graphs"]

        x_T = paddle.rand(shape=[batch["num_nodes"], 3]).to(self.device)
        crys_fam_T = paddle.randn([batch_size, 6]).to(self.device)
        crys_fam_T = self.crystal_family.proj_k_to_spacegroup(
            crys_fam_T, batch["spacegroup"]
        )
        t_T = paddle.randn([batch["num_nodes"], MAX_ATOMIC_NUM]).to(self.device)

        time_start = self.beta_scheduler.timesteps - 1

        l_T = self.crystal_family.v2m(crys_fam_T)

        x_T_all = paddle.concat(
            x=[
                x_T[batch["anchor_index"]],
                paddle.ones(shape=[batch["ops"].shape[0], 1]).to(x_T.place),
            ],
            axis=-1,
        ).unsqueeze(axis=-1)

        x_T = (batch["ops"] @ x_T_all).squeeze(axis=-1)[:, :3] % 1.0

        t_T = t_T[batch["anchor_index"]]

        traj = {
            time_start: {
                "num_atoms": batch["num_atoms"],
                "atom_types": t_T,
                "frac_coords": x_T % 1.0,
                "lattices": l_T,
                "crys_fam": crys_fam_T,
            }
        }

        for t in tqdm(range(time_start, 0, -1)):

            times = paddle.full(shape=(batch_size,), fill_value=t, dtype="int64")
            time_emb = self.time_embedding(times)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            alphas_cumprod_next = self.beta_scheduler.alphas_cumprod[t - 1]

            alphas_cumprod_next = paddle.sqrt(x=alphas_cumprod_next)

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            c0 = 1.0 / paddle.sqrt(x=alphas)
            c1 = (1 - alphas) / paddle.sqrt(x=1 - alphas_cumprod)

            x_t = traj[t]["frac_coords"]
            crys_fam_t = traj[t]["crys_fam"]
            t_t = traj[t]["atom_types"]

            # Corrector

            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )
            rand_t = (
                paddle.randn(shape=t_T.shape, dtype=t_T.dtype)
                if t > 1
                else paddle.zeros_like(x=t_T)
            )

            step_size = step_lr / (sigma_norm * self.sigma_scheduler.sigma_begin**2)
            std_x = paddle.sqrt(x=2 * step_size)

            rand_x_anchor = rand_x[batch["anchor_index"]]
            rand_x_anchor = (
                batch["ops_inv"][batch["anchor_index"]]
                @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)
            rand_x = (
                batch["ops"][:, :3, :3] @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)

            pred_crys_fam, pred_x, pred_t = self.decoder(
                time_emb, t_t, x_t, crys_fam_t, batch["num_atoms"], batch["batch"]
            )

            pred_x = pred_x * paddle.sqrt(x=sigma_norm)

            pred_x_proj = paddle.einsum("bij, bj-> bi", batch["ops_inv"], pred_x)
            pred_x_anchor = scatter(
                pred_x_proj, batch["anchor_index"], dim=0, reduce="mean"
            )[batch["anchor_index"]]

            pred_x = (
                batch["ops"][:, :3, :3] @ pred_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

            crys_fam_t_minus_05 = crys_fam_t

            frac_coords_all = paddle.concat(
                x=[
                    x_t_minus_05[batch["anchor_index"]],
                    paddle.ones(shape=[batch["ops"].shape[0], 1]).to(
                        x_t_minus_05.place
                    ),
                ],
                axis=-1,
            ).unsqueeze(axis=-1)

            x_t_minus_05 = (batch["ops"] @ frac_coords_all).squeeze(axis=-1)[
                :, :3
            ] % 1.0

            t_t_minus_05 = t_t

            # Predictor

            rand_crys_fam = paddle.randn(shape=crys_fam_T.shape, dtype=crys_fam_T.dtype)
            rand_crys_fam = self.crystal_family.proj_k_to_spacegroup(
                rand_crys_fam, batch["spacegroup"]
            )
            ori_crys_fam = crys_fam_t_minus_05
            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )
            rand_t = (
                paddle.randn(shape=t_T.shape, dtype=t_T.dtype)
                if t > 1
                else paddle.zeros_like(x=t_T)
            )

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t - 1]
            step_size = sigma_x**2 - adjacent_sigma_x**2
            std_x = paddle.sqrt(
                x=adjacent_sigma_x**2
                * (sigma_x**2 - adjacent_sigma_x**2)
                / sigma_x**2
            )

            rand_x_anchor = rand_x[batch["anchor_index"]]
            rand_x_anchor = (
                batch["ops_inv"][batch["anchor_index"]]
                @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)
            rand_x = (
                batch["ops"][:, :3, :3] @ rand_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)

            pred_crys_fam, pred_x, pred_t = self.decoder(
                time_emb,
                t_t_minus_05,
                x_t_minus_05,
                crys_fam_t,
                batch["num_atoms"],
                batch["batch"],
            )

            pred_x = pred_x * paddle.sqrt(x=sigma_norm)

            crys_fam_t_minus_1 = (
                c0 * (ori_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
            )
            crys_fam_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(
                crys_fam_t_minus_1, batch["spacegroup"]
            )

            pred_x_proj = paddle.einsum("bij, bj-> bi", batch["ops_inv"], pred_x)
            pred_x_anchor = scatter(
                pred_x_proj, batch["anchor_index"], dim=0, reduce="mean"
            )[batch["anchor_index"]]
            pred_x = (
                batch["ops"][:, :3, :3] @ pred_x_anchor.unsqueeze(axis=-1)
            ).squeeze(axis=-1)

            pred_t = scatter(pred_t, batch["anchor_index"], dim=0, reduce="mean")[
                batch["anchor_index"]
            ]

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

            l_t_minus_1 = self.crystal_family.v2m(crys_fam_t_minus_1)

            frac_coords_all = paddle.concat(
                x=[
                    x_t_minus_1[batch["anchor_index"]],
                    paddle.ones(shape=[batch["ops"].shape[0], 1]).to(x_t_minus_1.place),
                ],
                axis=-1,
            ).unsqueeze(axis=-1)

            x_t_minus_1 = (batch["ops"] @ frac_coords_all).squeeze(axis=-1)[:, :3] % 1.0

            t_t_minus_1 = c0 * (t_t_minus_05 - c1 * pred_t) + sigmas * rand_t

            t_t_minus_1 = t_t_minus_1[batch["anchor_index"]]

            traj[t - 1] = {
                "num_atoms": batch["num_atoms"],
                "atom_types": t_t_minus_1,
                "frac_coords": x_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
                "crys_fam": crys_fam_t_minus_1,
            }

        traj_stack = {
            "num_atoms": batch["num_atoms"],
            "atom_types": paddle.stack(
                x=[traj[i]["atom_types"] for i in range(time_start, -1, -1)]
            ).argmax(axis=-1)
            + 1,
            "all_frac_coords": paddle.stack(
                x=[traj[i]["frac_coords"] for i in range(time_start, -1, -1)]
            ),
            "all_lattices": paddle.stack(
                x=[traj[i]["lattices"] for i in range(time_start, -1, -1)]
            ),
        }

        res = dc(traj[0])
        res["atom_types"] = paddle.argmax(res["atom_types"], axis=-1) + 1

        return res, traj_stack
