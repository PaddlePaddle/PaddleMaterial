from typing import Optional

import paddle
import paddle.nn as nn
from tqdm import tqdm

from ppmat.models.common import initializer
from ppmat.models.common.noise_schedule import BetaScheduler
from ppmat.models.common.noise_schedule import DiscreteScheduler
from ppmat.models.common.noise_schedule import SigmaScheduler
from ppmat.models.common.noise_schedule import d_log_p_wrapped_normal
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.models.diffcsp.cspnet import CSPNet
from ppmat.utils.crystal import lattice_params_to_matrix_paddle


class CSPDiffusionWithD3PMGuidance(paddle.nn.Layer):
    def __init__(
        self,
        decoder_cfg,
        beta_scheduler_cfg,
        sigma_scheduler_cfg,
        discrete_scheduler_cfg,
        time_dim,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        type_loss_weight: float = 20.0,
        type_ce_loss_weight: float = 0.01,
        num_classes: int = 100,
        prop_names: Optional[list[str]] = None,
        property_input_dim: int = 1,
        property_dim: int = 512,
        drop_prob: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.decoder = CSPNet(**decoder_cfg)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_cfg)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_cfg)
        self.discrete_scheduler = DiscreteScheduler(**discrete_scheduler_cfg)

        self.time_dim = time_dim
        self.lattice_loss_weight = lattice_loss_weight
        self.coord_loss_weight = coord_loss_weight
        self.type_loss_weight = type_loss_weight
        self.type_ce_loss_weight = type_ce_loss_weight
        self.num_classes = num_classes

        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.lattice_loss_weight < 1e-05
        self.keep_coords = self.coord_loss_weight < 1e-05
        if prop_names is not None:
            self.prop_names = (
                prop_names if isinstance(prop_names, list) else [prop_names]
            )
            assert property_input_dim == 1, "Property input dimension must be 1"

            self.property_embedding = paddle.nn.Linear(len(prop_names), property_dim)
            self.drop_prob = drop_prob
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def _at(self, a, t, x):
        t = t.reshape((t.shape[0], *([1] * (x.dim() - 1))))
        return a[t - 1, x, :]

    def q_sample(self, x_0, t, noise):
        logits = paddle.log(
            x=self._at(self.discrete_scheduler.q_mats, t, x_0)
            + self.discrete_scheduler.eps
        )
        noise = paddle.clip(x=noise, min=self.discrete_scheduler.eps, max=1.0)
        gumbel_noise = -paddle.log(x=-paddle.log(x=noise))
        return paddle.argmax(x=logits + gumbel_noise, axis=-1)

    def q_posterior_logits(self, x_0, x_t, t):

        if x_0.dtype == paddle.int64 or x_0.dtype == paddle.int32:
            x_0_logits = paddle.log(
                x=nn.functional.one_hot(num_classes=self.num_classes, x=x_0).astype(
                    "int64"
                )
                + self.discrete_scheduler.eps
            )
        else:
            x_0_logits = x_0.clone()
        assert tuple(x_0_logits.shape) == tuple(x_t.shape) + (self.num_classes,)

        fact1 = self._at(self.discrete_scheduler.q_one_step_transposed, t, x_t)
        softmaxed = nn.functional.softmax(x=x_0_logits, axis=-1)
        index = t - 2
        index = paddle.where(condition=index < 0, x=index + self.num_classes, y=index)
        qmats2 = self.discrete_scheduler.q_mats[index].cast(softmaxed.dtype)

        fact2 = paddle.einsum("bc,bcd->bd", softmaxed, qmats2)
        out = paddle.log(x=fact1 + self.discrete_scheduler.eps) + paddle.log(
            x=fact2 + self.discrete_scheduler.eps
        )
        t_broadcast = t.reshape((t.shape[0], *([1] * x_t.dim())))
        bc = paddle.where(condition=t_broadcast == 1, x=x_0_logits, y=out)
        return bc

    def vb(self, dist1, dist2):
        dist1 = dist1.flatten(start_axis=0, stop_axis=-2)
        dist2 = dist2.flatten(start_axis=0, stop_axis=-2)
        out = nn.functional.softmax(x=dist1 + self.discrete_scheduler.eps, axis=-1) * (
            nn.functional.log_softmax(dist1 + self.discrete_scheduler.eps, axis=-1)
            - nn.functional.log_softmax(dist2 + self.discrete_scheduler.eps, axis=-1)
        )
        return out.sum(axis=-1).mean()

    def forward(self, batch):
        structure_array = batch["structure_array"]
        batch_size = structure_array.num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array.num_atoms
        )

        times = uniform_sample_t(batch_size, self.beta_scheduler.timesteps)
        time_emb = self.time_embedding(times)

        if self.prop_names is not None:
            prop_data = [batch[prop_name] for prop_name in self.prop_names]
            prop_data = paddle.concat(prop_data, axis=1)
            property_emb = self.property_embedding(prop_data)
            property_mask = paddle.bernoulli(
                paddle.zeros([batch_size, 1]) + self.drop_prob
            )
            property_mask = 1 - property_mask
        else:
            property_emb = None
            property_mask = None

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = paddle.sqrt(x=alphas_cumprod)
        c1 = paddle.sqrt(x=1.0 - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]
        if hasattr(structure_array, "lattice"):
            lattices = structure_array.lattice
        else:
            lattices = lattice_params_to_matrix_paddle(
                structure_array.lengths, structure_array.angles
            )

        frac_coords = structure_array.frac_coords
        rand_l, rand_x = paddle.randn(
            shape=lattices.shape, dtype=lattices.dtype
        ), paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(repeats=structure_array.num_atoms)[
            :, None
        ]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(
            repeats=structure_array.num_atoms
        )[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0

        atom_types = structure_array.atom_types - 1
        atom_types_times = times.repeat_interleave(repeats=structure_array.num_atoms)
        input_atom_types = self.q_sample(
            atom_types,
            atom_types_times,
            paddle.rand(shape=(*atom_types.shape, self.num_classes)),
        )
        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices
        pred_l, pred_x, pred_t = self.decoder(
            time_emb,
            input_atom_types,
            input_frac_coords,
            input_lattice,
            structure_array.num_atoms,
            batch_idx,
            property_emb=property_emb,
            property_mask=property_mask,
        )
        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom
        ) / paddle.sqrt(x=sigmas_norm_per_atom)
        loss_lattice = paddle.nn.functional.mse_loss(input=pred_l, label=rand_l)
        loss_coord = paddle.nn.functional.mse_loss(input=pred_x, label=tar_x)

        true_q_posterior_logits = self.q_posterior_logits(
            atom_types,
            input_atom_types,
            atom_types_times,
        )
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_t, input_atom_types, atom_types_times
        )
        loss_type_vb = self.vb(true_q_posterior_logits, pred_q_posterior_logits)
        loss_type_ce = nn.functional.cross_entropy(pred_t, atom_types)
        loss_type = loss_type_vb + self.type_ce_loss_weight * loss_type_ce

        loss = (
            self.lattice_loss_weight * loss_lattice
            + self.coord_loss_weight * loss_coord
            + self.type_loss_weight * loss_type
        )
        return {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
            "loss_type": loss_type,
            "loss_type_vb": loss_type_vb,
            "loss_type_ce": loss_type_ce,
        }

    @paddle.no_grad()
    def sample(self, batch, step_lr=1e-05, is_save_traj=False, guide_w=0.5):
        structure_array = batch["structure_array"]
        batch_size = structure_array.num_atoms.shape[0]
        num_atoms = structure_array.num_atoms
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array.num_atoms
        )
        num_nodes = structure_array.num_atoms.sum()

        l_T, x_T = paddle.randn(shape=[batch_size, 3, 3]), paddle.rand(
            shape=[structure_array.num_atoms.sum(), 3]
        )
        if self.discrete_scheduler.forward_type == "uniform":
            a_T = paddle.randint(
                low=0, high=self.num_classes, shape=[structure_array.num_atoms.sum()]
            )
        elif self.discrete_scheduler.forward_type == "absorbing":
            a_T = paddle.full(
                shape=[structure_array.num_atoms.sum()],
                fill_value=self.discrete_scheduler.mask_id,
                dtype="int64",
            )
        else:
            raise NotImplementedError

        if self.keep_coords:
            x_T = structure_array.frac_coords
        if self.keep_lattice:
            if hasattr(structure_array, "lattice"):
                l_T = structure_array.lattice
            else:
                l_T = lattice_params_to_matrix_paddle(
                    structure_array.lengths, structure_array.angles
                )
        traj = {
            self.beta_scheduler.timesteps: {
                "num_atoms": structure_array.num_atoms,
                "atom_types": a_T,
                "frac_coords": x_T % 1.0,
                "lattices": l_T,
            }
        }

        prop_data = [batch[prop_name] for prop_name in self.prop_names]
        prop_data = paddle.concat(prop_data, axis=1)
        property_emb = self.property_embedding(prop_data)
        property_mask = paddle.zeros([batch_size * 2, 1])
        property_mask[:batch_size] = 1

        num_atoms_double = paddle.concat([num_atoms, num_atoms])
        batch_double = paddle.concat([batch_idx, batch_idx + batch_size])
        property_emb_double = paddle.concat([property_emb, property_emb], axis=0)

        for t in tqdm(
            range(self.beta_scheduler.timesteps, 0, -1), leave=False, desc="Sampling..."
        ):
            times = paddle.full(shape=(batch_size,), fill_value=t, dtype="int64")
            time_emb = self.time_embedding(times)
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]
            c0 = 1.0 / paddle.sqrt(x=alphas)
            c1 = (1 - alphas) / paddle.sqrt(x=1 - alphas_cumprod)
            x_t = traj[t]["frac_coords"]
            l_t = traj[t]["lattices"]
            a_t = traj[t]["atom_types"]

            if self.keep_coords:
                x_t = x_T
            if self.keep_lattice:
                l_t = l_T
            rand_l = (
                paddle.randn(shape=l_T.shape, dtype=l_T.dtype)
                if t > 1
                else paddle.zeros_like(x=l_T)
            )
            rand_x = (
                paddle.randn(shape=x_T.shape, dtype=x_T.dtype)
                if t > 1
                else paddle.zeros_like(x=x_T)
            )
            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = paddle.sqrt(x=2 * step_size)

            time_emb_double = paddle.concat([time_emb, time_emb], axis=0)
            a_t_double = paddle.concat([a_t, a_t], axis=0)
            x_t_double = paddle.concat([x_t, x_t], axis=0)
            l_t_double = paddle.concat([l_t, l_t], axis=0)

            pred_l, pred_x, pred_t = self.decoder(
                time_emb_double,
                a_t_double,
                x_t_double,
                l_t_double,
                num_atoms_double,
                batch_double,
                property_emb=property_emb_double,
                property_mask=property_mask,
            )
            pred_l = (1 + guide_w) * pred_l[:batch_size] - guide_w * pred_l[batch_size:]
            pred_x = (1 + guide_w) * pred_x[:num_nodes] - guide_w * pred_x[num_nodes:]
            pred_t = (1 + guide_w) * pred_t[:num_nodes] - guide_w * pred_t[num_nodes:]

            pred_x = pred_x * paddle.sqrt(x=sigma_norm)
            x_t_minus_05 = (
                x_t - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )
            l_t_minus_05 = l_t
            t_t_minus_05 = a_t
            rand_l = (
                paddle.randn(shape=l_T.shape, dtype=l_T.dtype)
                if t > 1
                else paddle.zeros_like(x=l_T)
            )
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

            a_t_minus_05_double = paddle.concat([t_t_minus_05, t_t_minus_05], axis=0)
            x_t_minus_05_double = paddle.concat([x_t_minus_05, x_t_minus_05], axis=0)
            l_t_minus_05_double = paddle.concat([l_t_minus_05, l_t_minus_05], axis=0)
            pred_l, pred_x, pred_t = self.decoder(
                time_emb_double,
                a_t_minus_05_double,
                x_t_minus_05_double,
                l_t_minus_05_double,
                num_atoms_double,
                batch_double,
                property_emb=property_emb_double,
                property_mask=property_mask,
            )
            pred_l = (1 + guide_w) * pred_l[:batch_size] - guide_w * pred_l[batch_size:]
            pred_x = (1 + guide_w) * pred_x[:num_nodes] - guide_w * pred_x[num_nodes:]
            pred_t = (1 + guide_w) * pred_t[:num_nodes] - guide_w * pred_t[num_nodes:]

            pred_x = pred_x * paddle.sqrt(x=sigma_norm)
            x_t_minus_1 = (
                x_t_minus_05 - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )
            l_t_minus_1 = (
                c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l
                if not self.keep_lattice
                else l_t
            )
            noise = paddle.rand(shape=(*t_t_minus_05.shape, self.num_classes))
            atom_types_times = times.repeat_interleave(
                repeats=structure_array.num_atoms
            )
            pred_q_posterior_logits = self.q_posterior_logits(
                pred_t, t_t_minus_05, atom_types_times
            )
            noise = paddle.clip(x=noise, min=self.discrete_scheduler.eps, max=1.0)
            not_first_step = (
                (atom_types_times != 1)
                .astype(dtype="float32")
                .reshape((t_t_minus_05.shape[0], *([1] * t_t_minus_05.dim())))
            )
            gumbel_noise = -paddle.log(x=-paddle.log(x=noise))
            sample = paddle.argmax(
                x=pred_q_posterior_logits + gumbel_noise * not_first_step, axis=-1
            )
            t_t_minus_1 = sample

            traj[t - 1] = {
                "num_atoms": structure_array.num_atoms,
                "atom_types": t_t_minus_1,
                "frac_coords": x_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
            }

        if is_save_traj:
            for key in traj:
                start_idx = 0
                traj_single = []
                for i in range(batch_size):
                    end_idx = start_idx + traj[key]["num_atoms"][i]
                    traj_single.append(
                        {
                            "num_atoms": traj[key]["num_atoms"][i].tolist(),
                            "atom_types": (
                                traj[key]["atom_types"][start_idx:end_idx] + 1
                            ).tolist(),
                            "frac_coords": traj[key]["frac_coords"][
                                start_idx:end_idx
                            ].tolist(),
                            "lattices": traj[key]["lattices"][i].tolist(),
                        }
                    )
                    start_idx += traj[key]["num_atoms"][i]
                traj[key] = traj_single
            result = traj[0]
            return {"result": result, "traj": traj}
        else:
            start_idx = 0
            result = []
            for i in range(batch_size):
                end_idx = start_idx + traj[0]["num_atoms"][i]
                result.append(
                    {
                        "num_atoms": traj[0]["num_atoms"][i].tolist(),
                        "atom_types": (
                            traj[0]["atom_types"][start_idx:end_idx] + 1
                        ).tolist(),
                        "frac_coords": traj[0]["frac_coords"][
                            start_idx:end_idx
                        ].tolist(),
                        "lattices": traj[0]["lattices"][i].tolist(),
                    }
                )
                start_idx += traj[0]["num_atoms"][i]
            return {"result": result, "traj": {}}
