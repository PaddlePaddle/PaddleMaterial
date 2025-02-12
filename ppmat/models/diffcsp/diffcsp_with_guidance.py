import paddle
import paddle.nn as nn
from tqdm import tqdm

from ppmat.models.common import initializer
from ppmat.models.common.noise_schedule import BetaScheduler
from ppmat.models.common.noise_schedule import SigmaScheduler
from ppmat.models.common.noise_schedule import d_log_p_wrapped_normal
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.models.diffcsp.cspnet import CSPNet
from ppmat.utils.crystal import lattice_params_to_matrix_paddle


class CSPDiffusionWithGuidance(paddle.nn.Layer):
    def __init__(
        self,
        decoder_cfg: dict,
        beta_scheduler_cfg: dict,
        sigma_scheduler_cfg: dict,
        time_dim: int,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        num_classes: int = 100,
        prop_name: str = None,
        property_input_dim: int = 1,
        property_dim: int = 512,
        drop_prob: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.decoder = CSPNet(**decoder_cfg)

        self.beta_scheduler = BetaScheduler(**beta_scheduler_cfg)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_cfg)

        self.time_dim = time_dim
        self.lattice_loss_weight = lattice_loss_weight
        self.coord_loss_weight = coord_loss_weight
        self.num_classes = num_classes

        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.keep_lattice = self.lattice_loss_weight < 1e-05
        self.keep_coords = self.coord_loss_weight < 1e-05

        if prop_name is not None:
            self.prop_name = prop_name
            assert property_input_dim == 1, "Property input dimension must be 1"
            self.property_embedding = paddle.nn.Linear(property_input_dim, property_dim)
            self.drop_prob = drop_prob

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def forward(self, batch):
        structure_array = batch["structure_array"]
        batch_size = structure_array.num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array.num_atoms
        )

        times = uniform_sample_t(batch_size, self.beta_scheduler.timesteps)
        time_emb = self.time_embedding(times)

        if self.prop_name is not None:
            property_emb = self.property_embedding(batch[self.prop_name])
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
        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice = lattices
        pred_l, pred_x = self.decoder(
            time_emb,
            structure_array.atom_types - 1,
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
        loss = (
            self.lattice_loss_weight * loss_lattice
            + self.coord_loss_weight * loss_coord
        )
        return {"loss": loss, "loss_lattice": loss_lattice, "loss_coord": loss_coord}

    @paddle.no_grad()
    def sample(self, batch, step_lr=1e-05, is_save_traj=False, guide_w=0.5):
        structure_array = batch["structure_array"]
        batch_size = structure_array.num_atoms.shape[0]
        # batch_idx = paddle.repeat_interleave(
        #     paddle.arange(batch_size), repeats=structure_array.num_atoms
        # )

        l_T, x_T = paddle.randn(shape=[batch_size, 3, 3]), paddle.rand(
            shape=[structure_array.num_atoms.sum(), 3]
        )
        if self.keep_coords:
            x_T = structure_array.frac_coords
        if self.keep_lattice:
            if hasattr(structure_array, "lattice"):
                l_T = structure_array.lattice
            else:
                l_T = lattice_params_to_matrix_paddle(
                    structure_array.lengths, structure_array.angles
                )

        time_start = self.beta_scheduler.timesteps
        traj = {
            time_start: {
                "num_atoms": structure_array.num_atoms,
                "atom_types": structure_array.atom_types,
                "frac_coords": x_T % 1.0,
                "lattices": l_T,
            }
        }

        property_emb = self.property_embedding(batch[self.prop_name])
        property_mask = paddle.zeros([batch_size * 2, 1])
        property_mask[:batch_size] = 1

        num_atoms_double = paddle.concat([batch["num_atoms"], batch["num_atoms"]])
        batch_double = paddle.concat([batch["batch"], batch["batch"] + batch_size])
        property_emb_double = paddle.concat([property_emb, property_emb], axis=0)

        for t in tqdm(range(time_start, 0, -1), leave=False, desc="Sampling..."):
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

            # double concat it
            time_emb_double = paddle.concat([time_emb, time_emb], axis=0)
            atom_types = structure_array.atom_types - 1
            atom_types_double = paddle.concat([atom_types, atom_types], axis=0)
            x_t_double = paddle.concat([x_t, x_t], axis=0)
            l_t_double = paddle.concat([l_t, l_t], axis=0)
            pred_l, pred_x, pred_t = self.decoder(
                time_emb_double,
                atom_types_double,
                x_t_double,
                l_t_double,
                num_atoms_double,
                batch_double,
                property_emb=property_emb_double,
                property_mask=property_mask,
            )
            pred_l = (1 + guide_w) * pred_l[:batch_size] - guide_w * pred_l[batch_size:]
            pred_x = (1 + guide_w) * pred_x[: batch["num_nodes"]] - guide_w * pred_x[
                batch["num_nodes"] :
            ]
            pred_t = (1 + guide_w) * pred_t[: batch["num_nodes"]] - guide_w * pred_t[
                batch["num_nodes"] :
            ]

            pred_x = pred_x * paddle.sqrt(x=sigma_norm)
            x_t_minus_05 = (
                x_t - step_size * pred_x + std_x * rand_x
                if not self.keep_coords
                else x_t
            )
            l_t_minus_05 = l_t if not self.keep_lattice else l_t
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

            # double concat it
            x_t_minus_05_double = paddle.concat([x_t_minus_05, x_t_minus_05], axis=0)
            l_t_minus_05_double = paddle.concat([l_t_minus_05, l_t_minus_05], axis=0)
            pred_l, pred_x, pred_t = self.decoder(
                time_emb_double,
                atom_types_double,
                x_t_minus_05_double,
                l_t_minus_05_double,
                num_atoms_double,
                batch_double,
                property_emb=property_emb_double,
                property_mask=property_mask,
            )
            pred_l = (1 + guide_w) * pred_l[:batch_size] - guide_w * pred_l[batch_size:]
            pred_x = (1 + guide_w) * pred_x[: batch["num_nodes"]] - guide_w * pred_x[
                batch["num_nodes"] :
            ]
            pred_t = (1 + guide_w) * pred_t[: batch["num_nodes"]] - guide_w * pred_t[
                batch["num_nodes"] :
            ]

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
            traj[t - 1] = {
                "num_atoms": structure_array.num_atoms,
                "atom_types": structure_array.atom_types,
                "frac_coords": x_t_minus_1 % 1.0,
                "lattices": l_t_minus_1,
            }

        # after all the trajectories are generated, we need to convert them to a
        # list of structures:
        # {
        #     "time_step":[
        #         {
        #             "num_atoms": [...],
        #             "atom_types": [...],
        #             "frac_coords": [...],
        #             "lattices": [...]
        #         }, # sample1 generated by the model
        #         {
        #             "num_atoms": [...],
        #             "atom_types": [...],
        #             "frac_coords": [...],
        #             "lattices": [...]
        #         }, # sample2 generated by the model
        #     ]
        # }
        if is_save_traj:
            for key in traj:
                start_idx = 0
                traj_single = []
                for i in range(batch_size):
                    end_idx = start_idx + traj[key]["num_atoms"][i]
                    traj_single.append(
                        {
                            "num_atoms": traj[key]["num_atoms"][i]
                            .cpu()
                            .numpy()
                            .tolist(),
                            "atom_types": traj[key]["atom_types"][start_idx:end_idx]
                            .cpu()
                            .numpy()
                            .tolist(),
                            "frac_coords": traj[key]["frac_coords"][start_idx:end_idx]
                            .cpu()
                            .numpy()
                            .tolist(),
                            "lattices": traj[key]["lattices"][i].cpu().numpy().tolist(),
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
                        "num_atoms": traj[0]["num_atoms"][i].cpu().numpy().tolist(),
                        "atom_types": traj[0]["atom_types"][start_idx:end_idx]
                        .cpu()
                        .numpy()
                        .tolist(),
                        "frac_coords": traj[0]["frac_coords"][start_idx:end_idx]
                        .cpu()
                        .numpy()
                        .tolist(),
                        "lattices": traj[0]["lattices"][i].cpu().numpy().tolist(),
                    }
                )
                start_idx += traj[0]["num_atoms"][i]
            return {"result": result, "traj": {}}
