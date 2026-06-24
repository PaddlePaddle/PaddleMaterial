# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MatInvent main class for reinforcement learning.


"""

import logging
import os
import time
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import paddle
from omegaconf import DictConfig
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl.base import ReinL
from ppmat.models.matinvent.rl.models.base import ModelSuite
from ppmat.models.matinvent.rl.training_utils import is_valid_structure
from ppmat.models.matinvent.rl.training_utils import save_structures
from ppmat.utils.scatter import scatter


class MatInvent(ReinL):
    """MatInvent reinforcement learning pipeline for material generation."""

    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        topk_ratio: float,
        save_dir: str,
        save_freq: int = 50,
        device: str = None,
        logger=None,
        replay: bool = False,
        replay_args: Dict = None,
        div_filter: bool = False,
        df_args: Dict = None,
        **kwargs,
    ) -> None:
        super().__init__(
            rl_epoch=rl_epoch,
            model_suite=model_suite,
            reward=reward,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            save_dir=save_dir,
            save_freq=save_freq,
            device=device,
            logger=logger,
            replay=replay,
            replay_args=replay_args,
            **kwargs,
        )
        assert topk_ratio > 0.0 and topk_ratio <= 1.0
        self.topk_ratio = topk_ratio

        self.div_filter = div_filter
        self.df_args = df_args

        self.load_model()

    def load_model(self):
        """Load agent and prior models."""
        self.agent = self.model_suite.load_model()
        self.prior = self.model_suite.load_model()

        for param in self.agent.parameters():
            param.trainable = True
        for param in self.prior.parameters():
            param.stop_gradient = True

    def sample_step(self) -> Tuple[List, List[Structure], str, Dict]:
        """Generate samples using the agent model.

        Returns:
            Tuple of (sample_data, sample_struc, eval_xyz_path, metrics)
        """
        max_retry = int(self.sample_cfg.get("max_retry", 3))
        valid_data = []
        valid_struc = []
        for attempt in range(max_retry):
            sample_data, sample_struc = self.sampler.generate(
                model=self.agent,
            )

            valid_data = []
            valid_struc = []
            for data, struc in zip(sample_data, sample_struc):
                if is_valid_structure(struc):
                    valid_data.append(data)
                    valid_struc.append(struc)

            if len(valid_struc) > 0:
                break

            logging.warning(
                f"No valid structures generated on attempt {attempt + 1}/{max_retry}."
            )

        if len(valid_struc) == 0:
            logging.warning("No valid structures generated after retries!")
            return [], [], "", {}

        save_structures(
            structures=valid_struc,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_valid.extxyz",
        )

        if self.sample_cfg.get("filter"):
            filter_fn = self.sample_cfg.filter
            valid_data, valid_struc, metrics = filter_fn(valid_data, valid_struc, None)
            logging.info(f"Number of filtered samples: {len(valid_struc)}")
        else:
            metrics = {}

        if self.sample_cfg.get("max_num"):
            max_num = self.sample_cfg.max_num
            if len(valid_struc) > max_num:
                valid_data = valid_data[:max_num]
                valid_struc = valid_struc[:max_num]

        eval_xyz_path = save_structures(
            structures=valid_struc,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_eval.extxyz",
        )

        return valid_data, valid_struc, eval_xyz_path, metrics

    def ft_step(self, data_list: List, rewards: np.ndarray, baseline: float):
        """Fine-tune the agent model on high-reward samples.

        Args:
            data_list: List of data samples (structures)
            rewards: Array of reward values
            baseline: Baseline reward for advantage calculation
        """
        cfg = self.finetune_cfg
        loader = self.model_suite.get_dataloader(
            samples=data_list,
            rewards=rewards,
            batch_size=cfg.batch_size,
        )

        optimizer = paddle.optimizer.Adam(learning_rate=cfg.lr, parameters=self.agent.parameters())
        accum_steps = cfg.accum_steps

        for epoch in range(cfg.epochs):
            self.agent.train()

            loss_all, loss_diff_all, loss_kl_all = 0.0, 0.0, 0.0
            for batch in loader:
                batch_rewards = batch["structure_array"]["reward"]
                adv = batch_rewards

                n_graphs = int(batch["structure_array"]["num_atoms"].shape[0])

                loss, loss_diff, loss_kl = 0.0, 0.0, 0.0

                for t in range(cfg.timesteps):
                    noised_input = self._add_noise_to_model(self.agent, batch, t)
                    sample_loss, agent_pred = self._calc_sample_loss_from_model(
                        self.agent, noised_input
                    )

                    with paddle.no_grad():
                        _, prior_pred = self._calc_sample_loss_from_model(
                            self.prior, noised_input
                        )

                    _loss_diff = adv * sample_loss

                    kl_term = self._calc_kl_reg_from_models(
                        agent_pred, prior_pred, batch
                    )
                    _loss_kl = kl_term * (1.1 - adv)

                    _loss = (_loss_diff + _loss_kl * cfg.sigma).mean() / accum_steps

                    _loss.backward()
                    if (t + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.clear_grad()

                    loss += _loss.item() * accum_steps
                    loss_diff += _loss_diff.sum().item()
                    loss_kl += _loss_kl.sum().item()

                loss_diff = loss_diff / cfg.timesteps
                loss_kl = loss_kl / cfg.timesteps
                loss = loss / cfg.timesteps

                if (t + 1) % accum_steps != 0:
                    optimizer.step()
                    optimizer.clear_grad()

                loss_all += loss * n_graphs
                loss_diff_all += loss_diff
                loss_kl_all += loss_kl

            loss_dict = {
                "loss": loss_all / len(data_list),
                "loss_diff": loss_diff_all / len(data_list),
                "loss_kl": loss_kl_all / len(data_list),
            }
            log_str = [f"{k}: {v:.4f}" for k, v in loss_dict.items()]
            logging.info(f"Epoch {epoch}: " + ", ".join(log_str))

    def rl_step(self):
        """Execute one reinforcement learning step."""
        logging.info(f"*****   LOOP {self.step} START   *****")
        start_time = time.time()

        logging.info("SAMPLE:")
        sample_list, sample_struc, xyz_path, sample_metrics = self.sample_step()

        logging.info("SCORE:")
        sample_list, sample_struc, rewards, prop_dict = self.reward_step(
            sample_list,
            sample_struc,
            xyz_path,
            f"step_{self.step:0>4d}",
        )

        if len(rewards) > 0:
            log_dict = {f"{k} mean": v.mean() for k, v in prop_dict.items()}
            log_dict.update({f"{k} std": v.std() for k, v in prop_dict.items()})
            log_dict.update(
                {
                    "reward mean": rewards.mean(),
                    "reward std": rewards.std(),
                }
            )
        else:
            log_dict = {f"{k} mean": float("nan") for k in prop_dict.keys()}
            log_dict.update({f"{k} std": float("nan") for k in prop_dict.keys()})
            log_dict.update(
                {
                    "reward mean": float("nan"),
                    "reward std": float("nan"),
                }
            )
        log_dict.update(sample_metrics)

        if len(rewards) > 0:
            self.ltm.extend(sample_struc, rewards, self.step)
        else:
            logging.warning(
                "No successful rewards in this loop, "
                "skip memory/replay/finetune updates."
            )
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, "long_term_memory.csv"))
        logging.info(
            f"{len(self.ltm)} crystals generated so far, "
            + f"{len(self.ltm.unique_comps)} unique components."
            + f"  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}."
        )
        log_dict.update(
            {
                "crystal_num": len(self.ltm),
                "unique_comps": len(self.ltm.unique_comps),
                "burden": metrics[0],
                "div_ratio": metrics[1],
                "cost": self.cost,
            }
        )
        if self.logger is not None:
            try:
                self.logger.log(log_dict, step=self.step)
            except TypeError:
                if hasattr(self.logger, "info"):
                    self.logger.info(f"step={self.step} metrics={log_dict}")

        if len(rewards) == 0:
            if self.replay is not None and len(self.replay) > 0:
                logging.info("FINETUNE (replay fallback):")
                data_replay, reward_replay = self.replay.sample()
                if len(data_replay) > 0:
                    reward_replay = np.asarray(reward_replay, dtype=float)
                    baseline = self.ltm.get_baseline(self.step)
                    baseline = min(baseline, reward_replay.min())
                    self.ft_step(data_replay, reward_replay, baseline)
            end_time = time.time()
            total_time = (end_time - start_time) / 60
            logging.info(f"*****   LOOP {self.step} FINISH   *****")
            logging.info(f"Total time taken: {total_time:.2f} min.\n\n")
            return

        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            penalty_strucs = [sample_struc[p] for p in penalty_idx]
            logging.info(f"Diversity filter: tol_n={tol_n}, buff_n={buff_n}")

        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.finetune_cfg.batch_size * self.topk_ratio)]
        strucs_topk = [sample_struc[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        if self.replay is not None:
            if self.div_filter and len(penalty_strucs) > 0:
                self.replay.memory_purge(penalty_strucs)
            data_replay, reward_replay = self.replay.sample()  # Structure list
            ft_data = strucs_topk + data_replay
            ft_reward = np.concatenate((reward_topk, reward_replay))
            self.replay.extend(strucs_topk, strucs_topk, reward_topk)
            logging.info(f"replay buffer size={len(self.replay)}")
            logging.info(
                f"buffer reward mean=" f"{self.replay.buffer['reward'].values.mean()}"
            )
        else:
            ft_data = strucs_topk
            ft_reward = reward_topk

        logging.info("FINETUNE:")
        baseline = self.ltm.get_baseline(self.step)
        baseline = min(baseline, ft_reward.min())
        self.ft_step(ft_data, ft_reward, baseline)

        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f"*****   LOOP {self.step} FINISH   *****")
        logging.info(f"Total time taken: {total_time:.2f} min.\n\n")

    def run_rl(self):
        """Run the full reinforcement learning loop."""
        logging.info("*****   RL START   *****")
        start_time = time.time()

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f"loop_{step:0>4d}")
                self.model_suite.save_model(self.agent, ckpt_dir)
        ckpt_dir = os.path.join(self.models_dir, "final")
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info("*****   RL END   *****")
        end_time = time.time()
        logging.info(f"Total time taken: {int(end_time - start_time)} s.")

    def _add_noise_to_model(self, model, batch, timestep: int):
        structure_array = batch["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(paddle.arange(batch_size), repeats=num_atoms)
        N = model.num_train_timesteps
        has_atom = hasattr(model, "atom_scheduler")

        if has_atom:
            max_t = model.max_t if hasattr(model, "max_t") else 1.0
            t = paddle.full([batch_size], paddle.linspace(max_t, 1.0 / N, N)[timestep])
        else:
            t = paddle.full([batch_size], timestep, dtype="int64")

        frac_coords = structure_array["frac_coords"] % 1.0
        rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)

        if has_atom:
            input_frac_coords = model.coord_scheduler.add_noise(
                frac_coords, rand_x, timesteps=t, batch_idx=batch_idx, num_atoms=num_atoms
            )
        else:
            input_frac_coords = model.coord_scheduler.add_noise(
                frac_coords, rand_x, timesteps=t.repeat_interleave(repeats=num_atoms)
            )
            input_frac_coords = input_frac_coords % 1.0

        if "lattice" in structure_array:
            lattices = structure_array["lattice"]
        else:
            from ppmat.models.mattergen.mattergen import lattice_params_to_matrix_paddle
            lattices = lattice_params_to_matrix_paddle(structure_array["lengths"], structure_array["angles"])
        rand_l = paddle.randn(shape=lattices.shape, dtype=lattices.dtype)

        if has_atom:
            from ppmat.models.mattergen.mattergen import make_noise_symmetric_preserve_variance
            rand_l = make_noise_symmetric_preserve_variance(rand_l)
        ls_kwargs = {"timesteps": t}
        if has_atom:
            ls_kwargs["num_atoms"] = num_atoms
        input_lattice = model.lattice_scheduler.add_noise(lattices, rand_l, **ls_kwargs)

        noisy_batch = {"structure_array": {"frac_coords": input_frac_coords, "lattice": input_lattice, "num_atoms": num_atoms}, "batch_idx": batch_idx, "rand_l": rand_l, "rand_x": rand_x, "clean_frac_coords": frac_coords}

        if has_atom:
            atom_type = structure_array["atom_types"]
            atom_type_zero_based = atom_type - 1
            input_atom_type_zero_based = model.atom_scheduler.add_noise(atom_type_zero_based, timesteps=t, batch_idx=batch_idx)
            input_atom_type = input_atom_type_zero_based + 1
            noisy_batch["structure_array"]["atom_types"] = input_atom_type
            noisy_batch["atom_type_zero_based"] = atom_type_zero_based
            noisy_batch["input_atom_type_zero_based"] = input_atom_type_zero_based
        else:
            noisy_batch["structure_array"]["atom_types"] = structure_array["atom_types"]

        return noisy_batch, batch, t

    def _calc_sample_loss_from_model(self, model, noised_input):
        noisy_batch, clean_batch, t = noised_input
        batch_idx = noisy_batch["batch_idx"]
        sn = noisy_batch["structure_array"]
        num_atoms = sn["num_atoms"]
        has_atom = hasattr(model, "atom_scheduler")

        if has_atom:
            nb = {"frac_coords": sn["frac_coords"], "lattice": sn["lattice"],
                  "atom_types": sn["atom_types"], "num_atoms": num_atoms, "batch": batch_idx}
            o = model.model(nb, t)
            eps_pos, lattice_update = o["frac_coords"], o["lattice"]
        else:
            pred_l, eps_pos = model.decoder(
                model.time_embedding(t), sn["atom_types"] - 1,
                sn["frac_coords"], sn["lattice"], num_atoms, batch_idx,
            )
            lattice_update = pred_l

        rand_l = noisy_batch["rand_l"]
        clean_frac_coords = noisy_batch["clean_frac_coords"]

        if has_atom:
            loss_lattice = (lattice_update + rand_l).square().mean(axis=[1, 2])
            from ppmat.models.mattergen.mattergen import wrapped_normal_loss
            loss_coord = wrapped_normal_loss(
                corruption=model.coord_scheduler, score_model_output=eps_pos, t=t,
                batch_idx=batch_idx, batch_size=num_atoms.shape[0],
                x=clean_frac_coords, noisy_x=sn["frac_coords"],
                reduce="sum", batch=clean_batch["structure_array"],
            )
            atom_type_zero_based = noisy_batch["atom_type_zero_based"]
            input_atom_type_zero_based = noisy_batch["input_atom_type_zero_based"]
            loss_atom_type, _, _ = model.atom_scheduler.compute_loss(
                score_model_output=sn["atom_types"], t=t, batch_idx=batch_idx,
                batch_size=num_atoms.shape[0], x=atom_type_zero_based,
                noisy_x=input_atom_type_zero_based, reduce="sum",
                d3pm_hybrid_lambda=getattr(model, "d3pm_hybrid_lambda", None),
            )
            cw = getattr(model, "coord_loss_weight", 0.1)
            lw = getattr(model, "lattice_loss_weight", 1.0)
            aw = getattr(model, "atom_loss_weight", 1.0)
            total_loss = cw * loss_coord + lw * loss_lattice + aw * loss_atom_type
            prediction_dict = {"pos": eps_pos, "cell": lattice_update, "atomic_numbers": sn["atom_types"]}
        else:
            loss_lattice = (lattice_update - rand_l).square().mean(axis=[1, 2])
            loss_coord = paddle.pow(eps_pos - clean_frac_coords, 2).mean(axis=1)
            loss_coord = scatter(loss_coord, batch_idx, dim=0, reduce="mean")
            cw = getattr(model, "coord_loss_weight", 1.0)
            lw = getattr(model, "lattice_loss_weight", 1.0)
            total_loss = cw * loss_coord + lw * loss_lattice
            prediction_dict = {"coords": eps_pos, "lattice": lattice_update}

        return total_loss, prediction_dict

    def _calc_kl_reg_from_models(self, agent_pred, prior_pred, batch):
        if "batch_idx" in batch:
            batch_idx = batch["batch_idx"]
        elif "structure_array" in batch:
            num_atoms = batch["structure_array"]["num_atoms"]
            batch_idx = paddle.repeat_interleave(paddle.arange(num_atoms.shape[0]), repeats=num_atoms)
        else:
            raise ValueError("Cannot find batch index in batch input")

        if "pos" in agent_pred:
            pred_x, pred_l = agent_pred["pos"], agent_pred["cell"]
            pred_x_p, pred_l_p = prior_pred["pos"].detach(), prior_pred["cell"].detach()
        else:
            pred_x, pred_l = agent_pred["coords"], agent_pred["lattice"]
            pred_x_p, pred_l_p = prior_pred["coords"].detach(), prior_pred["lattice"].detach()

        kl_lattice = paddle.pow(pred_l - pred_l_p, 2).mean(axis=(1, 2))
        x_ap = paddle.pow(pred_x - pred_x_p, 2).mean(axis=1)
        kl_coord = scatter(x_ap, batch_idx, dim=0, reduce="mean")
        kl_term = kl_lattice + kl_coord

        if "pos" in agent_pred:
            pred_t = agent_pred["atomic_numbers"]
            pred_t_p = prior_pred["atomic_numbers"].detach()
            t_ap = paddle.pow(pred_t - pred_t_p, 2).mean(axis=1)
            kl_term += scatter(t_ap, batch_idx, dim=0, reduce="mean")

        return kl_term
