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
from ase.io import write
from omegaconf import DictConfig
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.rl.base import ReinL
from ppmat.models.matinvent.rl.models.base import ModelSuite
from ppmat.models.matinvent.rl.training_utils import is_valid_structure
from ppmat.models.matinvent.rl.utils import create_optimizer
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

        # diversity filter
        self.div_filter = div_filter
        self.df_args = df_args

        self.load_model()

    def load_model(self):
        """Load agent and prior models."""
        self.agent = self.model_suite.load_model()
        self.prior = self.model_suite.load_model()

        for param in self.agent.parameters():
            param.trainable = True
        # Freeze the parameter of prior (pretrained) model
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
            # Generate samples using the sampler
            sample_data, sample_struc = self.sampler.generate(
                model=self.agent,
            )

            # Filter invalid samples (basic validity check)
            valid_data = []
            valid_struc = []
            for data, struc in zip(sample_data, sample_struc):
                if self._is_valid_structure(struc):
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

        # save all generated valid structures
        self._save_structures(
            structures=valid_struc,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_valid.extxyz",
        )

        # Filter bad samples by selected metrics (if configured)
        if self.sample_cfg.get("filter"):
            filter_fn = self.sample_cfg.filter
            valid_data, valid_struc, metrics = filter_fn(valid_data, valid_struc, None)
            logging.info(f"Number of filtered samples: {len(valid_struc)}")
        else:
            metrics = {}

        # max sample size to score/reward
        if self.sample_cfg.get("max_num"):
            max_num = self.sample_cfg.max_num
            if len(valid_struc) > max_num:
                valid_data = valid_data[:max_num]
                valid_struc = valid_struc[:max_num]

        # save structures for evaluation
        eval_xyz_path = self._save_structures(
            structures=valid_struc,
            save_dir=self.sample_dir,
            filename=f"step_{self.step:0>4d}_eval.extxyz",
        )

        return valid_data, valid_struc, eval_xyz_path, metrics

    def _is_valid_structure(self, struc: Structure) -> bool:
        """Check if a structure is valid.

        Delegates to training_utils.is_valid_structure for code reuse.
        """
        return is_valid_structure(struc)

    def _save_structures(
        self,
        structures: List[Structure],
        save_dir: str,
        filename: str,
    ) -> str:
        """Save structures to extxyz file.

        Args:
            structures: List of pymatgen Structure objects
            save_dir: Directory to save to
            filename: Output filename

        Returns:
            Path to saved file
        """
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)

        # Convert pymatgen structures to ASE atoms and write
        from pymatgen.io.ase import AseAtomsAdaptor

        adaptor = AseAtomsAdaptor()

        with open(out_path, "w"):
            for struc in structures:
                atoms = adaptor.get_atoms(struc)
                write(out_path, atoms, append=True)

        return out_path

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

        optimizer = create_optimizer(model=self.agent, lr=cfg.lr)
        accum_steps = cfg.accum_steps

        for epoch in range(cfg.epochs):
            self.agent.train()

            loss_all, loss_diff_all, loss_kl_all = 0.0, 0.0, 0.0
            for batch in loader:
                batch_rewards = batch["structure_array"]["reward"]
                adv = batch_rewards

                # batch size for this mini-batch (equivalent to batch.num_graphs in PyG)
                n_graphs = int(batch["structure_array"]["num_atoms"].shape[0])

                loss, loss_diff, loss_kl = 0.0, 0.0, 0.0

                for t in range(cfg.timesteps):
                    #   noised_input = self.agent.add_noise(batch, t)
                    #   agent_pred <- agent.calc_sample_loss(noised_input)
                    #   prior_pred <- prior.calc_sample_loss(noised_input)
                    noised_input = self._add_noise_to_model(self.agent, batch, t)
                    sample_loss, agent_pred = self._calc_sample_loss_from_model(
                        self.agent, noised_input
                    )

                    # Prior uses the SAME noised_input (not re-sampled),
                    # so KL is computed at the same noise level.
                    with paddle.no_grad():
                        _, prior_pred = self._calc_sample_loss_from_model(
                            self.prior, noised_input
                        )

                    # Diffusion loss weighted by reward
                    _loss_diff = adv * sample_loss

                    # KL regularization with adaptive weight:
                    #   High reward -> small KL weight (allowed to deviate from prior)
                    #   Low reward  -> large KL weight (stay close to prior)
                    kl_term = self._calc_kl_reg_from_models(
                        agent_pred, prior_pred, batch
                    )
                    _loss_kl = kl_term * (1.1 - adv)

                    _loss = (_loss_diff + _loss_kl * cfg.sigma).mean() / accum_steps

                    # Backward pass
                    _loss.backward()
                    if (t + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.clear_grad()

                    # Track losses
                    loss += _loss.item() * accum_steps
                    loss_diff += _loss_diff.sum().item()
                    loss_kl += _loss_kl.sum().item()

                # Average losses over timesteps
                loss_diff = loss_diff / cfg.timesteps
                loss_kl = loss_kl / cfg.timesteps
                loss = loss / cfg.timesteps

                # Handle any remaining gradients not yet applied
                if (t + 1) % accum_steps != 0:
                    optimizer.step()
                    optimizer.clear_grad()

                # Accumulate weighted by batch size (original: loss * batch.num_graphs)
                loss_all += loss * n_graphs
                loss_diff_all += loss_diff
                loss_kl_all += loss_kl

            # Log epoch losses
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

        # sample scoring, remove failed samples, ranking and get top k samples
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

        # long-term memory
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
            # Support both experiment trackers (log(dict, step=...)) and stdlib logger.
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

        # diversity filter
        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            [sample_list[p] for p in penalty_idx]
            penalty_strucs = [sample_struc[p] for p in penalty_idx]
            logging.info(f"Diversity filter: tol_n={tol_n}, buff_n={buff_n}")

        # topk data points
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.finetune_cfg.batch_size * self.topk_ratio)]
        [sample_list[_i] for _i in topk_idx]
        strucs_topk = [sample_struc[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        # experience replay
        # ppmat 里对应的是 pymatgen Structure（RLDataset 期望 Structure，
        # 不是 dict），因此一律使用 strucs_topk（Structure 列表），
        # replay buffer 的 "data" 列也存 Structure
        if self.replay is not None:
            if self.div_filter and len(penalty_strucs) > 0:
                self.replay.memory_purge(penalty_strucs)
            data_replay, reward_replay = self.replay.sample()  # Structure list
            ft_data = strucs_topk + data_replay
            ft_reward = np.concatenate((reward_topk, reward_replay))
            # 存 strucs_topk 而非 sample_topk，使 replay.sample() 返回 Structure 列表
            self.replay.extend(strucs_topk, strucs_topk, reward_topk)
            logging.info(f"replay buffer size={len(self.replay)}")
            logging.info(
                f"buffer reward mean=" f"{self.replay.buffer['reward'].values.mean()}"
            )
        else:
            ft_data = strucs_topk  # Structure 列表，RLDataset 可正确处理
            ft_reward = reward_topk

        # finetuning
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
            # Save the agent weights every few iterations
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f"loop_{step:0>4d}")
                self.model_suite.save_model(self.agent, ckpt_dir)
        # If the entire training finishes, clean up
        ckpt_dir = os.path.join(self.models_dir, "final")
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info("*****   RL END   *****")
        end_time = time.time()
        logging.info(f"Total time taken: {int(end_time - start_time)} s.")

    def _add_noise_to_model(self, model, batch, timestep: int):
        """Add noise to batch for reinforcement learning fine-tuning.

        This code is adapted from:

        Args:
            model: The diffusion model (agent or prior)
            batch: Input batch data structure
            timestep: Diffusion timestep (0 to num_train_timesteps-1)

        Returns:
            Tuple of (noisy_batch, clean_batch, timesteps)
        """
        # DiffCSP 等模型自带与自身 scheduler 兼容的 add_noise()，直接委托
        if callable(getattr(model, "add_noise", None)):
            return model.add_noise(batch, timestep)

        structure_array = batch["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )

        # Normalize timestep to [0, max_t] range
        N = model.num_train_timesteps
        max_t = model.max_t if hasattr(model, "max_t") else 1.0
        time_list = paddle.linspace(max_t, 1.0 / N, N)
        t = paddle.full([batch_size], time_list[timestep])

        # Pre-corruption: normalize frac_coords to [0, 1)
        frac_coords = structure_array["frac_coords"] % 1.0

        # Add noise to coordinates
        rand_x = paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)
        input_frac_coords = model.coord_scheduler.add_noise(
            frac_coords, rand_x, timesteps=t, batch_idx=batch_idx, num_atoms=num_atoms
        )

        # Add noise to lattice
        if "lattice" in structure_array.keys():
            lattices = structure_array["lattice"]
        else:
            from ppmat.models.mattergen.mattergen import lattice_params_to_matrix_paddle

            lattices = lattice_params_to_matrix_paddle(
                structure_array["lengths"], structure_array["angles"]
            )
        rand_l = paddle.randn(shape=lattices.shape, dtype=lattices.dtype)
        from ppmat.models.mattergen.mattergen import (  # isort: skip
            make_noise_symmetric_preserve_variance,
        )

        rand_l = make_noise_symmetric_preserve_variance(rand_l)
        input_lattice = model.lattice_scheduler.add_noise(
            lattices, rand_l, timesteps=t, num_atoms=num_atoms
        )

        # Add noise to atom types
        atom_type = structure_array["atom_types"]
        atom_type_zero_based = atom_type - 1
        input_atom_type_zero_based = model.atom_scheduler.add_noise(
            atom_type_zero_based, timesteps=t, batch_idx=batch_idx
        )
        input_atom_type = input_atom_type_zero_based + 1

        # Create noisy batch structure
        noisy_batch = {
            "structure_array": {
                "frac_coords": input_frac_coords,
                "lattice": input_lattice,
                "atom_types": input_atom_type,
                "num_atoms": num_atoms,
            },
            "batch_idx": batch_idx,
            # Store noise / clean tensors for loss calculation
            "rand_l": rand_l,
            "rand_x": rand_x,
            "clean_frac_coords": frac_coords,
            "atom_type_zero_based": atom_type_zero_based,
            "input_atom_type_zero_based": input_atom_type_zero_based,
        }

        # Return tuple format expected by calc_sample_loss
        return noisy_batch, batch, t

    def _calc_sample_loss_from_model(self, model, noised_input):
        """Calculate sample loss for reinforcement learning fine-tuning.

        This code is adapted from:

        Args:
            model: The diffusion model (agent or prior)
            noised_input: Tuple of (noisy_batch, clean_batch, timesteps)

        Returns:
            Tuple of (loss, prediction_dict)
        """
        # DiffCSP 等模型自带与自身 scheduler 兼容的 calc_sample_loss()，直接委托
        if callable(getattr(model, "calc_sample_loss", None)):
            return model.calc_sample_loss(noised_input)

        noisy_batch, clean_batch, t = noised_input

        batch_idx = noisy_batch["batch_idx"]
        structure_array_noisy = noisy_batch["structure_array"]
        num_atoms = structure_array_noisy["num_atoms"]
        batch_size = num_atoms.shape[0]

        # Build the noise_batch dict that the model expects
        noise_batch = {
            "frac_coords": structure_array_noisy["frac_coords"],
            "lattice": structure_array_noisy["lattice"],
            "atom_types": structure_array_noisy["atom_types"],
            "num_atoms": num_atoms,
            "batch": batch_idx,
        }

        # Backward compatibility:
        # - legacy path: decoder(...) -> tuple
        # - current MatterGen path: model.model(x, t) -> dict
        if hasattr(model, "decoder"):
            eps_pos, lattice_update, atom_type_logits = model.decoder(
                z=model.noise_level_encoding(t),
                frac_coords=noise_batch["frac_coords"],
                atom_types=noise_batch["atom_types"],
                num_atoms=noise_batch["num_atoms"],
                batch=noise_batch["batch"],
                lattice=noise_batch["lattice"],
            )
        else:
            score_model_output = model.model(noise_batch, t)
            eps_pos = score_model_output["frac_coords"]
            lattice_update = score_model_output["lattice"]
            atom_type_logits = score_model_output["atom_types"]

        # Retrieve stored noise/clean tensors from add_noise
        rand_l = noisy_batch["rand_l"]
        clean_frac_coords = noisy_batch["clean_frac_coords"]
        atom_type_zero_based = noisy_batch["atom_type_zero_based"]
        input_atom_type_zero_based = noisy_batch["input_atom_type_zero_based"]

        # ----- coord loss: wrapped_normal_loss (score matching) -----
        from ppmat.models.mattergen.mattergen import wrapped_normal_loss

        clean_structure_array = clean_batch["structure_array"]
        loss_coord = wrapped_normal_loss(
            corruption=model.coord_scheduler,
            score_model_output=eps_pos,
            t=t,
            batch_idx=batch_idx,
            batch_size=batch_size,
            x=clean_frac_coords,
            noisy_x=structure_array_noisy["frac_coords"],
            reduce="sum",
            batch=clean_structure_array,
        )

        # ----- lattice loss: (pred + rand_l)^2 -----
        loss_lattice = (lattice_update + rand_l).square().mean(axis=[1, 2])

        # ----- atom type loss: D3PM -----
        loss_atom_type, _, _ = model.atom_scheduler.compute_loss(
            score_model_output=atom_type_logits,
            t=t,
            batch_idx=batch_idx,
            batch_size=batch_size,
            x=atom_type_zero_based,
            noisy_x=input_atom_type_zero_based,
            reduce="sum",
            d3pm_hybrid_lambda=(
                model.d3pm_hybrid_lambda
                if hasattr(model, "d3pm_hybrid_lambda")
                else None
            ),
        )

        # Weighted per-sample loss (same weights as training)
        coord_weight = getattr(model, "coord_loss_weight", 0.1)
        lattice_weight = getattr(model, "lattice_loss_weight", 1.0)
        atom_weight = getattr(model, "atom_loss_weight", 1.0)

        total_loss = (
            coord_weight * loss_coord
            + lattice_weight * loss_lattice
            + atom_weight * loss_atom_type
        )

        # Prediction dict for KL regularization
        prediction_dict = {
            "pos": eps_pos,
            "cell": lattice_update,
            "atomic_numbers": atom_type_logits,
        }

        return total_loss, prediction_dict

    def _calc_kl_reg_from_models(self, agent_pred, prior_pred, batch):
        """Calculate KL divergence regularization for reinforcement learning.


        Uses paddle.scatter to replace torch_scatter.

        Args:
            agent_pred: Prediction dict from agent model
                with keys 'pos', 'cell', 'atomic_numbers'
            prior_pred: Prediction dict from prior
                (frozen) model with same keys
            batch: Input batch data

        Returns:
            KL divergence loss per sample (tensor of shape batch_size)
        """
        # DiffCSP 等模型自带与自身 prediction_dict 格式匹配的 calc_kl_reg()，直接委托
        if callable(getattr(self.agent, "calc_kl_reg", None)):
            return self.agent.calc_kl_reg(agent_pred, prior_pred, batch)

        # Extract predictions from agent and prior
        pred_x, pred_l, pred_t = (
            agent_pred["pos"],
            agent_pred["cell"],
            agent_pred["atomic_numbers"],
        )
        pred_x_p, pred_l_p, pred_t_p = (
            prior_pred["pos"].detach(),
            prior_pred["cell"].detach(),
            prior_pred["atomic_numbers"].detach(),
        )

        # Get batch index for aggregation
        if "batch_idx" in batch:
            batch_idx = batch["batch_idx"]
        elif "structure_array" in batch:
            structure_array = batch["structure_array"]
            num_atoms = structure_array["num_atoms"]
            batch_size = num_atoms.shape[0]
            batch_idx = paddle.repeat_interleave(
                paddle.arange(batch_size), repeats=num_atoms
            )
        else:
            raise ValueError("Cannot find batch index in batch input")

        # Compute KL divergence for lattice (per sample)
        kl_term0 = paddle.pow(pred_l - pred_l_p, 2).mean(axis=(1, 2))

        # Compute KL divergence for positions (per atom, then aggregate to per sample)
        x_ap = paddle.pow(pred_x - pred_x_p, 2).mean(axis=1)
        kl_term1 = scatter(x_ap, batch_idx, dim=0, reduce="mean")

        # Compute KL divergence for atom types (per atom, then aggregate to per sample)
        t_ap = paddle.pow(pred_t - pred_t_p, 2).mean(axis=1)
        kl_term2 = scatter(t_ap, batch_idx, dim=0, reduce="mean")

        # Total KL divergence is sum of all three terms
        kl_term = kl_term0 + kl_term1 + kl_term2

        return kl_term
