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

from collections import defaultdict
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.chemeleon2.common import apply_lora_to_linear
from ppmat.models.chemeleon2.common import print_trainable_parameters
from ppmat.models.chemeleon2.common import merge_lora_weights
from ppmat.models.chemeleon2.common import to_dense_batch
from ppmat.models.chemeleon2.ldm_module.diffusion import create_diffusion
from ppmat.models.chemeleon2.vae_module.vae import VAEModule
from ppmat.utils.crystal import lattice_params_to_matrix_paddle


class LDMModule(nn.Layer):
    def __init__(
        self,
        normalize_latent=True,
        denoiser=None,
        augmentation=None,
        diffusion_configs=None,
        optimizer=None,
        scheduler=None,
        condition_module=None,
        vae=None,
        vae_ckpt_path=None,
        ldm_ckpt_path=None,
        lora_configs=None,
    ):
        super().__init__()

        # Build nested models if they are config dicts
        from ppmat.models import build_model
        if isinstance(denoiser, dict):
            self.denoiser = build_model(denoiser)
        else:
            self.denoiser = denoiser

        if isinstance(condition_module, dict):
            self.condition_module = build_model(condition_module)
        else:
            self.condition_module = condition_module

        if isinstance(vae, dict):
            vae = build_model(vae)

        self.normalize_latent = normalize_latent
        self.latent_std = paddle.to_tensor([1.0])

        self.diffusion_configs = diffusion_configs
        self.augmentation = augmentation
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler
        self.lora_configs = lora_configs

        if diffusion_configs is not None:
            self.diffusion = create_diffusion(**diffusion_configs)
        else:
            self.diffusion = None

        if vae is not None:
            self.vae = vae
            for param in self.vae.parameters():
                param.stop_gradient = True
            self.vae.eval()
        elif vae_ckpt_path is not None:
            self.vae = VAEModule.load_checkpoint(vae_ckpt_path)[0]
            for param in self.vae.parameters():
                param.stop_gradient = True
            self.vae.eval()

        if ldm_ckpt_path is not None:
            checkpoint = paddle.load(ldm_ckpt_path)
            self.set_state_dict(checkpoint['model_state_dict'])
            if 'latent_std' in checkpoint:
                self.latent_std = checkpoint['latent_std']
        
        if lora_configs is not None:
            rank = lora_configs.get('r', 8)
            alpha = lora_configs.get('lora_alpha', 16)
            dropout = lora_configs.get('lora_dropout', 0.0)
            target_modules = lora_configs.get('target_modules', None)
            
            self.denoiser = apply_lora_to_linear(
                self.denoiser,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules
            )
            print_trainable_parameters(self.denoiser)
            
        self.use_cfg = False
        if condition_module is not None:
            self.use_cfg = True
            self.condition_module = condition_module

    def _to_dense_batch(self, x, batch_idx):
        return to_dense_batch(x, batch_idx)
    
    def _apply_augmentation(self, batch):
        if self.augmentation is None:
            return batch
        return batch

    def forward(self, batch):
        """Forward pass for training compatibility.

        This method converts the standard dictionary format to CrystalBatch format
        and then calls calculate_loss. This provides compatibility with the
        standard training framework.

        Args:
            batch: Input batch data (dict with 'structure_array' key)

        Returns:
            dict: Contains 'loss_dict' with training losses (tensors for backward pass)
        """
        # Convert dict format to CrystalBatch format
        crystal_batch = self._dict_to_crystal_batch(batch)
        loss_dict = self.calculate_loss(crystal_batch, training=True)

        # The framework needs loss_dict with tensor values for backward pass
        # Don't detach or convert to scalars - keep tensors as-is
        return {"loss_dict": loss_dict}

    def _dict_to_crystal_batch(self, batch):
        """Convert dictionary format to CrystalBatch format.

        Args:
            batch: Dict with 'structure_array' key containing structure data

        Returns:
            CrystalBatch: Converted batch object
        """
        from ppmat.models.chemeleon2.common.schema import CrystalBatch

        structure_array = batch["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        total_atoms = num_atoms.sum().item()

        # Create CrystalBatch from structure_array
        crystal_batch = CrystalBatch()
        crystal_batch.atom_types = structure_array["atom_types"]
        crystal_batch.num_atoms = num_atoms
        crystal_batch.batch = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )

        # Handle frac_coords (required for training, optional for sampling)
        if "frac_coords" in structure_array:
            crystal_batch.frac_coords = structure_array["frac_coords"]
        else:
            # For sampling, generate random fractional coords
            crystal_batch.frac_coords = paddle.rand([total_atoms, 3])

        # Handle lattice - VAE encoder expects lattices (matrix format)
        if "lattice" in structure_array:
            crystal_batch.lattices = structure_array["lattice"]
        elif "lengths" in structure_array and "angles" in structure_array:
            # Convert lengths + angles to lattice matrix
            crystal_batch.lattices = lattice_params_to_matrix_paddle(
                structure_array["lengths"], structure_array["angles"]
            )
        else:
            # For sampling, generate random lattice parameters
            crystal_batch.lengths = paddle.rand([batch_size, 3]) * 10 + 5
            crystal_batch.angles = paddle.rand([batch_size, 3]) * 60 + 60
            crystal_batch.lattices = lattice_params_to_matrix_paddle(
                crystal_batch.lengths, crystal_batch.angles
            )

        # Additional fields
        crystal_batch.num_nodes = total_atoms
        crystal_batch.num_graphs = batch_size
        crystal_batch.token_idx = paddle.concat([
            paddle.arange(n) for n in num_atoms
        ])

        return crystal_batch

    def calculate_loss(self, batch, training=True):
        if not hasattr(self, 'vae') or self.vae is None:
            raise ValueError("VAE must be loaded before training. Set vae_ckpt_path in __init__.")
        
        if training and self.augmentation is not None:
            batch = self._apply_augmentation(batch)
        
        with paddle.no_grad():
            encoded = self.vae.encode(batch)
            x = encoded["posterior"].sample() / self.latent_std
            x, mask = self._to_dense_batch(x, encoded["batch"])
        
        t = paddle.randint(0, self.diffusion.num_timesteps, shape=[x.shape[0]], dtype='int64')
        
        y = None
        if self.use_cfg:
            y = batch.get("y")
            assert y is not None, "Batch must contain 'y' key when use_cfg=True"
            y = self.condition_module(y, training=training)
        
        model_kwargs = {"mask": mask, "y": y}
        loss_dict = self.diffusion.training_losses(
            model=self.denoiser,
            x_start=x,
            t=t,
            model_kwargs=model_kwargs,
        )
        # Convert all losses to scalar tensors (mean over all elements)
        for key in list(loss_dict.keys()):
            if paddle.is_tensor(loss_dict[key]):
                loss_dict[key] = loss_dict[key].mean()

        loss_dict["total_loss"] = loss_dict.get("loss", paddle.to_tensor([0.0]))
        return loss_dict

    def sample(
        self,
        batch,
        sampler="ddim",
        sampling_steps=50,
        eta=1.0,
        cfg_scale=2.0,
        return_atoms=False,
        return_structure=False,
        collect_trajectory=False,
        return_trajectory=False,
        progress=True,
    ):
        # Convert dict format to CrystalBatch format if needed
        if isinstance(batch, dict):
            batch = self._dict_to_crystal_batch(batch)

        if sampler == "ddim":
            timestep_respacing = "ddim" + str(sampling_steps)
        else:
            timestep_respacing = str(sampling_steps)

        sampling_configs = self.diffusion_configs.copy()
        sampling_configs.update(timestep_respacing=timestep_respacing)
        sampling_diffusion = create_diffusion(**sampling_configs)

        sampler_fn = (
            partial(sampling_diffusion.ddim_sample_loop, eta=eta)
            if sampler == "ddim"
            else sampling_diffusion.p_sample_loop
        )

        if progress:
            print(f"Using {sampler} sampler with {sampling_diffusion.num_timesteps} timesteps.")

        if not hasattr(self, 'vae') or self.vae is None:
            raise ValueError("VAE must be loaded before sampling. Set vae_ckpt_path in __init__.")

        if isinstance(batch.num_nodes, list):
            num_nodes = sum(batch.num_nodes)
        elif isinstance(batch.num_nodes, (int, np.integer)):
            num_nodes = int(batch.num_nodes)
        else:
            num_nodes = int(batch.num_nodes.item())
        z = paddle.randn([num_nodes, self.vae.latent_dim])
        z, mask = self._to_dense_batch(z, batch.batch)
        
        y = None
        if self.use_cfg:
            y = batch.get("y")
            assert y is not None, "Batch must contain 'y' key when use_cfg=True"
            z = paddle.concat([z, z], axis=0)
            mask = paddle.concat([mask, mask], axis=0)
            y = self.condition_module(y, training=False)
        
        model_kwargs = {
            "mask": mask,
            "y": y,
        }
        if self.use_cfg:
            model_kwargs["cfg_scale"] = cfg_scale
        
        trajectory = defaultdict(list)
        if collect_trajectory:
            trajectory["z"].append(z)
        
        model_fn = self.denoiser.forward_with_cfg if self.use_cfg else self.denoiser.forward
        
        diffusion_out = sampler_fn(
            model=model_fn,
            shape=z.shape,
            noise=z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
        )
        
        if self.use_cfg:
            diffusion_out, _ = paddle.chunk(diffusion_out, 2, axis=0)
            mask, _ = paddle.chunk(mask, 2, axis=0)
        
        diffusion_out = diffusion_out * self.latent_std
        
        encoded_batch = {
            "x": diffusion_out[mask],
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }
        decoder_out = self.vae.decode(encoded_batch)
        batch_rec = self.vae.reconstruct(decoder_out, batch)
        batch_rec.mask = mask

        if return_trajectory:
            return trajectory
        if return_atoms:
            return batch_rec.to_atoms()
        elif return_structure:
            return batch_rec.to_structures()

        # Convert CrystalBatch to dict format list for BuildStructure compatibility
        # This avoids the buggy code path in BuildStructure.__call__ (else branch)
        structures_list = batch_rec._split_by_batch_index()
        # Convert to the format expected by BuildStructure with format="array"
        result_list = []
        for struct_data in structures_list:
            # Convert tensors to numpy for dict format
            result_dict = {}
            if "atom_types" in struct_data:
                result_dict["atom_types"] = struct_data["atom_types"].cpu().numpy().tolist()
            if "frac_coords" in struct_data:
                result_dict["frac_coords"] = struct_data["frac_coords"].cpu().numpy().tolist()
            if "lattices" in struct_data:
                result_dict["lattice"] = struct_data["lattices"].cpu().numpy().tolist()
            result_list.append(result_dict)

        return {"result": result_list}

    def merge_lora(self):
        if self.lora_configs is not None:
            self.denoiser = merge_lora_weights(self.denoiser)
            self.lora_configs = None
            print("LoRA weights merged into base model")
        else:
            print("No LoRA weights to merge")

    def save_checkpoint(self, save_path, epoch=None, optimizer_state=None, scheduler_state=None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'normalize_latent': self.normalize_latent,
            'latent_std': self.latent_std,
            'diffusion_configs': self.diffusion_configs,
            'augmentation': self.augmentation,
            'lora_configs': self.lora_configs,
            'use_cfg': self.use_cfg,
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        paddle.save(checkpoint, save_path)
        
    @staticmethod
    def load_checkpoint(load_path, denoiser, condition_module=None, map_location=None):
        if map_location is not None and map_location == 'cpu':
            checkpoint = paddle.load(load_path, map_location=paddle.CPUPlace())
        else:
            checkpoint = paddle.load(load_path)
        
        normalize_latent = checkpoint.get('normalize_latent', True)
        diffusion_configs = checkpoint.get('diffusion_configs', None)
        augmentation = checkpoint.get('augmentation', None)
        lora_configs = checkpoint.get('lora_configs', None)
        
        model = LDMModule(
            normalize_latent=normalize_latent,
            denoiser=denoiser,
            augmentation=augmentation,
            diffusion_configs=diffusion_configs,
            condition_module=condition_module,
            lora_configs=lora_configs,
        )
        
        model.set_state_dict(checkpoint['model_state_dict'])
        if 'latent_std' in checkpoint:
            model.latent_std = checkpoint['latent_std']
        
        return model, checkpoint

    def get_config(self):
        return {
            'normalize_latent': self.normalize_latent,
            'diffusion_configs': self.diffusion_configs,
            'augmentation': self.augmentation,
            'lora_configs': self.lora_configs,
            'use_cfg': self.use_cfg,
        }

    def predict(self, data, sampling_steps=50, sampler="ddim"):
        """Predict method for compatibility with property prediction framework.

        This method provides a unified interface for structure generation,
        making Chemeleon2 compatible with the predict.py framework.

        Args:
            data: Input data dict with optional 'num_samples' key
            sampling_steps: Number of diffusion sampling steps (default: 50)
            sampler: Sampling method - 'ddim' or 'ddpm' (default: 'ddim')

        Returns:
            dict: Results containing 'result' key with generated CrystalBatch
        """
        from ppmat.models.chemeleon2.common.schema import CrystalBatch

        num_samples = data.get('num_samples', 1) if isinstance(data, dict) else 1
        batch_size = data.get('batch_size', num_samples) if isinstance(data, dict) else num_samples

        if 'num_atoms' in data:
            num_atoms_list = [data['num_atoms']] * num_samples
        else:
            num_atom_distribution = {
                1: 0.0021742334905660377, 2: 0.021079009433962265,
                3: 0.019826061320754717, 4: 0.15271226415094338,
                5: 0.047132959905660375, 6: 0.08464770047169812,
                7: 0.021079009433962265, 8: 0.07808814858490566,
                9: 0.03434551886792453, 10: 0.0972877358490566,
                11: 0.013303360849056603, 12: 0.09669811320754718,
                13: 0.02155807783018868, 14: 0.06522700471698113,
                15: 0.014372051886792452, 16: 0.06703272405660378,
                17: 0.00972877358490566, 18: 0.053176591981132074,
                19: 0.010576356132075472, 20: 0.08995430424528301,
            }
            probs = np.array(list(num_atom_distribution.values()))
            probs = probs / probs.sum()
            num_atoms_list = np.random.choice(
                list(num_atom_distribution.keys()),
                p=probs,
                size=num_samples,
            ).tolist()

        all_results = []
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            current_num_atoms = num_atoms_list[i:i+current_batch_size]

            batch = CrystalBatch()
            total_atoms = sum(current_num_atoms)
            batch.atom_types = paddle.randint(1, 95, [total_atoms])
            batch.frac_coords = paddle.rand([total_atoms, 3])
            batch.lengths = paddle.rand([current_batch_size, 3]) * 10 + 5
            batch.angles = paddle.rand([current_batch_size, 3]) * 60 + 60
            batch.num_atoms = paddle.to_tensor(current_num_atoms, dtype='int64')
            batch.batch = paddle.repeat_interleave(
                paddle.arange(current_batch_size),
                paddle.to_tensor(current_num_atoms),
            )
            batch.token_idx = paddle.concat([paddle.arange(n) for n in current_num_atoms])
            batch.num_nodes = total_atoms
            batch.num_graphs = current_batch_size

            with paddle.no_grad():
                result = self.sample(batch, sampler=sampler, sampling_steps=sampling_steps, progress=False)
            all_results.append(result)

        return {'result': all_results}
