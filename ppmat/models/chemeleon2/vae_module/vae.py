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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.chemeleon2.common import apply_augmentation, apply_noise
from ppmat.models.chemeleon2.common import DiagonalGaussianDistribution


class VAEModule(nn.Layer):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        loss_weights,
        augmentation=None,
        noise=None,
        atom_type_predict=True,
        structure_matcher=None,
        optimizer=None,
        scheduler=None,
    ):
        super().__init__()

        # Build nested models if encoder/decoder are config dicts
        from ppmat.models import build_model
        if isinstance(encoder, dict):
            self.encoder = build_model(encoder)
        else:
            self.encoder = encoder

        if isinstance(decoder, dict):
            self.decoder = build_model(decoder)
        else:
            self.decoder = decoder
        self.latent_dim = latent_dim
        self.loss_weights = loss_weights
        self.augmentation = augmentation
        self.noise = noise
        self.atom_type_predict = atom_type_predict
        self.structure_matcher = structure_matcher
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        self.quant_conv = nn.Linear(
            self.encoder.hidden_dim, 2 * latent_dim, bias_attr=False
        )
        self.post_quant_conv = nn.Linear(
            latent_dim, self.decoder.hidden_dim, bias_attr=False
        )

        if self.loss_weights.get("fa", 0) > 0:
            self.proj = nn.Linear(self.latent_dim, 256)

    def encode(self, batch):
        encoded = self.encoder(batch)
        encoded["moments"] = self.quant_conv(encoded["x"])
        encoded["posterior"] = DiagonalGaussianDistribution(encoded["moments"])
        return encoded

    def decode(self, encoded):
        encoded["x"] = self.post_quant_conv(encoded["x"])
        decoder_out = self.decoder(encoded)
        return decoder_out
    
    def reconstruct(self, decoder_out, batch):
        from ppmat.models.chemeleon2.common.schema import CrystalBatch
        from pymatgen.core import Lattice
        import numpy as np

        batch_rec = CrystalBatch()

        if decoder_out["atom_types"].ndim == 2:
            batch_rec.atom_types = paddle.argmax(decoder_out["atom_types"], axis=1)
        else:
            batch_rec.atom_types = decoder_out["atom_types"]

        batch_rec.frac_coords = decoder_out["frac_coords"]

        # Decoder outputs lengths_scaled, need to scale back by num_atoms^(1/3)
        lengths_scaled = decoder_out["lengths"]
        if isinstance(batch.num_atoms, paddle.Tensor):
            num_atoms = batch.num_atoms.unsqueeze(-1) if batch.num_atoms.ndim == 1 else batch.num_atoms
        elif isinstance(batch.num_atoms, list):
            num_atoms = paddle.to_tensor(batch.num_atoms, dtype='float32').unsqueeze(-1)
        else:
            num_atoms = paddle.to_tensor([batch.num_atoms], dtype='float32').unsqueeze(-1)

        lengths = lengths_scaled * num_atoms ** (1 / 3)
        batch_rec.lengths = lengths
        batch_rec.lengths_scaled = lengths_scaled

        angles_radians = decoder_out["angles"]
        angles_degrees = paddle.rad2deg(angles_radians)
        batch_rec.angles = angles_degrees
        batch_rec.angles_radians = angles_radians

        lengths_np = lengths.cpu().numpy()
        angles_np = angles_degrees.cpu().numpy()

        lattices_list = []
        for i in range(lengths_np.shape[0]):
            lattice = Lattice.from_parameters(
                lengths_np[i, 0], lengths_np[i, 1], lengths_np[i, 2],
                angles_np[i, 0], angles_np[i, 1], angles_np[i, 2]
            )
            lattices_list.append(lattice.matrix)

        lattices_array = np.stack(lattices_list, axis=0)
        batch_rec.lattices = paddle.to_tensor(lattices_array, dtype='float32')

        batch_rec.num_atoms = batch.num_atoms
        batch_rec.batch = batch.batch
        batch_rec.token_idx = batch.token_idx
        batch_rec.num_nodes = batch.num_nodes
        batch_rec.num_graphs = batch.num_graphs
        return batch_rec

    def _dict_to_crystal_batch(self, batch):
        """Convert dictionary format to CrystalBatch format.

        Args:
            batch: Dict with 'structure_array' key containing structure data

        Returns:
            CrystalBatch: Converted batch object
        """
        from ppmat.models.chemeleon2.common.schema import CrystalBatch
        from ppmat.utils.crystal import lattice_params_to_matrix_paddle

        structure_array = batch["structure_array"]
        num_atoms = structure_array["num_atoms"]
        batch_size = num_atoms.shape[0]
        total_atoms = num_atoms.sum().item()

        # Create CrystalBatch from structure_array
        crystal_batch = CrystalBatch()
        crystal_batch.atom_types = structure_array["atom_types"]
        crystal_batch.frac_coords = structure_array["frac_coords"]
        crystal_batch.num_atoms = num_atoms
        crystal_batch.batch = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=num_atoms
        )

        # Handle lattice - convert lengths + angles to lattice matrix
        if "lattice" in structure_array:
            crystal_batch.lattices = structure_array["lattice"]
        else:
            crystal_batch.lattices = lattice_params_to_matrix_paddle(
                structure_array["lengths"], structure_array["angles"]
            )

        # Store original lengths and angles (scaled/radians as needed)
        crystal_batch.lengths = structure_array["lengths"]
        crystal_batch.angles = structure_array["angles"]
        crystal_batch.angles_radians = structure_array["angles"] * 3.141592653589793 / 180.0

        # Calculate lengths_scaled
        num_atoms_tensor = structure_array["num_atoms"]
        if num_atoms_tensor.ndim == 1:
            num_atoms_tensor = num_atoms_tensor.unsqueeze(-1)
        crystal_batch.lengths_scaled = structure_array["lengths"] / (
            num_atoms_tensor ** (1 / 3)
        )

        # Calculate cart_coords: cart_coords = frac_coords @ lattice.T
        # lattice shape: [batch_size, 3, 3], frac_coords shape: [total_atoms, 3]
        cart_coords = paddle.matmul(crystal_batch.frac_coords, crystal_batch.lattices[crystal_batch.batch])
        crystal_batch.cart_coords = cart_coords

        # Additional fields
        crystal_batch.num_nodes = total_atoms
        crystal_batch.num_graphs = batch_size
        crystal_batch.token_idx = paddle.concat([
            paddle.arange(n) for n in num_atoms
        ])

        return crystal_batch

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
        # Add 'loss' key for framework compatibility
        loss_dict["loss"] = loss_dict.get("total_loss", paddle.to_tensor([0.0]))

        return {"loss_dict": loss_dict}

    def calculate_loss(self, batch, training=True):
        if training and self.augmentation is not None:
            translate = self.augmentation.get('translate', False)
            rotate = self.augmentation.get('rotate', False)
            batch = apply_augmentation(batch, translate=translate, rotate=rotate)

        if training and self.noise is not None:
            ratio = self.noise.get('ratio', 0.0)
            corruption_scale = self.noise.get('corruption_scale', 0.1)
            if ratio > 0:
                batch = apply_noise(batch, ratio=ratio, corruption_scale=corruption_scale)

        # Directly call encode and decode to avoid recursion with new forward method
        encoded = self.encode(batch)
        z = encoded["posterior"].sample()
        encoded["x"] = z
        encoded["z"] = z
        decoder_out = self.decode(encoded)

        loss_atom_types = 0
        if self.atom_type_predict:
            loss_atom_types = F.cross_entropy(
                decoder_out["atom_types"], batch.atom_types
            )
        loss_lengths = F.mse_loss(decoder_out["lengths"], batch.lengths_scaled)
        loss_angles = F.mse_loss(decoder_out["angles"], batch.angles_radians)
        loss_frac_coords = F.mse_loss(
            decoder_out["frac_coords"], batch.frac_coords
        )

        loss_kl = encoded["posterior"].kl().mean()

        fa_loss = 0
        if self.loss_weights.get("fa", 0) > 0:
            z = self.proj(encoded["z"])
            mace_features = batch.mace_features
            z_norm = F.normalize(z, axis=-1)
            mace_features_norm = F.normalize(mace_features, axis=-1)
            z_cos_sim = paddle.einsum("ij,kj->ik", z_norm, z_norm)
            mace_cos_sim = paddle.einsum(
                "ij,kj->ik", mace_features_norm, mace_features_norm
            )
            diff = paddle.abs(z_cos_sim - mace_cos_sim)
            fa_loss_1 = F.relu(diff - 0.25).mean()
            fa_loss_2 = F.relu(1 - 0.5 - F.cosine_similarity(mace_features, z)).mean()
            fa_loss = fa_loss_1 + fa_loss_2

        loss = (
            self.loss_weights.get("atom_types", 1.0) * loss_atom_types
            + self.loss_weights.get("lengths", 1.0) * loss_lengths
            + self.loss_weights.get("angles", 1.0) * loss_angles
            + self.loss_weights.get("frac_coords", 1.0) * loss_frac_coords
            + self.loss_weights.get("kl", 1.0) * loss_kl
            + self.loss_weights.get("fa", 0.0) * fa_loss
        )

        return {
            "total_loss": loss.mean(),
            "loss_atom_types": loss_atom_types,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
            "loss_frac_coords": loss_frac_coords,
            "loss_kl": loss_kl,
            "fa_loss": fa_loss,
        }

    def save_checkpoint(self, save_path, epoch=None, optimizer_state=None, scheduler_state=None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'loss_weights': self.loss_weights,
            'augmentation': self.augmentation,
            'noise': self.noise,
            'atom_type_predict': self.atom_type_predict,
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        paddle.save(checkpoint, save_path)
        
    @staticmethod
    def load_checkpoint(load_path, encoder, decoder, map_location=None):
        if map_location is not None and map_location == 'cpu':
            checkpoint = paddle.load(load_path, map_location=paddle.CPUPlace())
        else:
            checkpoint = paddle.load(load_path)
        
        latent_dim = checkpoint['latent_dim']
        loss_weights = checkpoint['loss_weights']
        augmentation = checkpoint.get('augmentation', None)
        noise = checkpoint.get('noise', None)
        atom_type_predict = checkpoint.get('atom_type_predict', True)
        
        model = VAEModule(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            loss_weights=loss_weights,
            augmentation=augmentation,
            noise=noise,
            atom_type_predict=atom_type_predict,
        )
        
        model.set_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint

    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'loss_weights': self.loss_weights,
            'augmentation': self.augmentation,
            'noise': self.noise,
            'atom_type_predict': self.atom_type_predict,
        }
