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
SGEQUIDiff Sampler Wrapper.

Adapts CrystalSampler to the PaddleMaterials sampling interface

Usage (via sample.py):
    python structure_generation/sample.py \\
        --config_path structure_generation/configs/sgequidiff/sgequidiff_mp20_sample.yaml \\
        --checkpoint_path /path/to/weights \\
        --mode by_num_atoms --num_atoms 8 --save_path ./output_sgequidiff

"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import paddle
import paddle.nn as nn

from ppmat.models.sgequidiff.constants import chemical_symbols
from ppmat.models.sgequidiff.crystal_sampler import CrystalSampler
from ppmat.models.sgequidiff.crystal_sampler import CrystalSamplerConfig
from ppmat.models.sgequidiff.embedding_utils import set_global_embedding_tools
from ppmat.models.sgequidiff.weight_utils import load_pretrained_weights
from ppmat.utils import logger


class SGEQUIDiffSampler(nn.Layer):
    """Wrapper that adapts CrystalSampler to the PaddleMaterials sampling interface.

    This wrapper provides a ``sample(batch_data)`` method compatible with
    ``structure_generation/sample.py``, translating between the standard
    ``batch_data`` dict format and CrystalSampler's native API.

    Args:
        dataset_name: Dataset name, e.g. ``"mp_20"`` or ``"mpts_52"``.
        diffusion_snr: SNR for the diffusion sampling step.
        temperature: Sampling temperature.
        weight_dir: Optional local directory containing the 4 weight files.
            If ``None``, weights are auto-downloaded from BOS.
    """

    def __init__(
        self,
        dataset_name: str = "mp_20",
        diffusion_snr: float = 0.4,
        temperature: float = 1.0,
        weight_dir: Optional[str] = None,
        num_timesteps: int = 1000,
        noise_scheduler_num_monte_carlo_samples: int = 2500,
        num_wn_lattice_translations: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.diffusion_snr = diffusion_snr
        self.temperature = temperature

        # Build CrystalSampler from default config
        from ppmat.models.sgequidiff.diffusion_model import (
            EquivariantDiffusionModelConfig,
        )
        from ppmat.models.sgequidiff.lattice_sampler import LatticeSamplerConfig
        from ppmat.models.sgequidiff.wyckoff_transformer import (
            WyckoffElementTransformerConfig,
        )
        from ppmat.models.sgequidiff.non_equivariant_drift_modules import GNNConfig
        from ppmat.models.sgequidiff.constants import lattice_parameter_ranges

        lr = lattice_parameter_ranges.get(
            dataset_name, lattice_parameter_ranges["mp_20"]
        )

        gnn_cfg = GNNConfig(
            num_plane_wave_freqs=96,
            num_cartesian_distance_gaussians=96,
            edge_hidden_dim=128,
            atom_hidden_dim=256,
            use_vpa=True,
            use_graph_norm=True,
            num_msg_pass_steps=5,
            cutoff=10.0,
            use_frac_coords_in_node_emb=True,
            dataset_name=dataset_name,
        )
        diff_cfg = EquivariantDiffusionModelConfig(
            model_type="gnn",
            num_timesteps=num_timesteps,
            noise_scheduler_num_monte_carlo_samples=noise_scheduler_num_monte_carlo_samples,
            num_wn_lattice_translations=num_wn_lattice_translations,
            sigma_min=0.002,
            sigma_max=0.5,
            time_emb_dim=128,
            num_plane_wave_freqs=96,
            gnn_config=gnn_cfg,
        )
        lattice_cfg = LatticeSamplerConfig(
            input_dimension=128,
            hidden_dimension=256,
            min_lattice_length=lr["min_lattice_length"],
            max_lattice_length=lr["max_lattice_length"],
            min_lattice_angle=lr["min_lattice_angle"],
            max_lattice_angle=lr["max_lattice_angle"],
            lattice_param_dim=32,
            n_emb_layers=2,
            lattice_length_bin_embedder_fourier_scale=2.0,
            lattice_angle_bin_embedder_fourier_scale=1.0,
            lattice_length_embedder_fourier_scale=5.0,
            lattice_angle_embedder_fourier_scale=1.0,
        )
        we_cfg = WyckoffElementTransformerConfig(
            hidden_dim=256,
            dataset_name=dataset_name,
            num_heads=2,
            num_hidden_layers=4,
            dropout_rate=0.1,
        )
        sampler_cfg = CrystalSamplerConfig(
            diffusion_model_config=diff_cfg,
            lattice_model_config=lattice_cfg,
            transformer_config=we_cfg,
        )

        # Initialize global embedding tools BEFORE building CrystalSampler,
        # because CrystalSampler.__init__ -> GNN.__init__ requires embedding_tools
        set_global_embedding_tools(
            element_embedding_json_path="cgcnn_atom_init.json",
            space_group_embedding_json_path="init_tokens/space_group_features/space_group_embeddings_62dim.json",
            wyckoff_embedding_json_path="init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json",
            chemistry_embedding_type="identity",
        )

        self.crystal_sampler = CrystalSampler(sampler_cfg)

        # Load pretrained weights
        load_pretrained_weights(
            self.crystal_sampler,
            dataset_name=dataset_name,
            weight_dir=weight_dir,
            verbose=True,
        )

        logger.info(
            f"[SGEQUIDiffSampler] Initialized: dataset={dataset_name}, "
            f"snr={diffusion_snr}, temperature={temperature}"
        )

    # ------------------------------------------------------------------
    # Core sampling interface compatible with sample.py
    # ------------------------------------------------------------------

    @paddle.no_grad()
    def sample(self, batch_data: Dict, **kwargs) -> Dict:
        """Sample crystals compatible with ``structure_generation/sample.py``.

        Args:
            batch_data: Dict with key ``"structure_array"`` containing at
                least ``"num_atoms"`` (a 1-D int tensor of batch sizes).
                SGEQuiDiff is an **unconditional** generator, so the actual
                atom counts in *batch_data* are ignored -- they only
                determine the *batch size*.

        Returns:
            Dict with key ``"result"`` mapping to a list of dicts, each
            containing:
            - ``"num_atoms"``: int
            - ``"atom_types"``: list[int]  (1-indexed atomic numbers)
            - ``"frac_coords"``: list[list[float]]
            - ``"lengths"``: list[float]  (a, b, c in Angstrom)
            - ``"angles"``: list[float]   (alpha, beta, gamma in degrees)
        """
        structure_array = batch_data["structure_array"]
        num_atoms_tensor = structure_array["num_atoms"]
        batch_size = num_atoms_tensor.shape[0]

        # SGEQuiDiff is unconditional: generate from noise
        crystals = self.crystal_sampler.sample_crystal(
            batch_size=batch_size,
            diffusion_snr=self.diffusion_snr,
            temperature=self.temperature,
        )

        # Convert ASUCrystal objects to the standard result format
        # expected by BuildStructure(format="array")
        result = []
        for crystal in crystals:
            # Filter out placeholder elements (X, X0+, empty)
            valid_indices = []
            for idx in crystal.element_indices.tolist():
                if 0 <= idx < len(chemical_symbols):
                    elem = chemical_symbols[idx]
                    if elem not in ("X", "X0+", ""):
                        valid_indices.append(idx)

            if len(valid_indices) == 0:
                logger.warning("Generated crystal has no valid elements, skipping.")
                continue

            # Extract valid atoms
            frac_coords = crystal.conventional_frac_coords
            valid_mask = []
            for i, idx in enumerate(crystal.element_indices.tolist()):
                if 0 <= idx < len(chemical_symbols):
                    elem = chemical_symbols[idx]
                    if elem not in ("X", "X0+", ""):
                        valid_mask.append(i)

            valid_frac_coords = frac_coords[valid_mask].numpy().tolist()
            valid_atom_types = [
                crystal.element_indices[i].item() + 1  # 1-indexed atomic number
                for i in valid_mask
            ]

            lengths = crystal.conventional_lattice_lengths.numpy().tolist()
            angles = crystal.conventional_lattice_angles.numpy().tolist()

            result.append(
                {
                    "num_atoms": len(valid_atom_types),
                    "atom_types": valid_atom_types,
                    "frac_coords": valid_frac_coords,
                    "lengths": lengths,
                    "angles": angles,
                }
            )

        return {"result": result}
