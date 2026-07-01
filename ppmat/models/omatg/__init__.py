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

"""OMatG: Open Materials Generation for crystal structure prediction.
Based on Stochastic Interpolants (ICML 2025, NeurIPS 2025).
"""

import os
import os.path as osp

from ppmat.models.omatg.model import OMATGCSPNetFull
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils import save_load

# OMATG_WEIGHTS: OMatG has 57+ weight files across 6 datasets and 11+ variants.
# These are NOT registered in ppmat/models/__init__.py MODEL_REGISTRY because
# MODEL_REGISTRY binds one model_name -> one zip URL, while OMatG needs
# dataset x variant cross-product per .pdparams file (not zip bundles with yaml).
# A single zip containing all variants would be >10GB, and MODEL_REGISTRY
# requires each entry to be a self-contained zip with model_name.yaml.
# We keep per-weight URLs here for fine-grained download via build_omatg_model().
OMATG_WEIGHTS = {
    "perov_5_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/VPSBD-SDE.pdparams",
    },
    "mpts_52_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/Trig-SDE-Gamma.pdparams",
    },
    "mp_20_dng": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/VPSBD-SDE.pdparams",
    },
    "mp_20_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/VPSBD-SDE.pdparams",
    },
    "alex_mp_20_csp": {
        "encdec_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-ODE-Gamma.pdparams",
        "encdec_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-SDE-Gamma.pdparams",
        "linear_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-ODE-Gamma.pdparams",
        "linear_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-ODE.pdparams",
        "linear_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Linear-SDE-Gamma.pdparams",
        "trig_ode_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-ODE-Gamma.pdparams",
        "trig_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-ODE.pdparams",
        "trig_sde_gamma": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/Trig-SDE-Gamma.pdparams",
        "vesbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VESBD-ODE.pdparams",
        "vpsbd_ode": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VPSBD-ODE.pdparams",
        "vpsbd_sde": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/VPSBD-SDE.pdparams",
    },
}

__all__ = [
    "OMATG_WEIGHTS",
    "build_omatg_model",
    "get_omatg_model_url",
]


def build_omatg_model(dataset: str, variant: str, weights_name: str = None):
    """Build OMatG model with automatic weight downloading.

    Uses the project-wide factory pattern (build_model from ppmat.models) to
    construct the model from configuration.

    Args:
        dataset: Dataset name, e.g., "mp_20_csp", "perov_5_csp", "mpts_52_csp",
                      "mp_20_dng", "alex_mp_20_csp"
        variant: Model variant, e.g., "linear_ode", "trig_ode_gamma", "encdec_sde_gamma"
        weights_name: Specific weight file name (optional). If None, uses the default
                     weight file corresponding to the variant.

    Returns:
        Loaded model and configuration dictionary

    Example:
        >>> model, config = build_omatg_model("mp_20_csp", "linear_ode")
        >>> model, config = build_omatg_model("perov_5_csp", "trig_ode_gamma", "custom.pdparams")
    """
    # Validate dataset and variant
    if dataset not in OMATG_WEIGHTS:
        available_datasets = list(OMATG_WEIGHTS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset}. Available datasets: {available_datasets}"
        )

    if variant not in OMATG_WEIGHTS[dataset]:
        available_variants = list(OMATG_WEIGHTS[dataset].keys())
        raise ValueError(
            f"Unknown variant: {variant} for dataset {dataset}. "
            f"Available variants: {available_variants}"
        )

    # Get weight URL
    weight_url = OMATG_WEIGHTS[dataset][variant]

    logger.info(f"Building OMatG model: {dataset} / {variant}")
    logger.info(f"Weight URL: {weight_url}")

    # Download weight to dataset-specific cache dir to avoid filename collision.
    weight_path = osp.join(download.WEIGHTS_HOME, f"omatg_{dataset}",
                           weight_url.split("/")[-1])
    if not osp.exists(weight_path):
        os.makedirs(osp.dirname(weight_path), exist_ok=True)
        weight_path = download._download(weight_url, osp.dirname(weight_path))
    else:
        logger.message(f"Found {weight_path} exists, skip downloading.")
    logger.info(f"Weight saved to: {weight_path}")

    # If custom weights_name is specified, use it; otherwise use the variant name
    if weights_name is None:
        weight_filename = weight_url.split("/")[-1]
        weights_name = weight_filename.replace(".pdparams", "")
        logger.info(f"Using default weights: {weights_name}")

    # DNG (Discrete Flow Matching with Mask) variants need pred_type=True
    is_dng = "dng" in dataset.lower()

    # Build OMATGCSPNetFull directly with pretrained-compatible default params
    try:
        model = OMATGCSPNetFull(
            hidden_dim=512,
            num_layers=6,
            max_atoms=100,
            act_fn="silu",
            dis_emb="sin",
            num_freqs=128,
            edge_style="fc",
            cutoff=7.0,
            max_neighbors=20,
            ln=True,
            ip=True,
            smooth=False,
            pred_type=is_dng,
            pred_scalar=False,
            time_embed_dim=256,
        )
        if is_dng:
            model.enable_masked_species()

        save_load.load_pretrain(model, weight_path, weights_name)

        logger.info(f"Successfully built and loaded OMatG model: {dataset}/{variant}")

        return model, {
            "dataset": dataset,
            "variant": variant,
            "weights_name": weights_name,
        }

    except Exception as e:
        logger.error(f"Failed to build OMatG model: {e}")
        raise


def get_omatg_model_url(dataset: str, variant: str) -> str:
    """Get the weight URL for an OMatG model variant.

    This is a convenience function for quickly getting the weight URL
    without building the full model.

    Args:
        dataset: Dataset name
        variant: Model variant

    Returns:
        Weight URL string

    Example:
        >>> url = get_omatg_model_url("mp_20_csp", "linear_ode")
        >>> weight_path = download.get_weights_path_from_url(url)
    """
    if dataset not in OMATG_WEIGHTS:
        available_datasets = list(OMATG_WEIGHTS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset}. Available datasets: {available_datasets}"
        )

    if variant not in OMATG_WEIGHTS[dataset]:
        available_variants = list(OMATG_WEIGHTS[dataset].keys())
        raise ValueError(
            f"Unknown variant: {variant} for dataset {dataset}. "
            f"Available variants: {available_variants}"
        )

    return OMATG_WEIGHTS[dataset][variant]
