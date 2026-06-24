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

import os
from pathlib import Path
from typing import List

import numpy as np
import paddle
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rl.data import create_rl_dataloader
from ppmat.models.matinvent.rl.data import DiffCSPSampler
from ppmat.models.matinvent.rl.data import MatterGenSampler


_DIFFCSP_DEFAULT_CFG = dict(
    decoder_cfg=dict(hidden_dim=512, latent_dim=256, num_layers=6, act_fn="silu",
        dis_emb="sin", num_freqs=128, edge_style="fc", ln=True, ip=True,
        smooth=False, pred_type=False, prop_dim=512, pred_scalar=False, num_classes=100),
    lattice_noise_scheduler_cfg={"__class_name__": "DDPMScheduler",
        "__init_params__": {"beta_schedule": "squaredcos_cap_v2", "num_train_timesteps": 1000, "clip_sample": False}},
    coord_noise_scheduler_cfg={"__class_name__": "ScoreSdeVeSchedulerWrapped",
        "__init_params__": {"num_train_timesteps": 1000, "sigma_min": 0.005, "sigma_max": 0.5, "snr": 1e-5}},
    num_train_timesteps=1000, time_dim=256, lattice_loss_weight=1.0, coord_loss_weight=1.0)

_MATTERGEN_DEFAULT_CFG = dict(
    decoder_cfg={"gemnet_cfg": {"num_targets": 1, "latent_dim": 512,
        "atom_embedding_cfg": {"emb_size": 512, "with_mask_type": True},
        "max_neighbors": 50, "max_cell_images_per_dim": 5, "cutoff": 7.0,
        "num_blocks": 4, "otf_graph": True}},
    lattice_noise_scheduler_cfg={"__class_name__": "LatticeVPSDEScheduler",
        "limit_density": 0.05771451654022283, "__init_params__": {}},
    coord_noise_scheduler_cfg={"__class_name__": "NumAtomsVarianceAdjustedWrappedVESDE",
        "__init_params__": {}},
    atom_noise_scheduler_cfg={"__class_name__": "D3PMScheduler", "__init_params__": {}},
    num_train_timesteps=1000, time_dim=256, lattice_loss_weight=1,
    coord_loss_weight=0.1, atom_loss_weight=1)


class ModelSuite:
    def __init__(self, model_name: str, sample_cfg: DictConfig, finetune_cfg: DictConfig,
                 model_path: str | None = None, config_overrides: list[str] = [],
                 device: str | None = None, **kwargs):
        self.model_name = model_name
        self.sample_cfg = sample_cfg
        self.finetune_cfg = finetune_cfg
        self.model_path = model_path
        self.config_overrides = config_overrides
        if device is None:
            device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(device)
        self.device = device
        self.cfg = OmegaConf.create(kwargs)

    def load_model(self):
        raise NotImplementedError
    def get_sampler(self):
        raise NotImplementedError
    def get_dataloader(self):
        raise NotImplementedError
    def save_model(self):
        raise NotImplementedError


class DiffCSPSuite(ModelSuite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def load_model(self):
        from ppmat.models.diffcsp.diffcsp import DiffCSP
        if self.model_path is None:
            raise ValueError("model_path must be specified for DiffCSPSuite")
        ckpt = Path(self.model_path).expanduser()
        if not ckpt.is_absolute():
            ckpt = Path.cwd() / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model = DiffCSP(**_DIFFCSP_DEFAULT_CFG)
        model.set_state_dict({"decoder." + k: v for k, v in paddle.load(str(ckpt)).items()})
        model.eval()
        return model

    def get_sampler(self):
        return DiffCSPSampler(batch_size=self.sample_cfg.get("batch_size", 16),
            num_batches=self.sample_cfg.get("num_batches", 4),
            num_inference_steps=self.sample_cfg.get("num_inference_steps", 1000))

    def get_dataloader(self, samples: List[Structure], rewards: np.ndarray, batch_size: int = 8):
        return create_rl_dataloader(structures=samples, rewards=rewards, batch_size=batch_size)

    def save_model(self, model, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(ckpt_dir, "model.pdparams"))


class MatterGenRLAdapter:
    def __init__(self, base_model):
        self._base_model = base_model

    def __getattr__(self, name):
        return getattr(self._base_model, name)

    def noise_level_encoding(self, t):
        if hasattr(self._base_model, "noise_level_encoding"):
            return self._base_model.noise_level_encoding(t)
        from ppmat.models.common.sinusoidal_embedding import SinusoidalEmbeddings
        return SinusoidalEmbeddings(dim=self.time_dim)(t)

    @property
    def max_t(self): return getattr(self._base_model, "max_t", 1.0)
    @property
    def time_dim(self): return getattr(self._base_model, "time_dim", 256)
    @property
    def lattice_loss_weight(self): return getattr(self._base_model, "lattice_loss_weight", 1.0)
    @property
    def coord_loss_weight(self): return getattr(self._base_model, "coord_loss_weight", 0.1)
    @property
    def atom_loss_weight(self): return getattr(self._base_model, "atom_loss_weight", 1.0)
    @property
    def d3pm_hybrid_lambda(self): return getattr(self._base_model, "d3pm_hybrid_lambda", None)


def create_matinvent_adapter(model):
    return MatterGenRLAdapter(model)


class MatterGenSuite(ModelSuite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None

    def load_model(self):
        from ppmat.models.matinvent.mattergen_compat import MatinventMatterGen
        if self.model_path is None:
            raise ValueError("model_path must be specified for MatterGenSuite")
        ckpt = Path(self.model_path).expanduser()
        if not ckpt.is_absolute():
            ckpt = Path.cwd() / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        model = MatinventMatterGen(**_MATTERGEN_DEFAULT_CFG)
        model.set_state_dict(paddle.load(str(ckpt)))
        model.eval()
        return create_matinvent_adapter(model)

    def get_sampler(self):
        return MatterGenSampler(batch_size=self.sample_cfg.get("batch_size", 16),
            num_batches=self.sample_cfg.get("num_batches", 4),
            num_inference_steps=self.sample_cfg.get("num_inference_steps", 1000))

    def get_dataloader(self, samples: List[Structure], rewards: np.ndarray, batch_size: int = 8):
        return create_rl_dataloader(structures=samples, rewards=rewards, batch_size=batch_size)

    def save_model(self, model, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        paddle.save(model.state_dict(), os.path.join(ckpt_dir, "model.pdparams"))
