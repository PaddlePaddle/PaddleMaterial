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

import argparse
import os
import sys

import paddle
import paddle.nn as nn
from omegaconf import OmegaConf

from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.models.matinvent.core import MatInvent
from ppmat.models.matinvent.models import DiffCSPSuite
from ppmat.models.matinvent.models import MatterGenSuite
from ppmat.utils import logger as ppmat_logger


class RLWrapperModel(nn.Layer):
    SAMPLE_CONFIG = "structure_generation/configs/mattergen/mattergen_mp20.yaml"
    CONFIG_CANDIDATES = [
        "structure_generation/configs/matinvent/matinvent_mattergen.yaml",
        "configs/matinvent/matinvent_mattergen.yaml",
        "structure_generation/configs/mattergen/mattergen_mp20.yaml",
    ]
    MODEL_TYPE = "mattergen"

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self._sample_model = None
        self._dummy_param = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0.0))

    def _ensure_sample_model(self):
        if self._sample_model is not None:
            return self._sample_model
        cfg_path = os.environ.get("MATINVENT_SAMPLE_MODEL_CONFIG", self.SAMPLE_CONFIG)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        config = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        model_cfg = config.get("Model")
        if model_cfg is None:
            raise ValueError(f"Model section not found: {cfg_path}")
        from ppmat.models import build_model
        self._sample_model = build_model(model_cfg)
        from ppmat.models.mattergen.mattergen import GemNetT, GemNetTCtrl, MatterGen
        if isinstance(self._sample_model, MatterGen):
            from ppmat.models.matinvent.mattergen_compat import MatinventGemNetT, MatinventGemNetTCtrl
            gemnet = self._sample_model.model.gemnet
            if isinstance(gemnet, GemNetTCtrl):
                gemnet.__class__ = MatinventGemNetTCtrl
            elif isinstance(gemnet, GemNetT):
                gemnet.__class__ = MatinventGemNetT
        return self._sample_model

    def sample(self, data, **sample_params):
        model = self._ensure_sample_model()
        sa = data.get("structure_array") if isinstance(data, dict) else None
        if isinstance(sa, dict):
            if "pbc" in sa:
                sa = dict(sa); sa.pop("pbc")
                data = dict(data); data["structure_array"] = sa
            if "num_atoms" in sa and hasattr(sa["num_atoms"], "shape") and int(sa["num_atoms"].shape[0]) == 1:
                sa = dict(sa); sa["num_atoms"] = paddle.concat([sa["num_atoms"], sa["num_atoms"]], axis=0)
                data = dict(data); data["structure_array"] = sa
        if "num_inference_steps" not in sample_params:
            sample_params["num_inference_steps"] = int(os.environ.get("MATINVENT_NUM_INFERENCE_STEPS", "50"))
        return model.sample(data, **sample_params)

    def set_state_dict(self, state_dict, **kwargs):
        return self._ensure_sample_model().set_state_dict(state_dict, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._sample_model.state_dict(*args, **kwargs) if self._sample_model else super().state_dict(*args, **kwargs)

    def eval(self):
        super().eval()
        if self._sample_model:
            self._sample_model.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._sample_model:
            self._sample_model.train() if mode else self._sample_model.eval()
        return self

    @classmethod
    def _find_config(cls):
        cfg = os.environ.get("CONFIG_PATH")
        if not cfg:
            for c in cls.CONFIG_CANDIDATES:
                if os.path.exists(c):
                    cfg = c
                    break
        if not cfg:
            raise ValueError("Set CONFIG_PATH")
        return cfg

    def forward(self, batch_data):
        config_path = self._find_config()
        _run_rl_training(config_path, self.MODEL_TYPE)
        return {"loss_dict": {"loss": self._dummy_param * 0.0}}


class DiffCSPRLWrapper(RLWrapperModel):
    SAMPLE_CONFIG = "structure_generation/configs/diffcsp/diffcsp_mp20.yaml"
    CONFIG_CANDIDATES = [
        "structure_generation/configs/matinvent/matinvent_diffcsp.yaml",
        "configs/matinvent/matinvent_diffcsp.yaml",
    ]
    MODEL_TYPE = "diffcsp"

    def _ensure_sample_model(self):
        if self._sample_model is not None:
            return self._sample_model
        cfg_path = os.environ.get("MATINVENT_SAMPLE_MODEL_CONFIG")
        if not cfg_path:
            for c in self.CONFIG_CANDIDATES:
                if os.path.exists(c):
                    cfg_path = c
                    break
        if not cfg_path or not os.path.exists(cfg_path):
            raise FileNotFoundError("DiffCSP config not found. Set MATINVENT_SAMPLE_MODEL_CONFIG.")
        config = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        model_cfg = config.get("Model")
        if model_cfg is None:
            raise ValueError(f"Model section not found: {cfg_path}")
        from ppmat.models import build_model
        self._sample_model = build_model(model_cfg)
        return self._sample_model

    def sample(self, data, **sample_params):
        model = self._ensure_sample_model()
        if "num_inference_steps" not in sample_params:
            sample_params["num_inference_steps"] = int(os.environ.get("MATINVENT_NUM_INFERENCE_STEPS", "1000"))
        return model.sample(data, **sample_params)


def _run_rl_training(config_path: str, model_type: str):
    config = OmegaConf.load(config_path)
    output_dir = config.get("Trainer", {}).get("output_dir", config.get("Global", {}).get("output_dir", "./output/matinvent"))
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    ppmat_logger.init_logger(name="matinvent_rl", log_file=log_file, log_level=20)
    logger = ppmat_logger._logger
    logger.info(f"Config: {config_path}, output: {output_dir}")

    suite_cls = MatterGenSuite if model_type == "mattergen" else DiffCSPSuite
    model_suite = suite_cls(
        model_name=model_type, sample_cfg=config.RL.sample_cfg,
        finetune_cfg=config.RL.finetune_cfg, model_path=config.get("model_path"),
        device=config.get("Global", {}).get("device"),
    )
    reward = Reward(root_dir=os.path.join(output_dir, "rewards"),
                    prop_cfg=config.RL.reward_cfg.prop_cfg,
                    reward_threshold=config.RL.reward_cfg.reward_threshold,
                    reduce=config.RL.reward_cfg.reduce)

    mat_invent = MatInvent(rl_epoch=config.RL.rl_epoch, model_suite=model_suite,
                           reward=reward, sample_cfg={}, finetune_cfg={},
                           topk_ratio=config.RL.topk_ratio, save_dir=output_dir,
                           save_freq=config.RL.save_freq, device=str(paddle.get_device()),
                           logger=logger,
                           replay=config.RL.get("replay_cfg") is not None,
                           replay_args=config.RL.get("replay_cfg", {}),
                           div_filter=config.RL.div_filter_cfg.enabled,
                           df_args=config.RL.div_filter_cfg)
    mat_invent.run_rl()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default="mattergen", choices=["mattergen", "diffcsp"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    _run_rl_training(args.config, args.model)
