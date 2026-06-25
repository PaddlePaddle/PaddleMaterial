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

import paddle
import paddle.nn as nn
from omegaconf import OmegaConf

import ppmat
from ppmat.models.matinvent.rl_loop import MatInvent
from ppmat.models.matinvent.suites import ModelSuite
from ppmat.models.matinvent.rewards.reward import Reward
from ppmat.utils import logger as ppmat_logger

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(ppmat.__file__)),
    "structure_generation", "configs",
)


class MatinventRL(nn.Layer):
    def __init__(self, model_type="mattergen", num_inference_steps=None,
                 sample_config=None, config_candidates=None, patch_gemnet=True):
        super().__init__()
        self.model_type = model_type
        self.num_inference_steps = num_inference_steps or (50 if model_type == "mattergen" else 1000)
        self.patch_gemnet = patch_gemnet
        self._model = None
        self._dummy = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0.0))

        self._sample_cfg = sample_config or os.path.join(
            _CONFIG_DIR, model_type, f"{model_type}_mp20.yaml")
        self._cfg_candidates = config_candidates or (
            [os.path.join(_CONFIG_DIR, "matinvent", f"matinvent_{model_type}.yaml"),
             os.path.join(_CONFIG_DIR, model_type, f"{model_type}_mp20.yaml")]
            if model_type == "mattergen"
            else [os.path.join(_CONFIG_DIR, "matinvent", f"matinvent_{model_type}.yaml")]
        )

    def _resolve_config(self):
        for src in (os.environ.get("MATINVENT_SAMPLE_MODEL_CONFIG"),
                    os.environ.get("CONFIG_PATH")):
            if src and os.path.exists(src):
                return src
        for c in self._cfg_candidates:
            if os.path.exists(c):
                return c
        if os.path.exists(self._sample_cfg):
            return self._sample_cfg
        raise FileNotFoundError(
            f"No config found. Set MATINVENT_SAMPLE_MODEL_CONFIG or CONFIG_PATH. "
            f"Candidates: {self._cfg_candidates}")

    def _ensure_model(self):
        if self._model is None:
            cfg = OmegaConf.to_container(OmegaConf.load(self._resolve_config()), resolve=True)
            model_cfg = cfg.get("Model")
            if model_cfg is None:
                raise ValueError(f"Model section not found in config")
            from ppmat.models import build_model
            model = build_model(model_cfg)
            if self.patch_gemnet:
                from ppmat.models.mattergen.mattergen import MatterGen, GemNetT, GemNetTCtrl
                if isinstance(model, MatterGen):
                    from ppmat.models.matinvent.mattergen_compat import MatinventGemNetT, MatinventGemNetTCtrl
                    g = model.model.gemnet
                    if isinstance(g, GemNetTCtrl):
                        g.__class__ = MatinventGemNetTCtrl
                    elif isinstance(g, GemNetT):
                        g.__class__ = MatinventGemNetT
            self._model = model
        return self._model

    def sample(self, data, **sample_params):
        model = self._ensure_model()
        if "num_inference_steps" not in sample_params:
            sample_params["num_inference_steps"] = int(
                os.environ.get("MATINVENT_NUM_INFERENCE_STEPS", str(self.num_inference_steps)))
        return model.sample(data, **sample_params)

    def set_state_dict(self, state_dict, **kwargs):
        return self._ensure_model().set_state_dict(state_dict, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs) if self._model else super().state_dict(*args, **kwargs)

    def eval(self):
        super().eval()
        if self._model:
            self._model.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._model:
            self._model.train(mode)
        return self

    def forward(self, batch_data):
        _run_rl_training(self._resolve_config(), self.model_type)
        return {"loss_dict": {"loss": self._dummy * 0.0}}


def _run_rl_training(config_path: str, model_type: str):
    config = OmegaConf.load(config_path)
    output_dir = config.get("Trainer", {}).get("output_dir", config.get("Global", {}).get("output_dir", "./output/matinvent"))
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    ppmat_logger.init_logger(name="matinvent_rl", log_file=log_file, log_level=20)
    logger = ppmat_logger._logger
    logger.info(f"Config: {config_path}, output: {output_dir}")

    model_suite = ModelSuite(
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
