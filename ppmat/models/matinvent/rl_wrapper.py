#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Materials Authors. All Rights Reserved.

import os
import sys

import paddle
import paddle.nn as nn
from omegaconf import OmegaConf


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
                sa = dict(sa); sa.pop("pbc"); data = dict(data); data["structure_array"] = sa
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
        config_path = os.environ.get("CONFIG_PATH")
        if not config_path:
            for c in cls.CONFIG_CANDIDATES:
                if os.path.exists(c):
                    config_path = c
                    break
        if not config_path:
            raise ValueError("Set CONFIG_PATH")
        return config_path

    def forward(self, batch_data):
        config_path = self._find_config()
        config = OmegaConf.load(config_path)
        output_dir = config.get("Trainer", {}).get("output_dir", config.get("Global", {}).get("output_dir", "./output/matinvent"))
        orig_argv = sys.argv
        sys.argv = ["rl_wrapper.py", "--config", config_path, "--model", self.MODEL_TYPE, "--output_dir", output_dir]
        try:
            from ppmat.models.matinvent.rl_train import main as rl_main
            rl_main()
        finally:
            sys.argv = orig_argv
        return {"loss_dict": {"loss": self._dummy_param * 0.0}}
