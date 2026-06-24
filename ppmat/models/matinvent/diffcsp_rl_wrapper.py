#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Materials Authors. All Rights Reserved.

"""
DiffCSP RL wrapper model.

与 RLWrapperModel 逻辑完全相同，仅固定:
  - model_type = "diffcsp"
  - CONFIG_PATH fallback 搜索 matinvent_diffcsp.yaml

原始代码参考: ppmat/models/matinvent/rl_wrapper.py::RLWrapperModel
"""

import os
import sys

from omegaconf import OmegaConf

from ppmat.models.matinvent.rl_wrapper import RLWrapperModel

class DiffCSPRLWrapper(RLWrapperModel):
    CONFIG_CANDIDATES = [
        "structure_generation/configs/matinvent/matinvent_diffcsp.yaml",
        "configs/matinvent/matinvent_diffcsp.yaml",
    ]
    MODEL_TYPE = "diffcsp"
    _SAMPLE_CONFIG_CANDIDATES = [
        "structure_generation/configs/diffcsp/diffcsp_mp20.yaml",
        "configs/diffcsp/diffcsp_mp20.yaml",
    ]

    def _ensure_sample_model(self):
        if self._sample_model is not None:
            return self._sample_model

        cfg_path = os.environ.get("MATINVENT_SAMPLE_MODEL_CONFIG")
        if not cfg_path:
            for c in self._SAMPLE_CONFIG_CANDIDATES:
                if os.path.exists(c):
                    cfg_path = c
                    break
        if not cfg_path or not os.path.exists(cfg_path):
            raise FileNotFoundError("DiffCSP sample model config not found. Set MATINVENT_SAMPLE_MODEL_CONFIG or place diffcsp_mp20.yaml.")

        config = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        model_cfg = config.get("Model")
        if model_cfg is None:
            raise ValueError(f"Model section not found in sample config: {cfg_path}")

        from ppmat.models import build_model
        self._sample_model = build_model(model_cfg)
        return self._sample_model

    def sample(self, data, **sample_params):
        model = self._ensure_sample_model()
        if "num_inference_steps" not in sample_params:
            sample_params["num_inference_steps"] = int(os.environ.get("MATINVENT_NUM_INFERENCE_STEPS", "1000"))
        return model.sample(data, **sample_params)
