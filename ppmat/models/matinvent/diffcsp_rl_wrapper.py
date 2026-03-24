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

# Diffcsp yaml 的相对路径候选（按优先级）
_DIFFCSP_CONFIG_CANDIDATES = [
    "structure_generation/configs/matinvent/matinvent_diffcsp.yaml",
    "configs/matinvent/matinvent_diffcsp.yaml",
]

_DIFFCSP_SAMPLE_CONFIG_CANDIDATES = [
    "structure_generation/configs/diffcsp/diffcsp_mp20.yaml",
    "configs/diffcsp/diffcsp_mp20.yaml",
]


class DiffCSPRLWrapper(RLWrapperModel):
    """DiffCSP RL wrapper -- 固定使用 diffcsp 模型和 matinvent_diffcsp.yaml 配置。

    相比 RLWrapperModel，本类:
      1. CONFIG_PATH fallback 优先搜索 matinvent_diffcsp.yaml
      2. model_type 固定为 "diffcsp"，不通过文件名猜测
      3. _ensure_sample_model 加载 DiffCSP 模型而非 MatterGen
    """

    def _ensure_sample_model(self):
        if self._sample_model is not None:
            return self._sample_model

        cfg_path = os.environ.get("MATINVENT_SAMPLE_MODEL_CONFIG", None)
        if not cfg_path:
            for candidate in _DIFFCSP_SAMPLE_CONFIG_CANDIDATES:
                if os.path.exists(candidate):
                    cfg_path = candidate
                    break
        if not cfg_path or not os.path.exists(cfg_path):
            raise FileNotFoundError(
                "DiffCSP sample model config not found. "
                "Set MATINVENT_SAMPLE_MODEL_CONFIG or place diffcsp_mp20.yaml."
            )

        config = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        model_cfg = config.get("Model")
        if model_cfg is None:
            raise ValueError(f"Model section not found in sample config: {cfg_path}")

        from ppmat.models import build_model

        self._sample_model = build_model(model_cfg)
        return self._sample_model

    def sample(self, data, **sample_params):
        """DiffCSP 采样 -- 跳过父类中 MatterGen 专用的 num_atoms 复制逻辑。"""
        model = self._ensure_sample_model()
        if "num_inference_steps" not in sample_params:
            sample_params["num_inference_steps"] = int(
                os.environ.get("MATINVENT_NUM_INFERENCE_STEPS", "1000")
            )
        return model.sample(data, **sample_params)

    def forward(self, batch_data):
        """Execute DiffCSP RL training when forward is called.

        原始代码: RLWrapperModel.forward，model_type 改为固定 diffcsp
        """
        # 1. 优先读环境变量
        config_path = os.environ.get("CONFIG_PATH", None)

        # 2. fallback: 搜索 diffcsp 配置
        if not config_path:
            for candidate in _DIFFCSP_CONFIG_CANDIDATES:
                if os.path.exists(candidate):
                    config_path = candidate
                    break

        # 3. 再 fallback: 搜索任意 diffcsp yaml
        if not config_path:
            import glob

            for f in glob.glob("**/*.yaml", recursive=True):
                if "diffcsp" in f.lower():
                    config_path = f
                    break

        if not config_path:
            raise ValueError(
                "找不到 DiffCSP 配置文件。请设置环境变量 CONFIG_PATH 指向 "
                "matinvent_diffcsp.yaml，或在当前目录下放置该文件。"
            )

        config = OmegaConf.load(config_path)

        # model_type 固定为 diffcsp，不依赖文件名猜测
        model_type = "diffcsp"

        output_dir = config.get("Trainer", {}).get(
            "output_dir",
            config.get("Global", {}).get("output_dir", "./output/matinvent_diffcsp"),
        )

        rl_args = [
            "--config",
            config_path,
            "--model",
            model_type,
            "--output_dir",
            output_dir,
        ]

        original_argv = sys.argv
        sys.argv = ["diffcsp_rl_wrapper.py"] + rl_args

        try:
            from ppmat.models.matinvent.rl_train import main as rl_main

            rl_main()
        finally:
            sys.argv = original_argv

        return {"loss_dict": {"loss": self._dummy_param * 0.0}}
