#!/usr/bin/env python
# Copyright (c) 2025 PaddlePaddle Materials Authors. All Rights Reserved.

import builtins
import os
import sys

import paddle
import paddle.nn as nn
from omegaconf import OmegaConf

import ppmat.models

builtins.ppmat = ppmat


class RLWrapperModel(nn.Layer):
    """Wrapper model that executes RL training when called."""

    def __init__(self, **kwargs):
        super().__init__()
        # Store kwargs for potential use
        self.kwargs = kwargs
        self._sample_model = None
        self._dummy_param = self.create_parameter(
            shape=[1],
            default_initializer=nn.initializer.Constant(0.0),
        )

    def _ensure_sample_model(self):
        if self._sample_model is not None:
            return self._sample_model

        cfg_path = os.environ.get(
            "MATINVENT_SAMPLE_MODEL_CONFIG",
            "structure_generation/configs/mattergen/mattergen_mp20.yaml",
        )
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Sample model config not found: {cfg_path}. "
                "Set MATINVENT_SAMPLE_MODEL_CONFIG to a valid yaml path."
            )

        config = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        model_cfg = config.get("Model")
        if model_cfg is None:
            raise ValueError(f"Model section not found in sample config: {cfg_path}")

        from ppmat.models import build_model

        self._sample_model = build_model(model_cfg)

        # MatterGen 的 radius_graph_pbc 存在 paddle.all non-contiguous bug，
        # 将 gemnet 实例的类切换为修复版子类（参见 mattergen_compat.py）
        from ppmat.models.mattergen.mattergen import GemNetT
        from ppmat.models.mattergen.mattergen import GemNetTCtrl
        from ppmat.models.mattergen.mattergen import MatterGen

        if isinstance(self._sample_model, MatterGen):
            from ppmat.models.matinvent.mattergen_compat import MatinventGemNetT
            from ppmat.models.matinvent.mattergen_compat import MatinventGemNetTCtrl

            gemnet = self._sample_model.model.gemnet
            if isinstance(gemnet, GemNetTCtrl):
                gemnet.__class__ = MatinventGemNetTCtrl
            elif isinstance(gemnet, GemNetT):
                gemnet.__class__ = MatinventGemNetT

        return self._sample_model

    def sample(self, data, **sample_params):
        model = self._ensure_sample_model()
        structure_array = (
            data.get("structure_array") if isinstance(data, dict) else None
        )
        if isinstance(structure_array, dict) and "pbc" in structure_array:
            structure_array = dict(structure_array)
            structure_array.pop("pbc", None)
            data = dict(data)
            data["structure_array"] = structure_array

        if isinstance(structure_array, dict) and "num_atoms" in structure_array:
            num_atoms = structure_array["num_atoms"]
            if hasattr(num_atoms, "shape") and int(num_atoms.shape[0]) == 1:
                dup_num_atoms = paddle.concat([num_atoms, num_atoms], axis=0)
                structure_array = dict(structure_array)
                structure_array["num_atoms"] = dup_num_atoms
                data = dict(data)
                data["structure_array"] = structure_array

        if "num_inference_steps" not in sample_params:
            sample_params["num_inference_steps"] = int(
                os.environ.get("MATINVENT_NUM_INFERENCE_STEPS", "50")
            )
        return model.sample(data, **sample_params)

    def set_state_dict(self, state_dict, use_structured_name=True):
        model = self._ensure_sample_model()
        return model.set_state_dict(
            state_dict,
            use_structured_name=use_structured_name,
        )

    def state_dict(self, *args, **kwargs):
        if self._sample_model is None:
            return super().state_dict(*args, **kwargs)
        return self._sample_model.state_dict(*args, **kwargs)

    def eval(self):
        super().eval()
        if self._sample_model is not None:
            self._sample_model.eval()
        return self

    def train(self):
        super().train()
        if self._sample_model is not None:
            self._sample_model.train()
        return self

    def forward(self, batch_data):
        """Execute RL training when forward is called."""
        # Get config path from environment variable or current working directory
        config_path = os.environ.get("CONFIG_PATH", None)

        # If config_path not set, try to find it from common locations
        if not config_path:
            # Check current directory
            if os.path.exists(
                "structure_generation/configs/matinvent" "/matinvent_mattergen.yaml"
            ):
                config_path = (
                    "structure_generation/configs/matinvent" "/matinvent_mattergen.yaml"
                )
            elif os.path.exists("configs/matinvent/matinvent_mattergen.yaml"):
                config_path = "configs/matinvent/matinvent_mattergen.yaml"
            else:
                # Try to find any yaml file in configs directory
                import glob

                yaml_files = glob.glob("**/*.yaml", recursive=True)
                if yaml_files:
                    config_path = yaml_files[0]

        if not config_path:
            raise ValueError(
                "Could not find config file. " "Set CONFIG_PATH environment variable."
            )

        # Load config to determine model type
        config = OmegaConf.load(config_path)
        model_type = "mattergen"
        if (
            "diffcsp" in config_path.lower()
            or "diffcsp" in str(config.get("Model", {})).lower()
        ):
            model_type = "diffcsp"

        # Get output directory from config (prefer trainer output_dir).
        output_dir = config.get("Trainer", {}).get(
            "output_dir",
            config.get("Global", {}).get("output_dir", "./output/matinvent"),
        )

        # Build RL training arguments
        rl_args = [
            "--config",
            config_path,
            "--model",
            model_type,
            "--output_dir",
            output_dir,
        ]

        # Temporarily modify sys.argv and execute RL training
        original_argv = sys.argv
        sys.argv = ["rl_wrapper.py"] + rl_args

        try:
            from ppmat.models.matinvent.rl_train import main as rl_main

            rl_main()
        finally:
            sys.argv = original_argv

        # Return a dummy loss dict to satisfy the trainer
        return {"loss_dict": {"loss": self._dummy_param * 0.0}}
