# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional

import numpy as np
import paddle

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.liflow_dataset import LiFlowDataset
from ppmat.predictor.base import BasePredictor


class LiFlowPredictor(BasePredictor):
    """Predict atomic flow velocities from LiFlow trajectory samples."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config_overrides=None,
    ):
        super().__init__(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            work_dir="",
            device=device,
            config_overrides=config_overrides,
        )
        self.load_inference_model()

    def from_dataset_sample(
        self,
        data_path: Optional[str] = None,
        index_file: Optional[str] = None,
        sample_index: Optional[int] = None,
    ):
        path = data_path or self.predict_config.get("path") or self.predict_config.get("data_path")
        index_file = index_file or self.predict_config.get("index_file")
        if path is None or index_file is None:
            raise ValueError("path/data_path and index_file must be provided.")
        sample_index = int(self.predict_config.get("sample_index", 0) if sample_index is None else sample_index)
        dataset = LiFlowDataset(
            path=path,
            index_file=index_file,
            time_delay_steps=self.predict_config.get("time_delay_steps", 100),
            prior_scale_li=self.predict_config.get("prior_scale_li", (1.0, 10.0)),
            prior_scale_frame=self.predict_config.get("prior_scale_frame", (0.316, 3.16)),
            seed=self.predict_config.get("seed", 42),
            random_time=False,
        )
        if not 0 <= sample_index < len(dataset):
            raise IndexError(f"sample_index {sample_index} outside dataset of size {len(dataset)}")
        sample = dataset[sample_index]
        batch = DefaultCollator()([sample])
        for key, value in list(batch.items()):
            if isinstance(value, np.ndarray):
                batch[key] = paddle.to_tensor(value)
        result = self._run_model(batch)
        return {
            "name": sample["name"],
            "frame_start": int(sample["frame_start"]),
            "frame_end": int(sample["frame_end"]),
            "num_atoms": len(sample["elements"]),
            "velocity": result["velocity"],
            "target": result["target"],
        }
