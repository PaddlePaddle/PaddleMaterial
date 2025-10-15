# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf

from ppmat.predictor import StructureSampler
from ppmat.utils import logger


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig):
    # Save the loaded config
    OmegaConf.save(cfg, "config_saved.yaml")

    # Initialize logger
    output_dir = os.getcwd()
    logger_path = cfg.Logger.get("log_file", "out.log")
    logger.init_logger(
        log_file=logger_path, log_level=cfg["Logger"].get("log_level", "INFO")
    )
    logger.info("[PPMaterial] Logger initialized")
    logger.info(f"Working directory: {output_dir}")
    logger.info(f"Log file path    : {os.path.abspath(logger_path)}")

    # Initialize the model
    load_model = instantiate(cfg.Model)
    sampler = StructureSampler(save_path=output_dir, device=cfg.device, **load_model)

    # Initialize the sample
    sample_task = instantiate(cfg.Sample)

    if sample_task.name == "compute_metric":
        metric_result = sampler.compute_metric(save_path=cfg.Run.work_dir)
        for metric_name, metric_value in metric_result.items():
            logger.info(f"{metric_name}: {metric_value}")
    elif sample_task.name == "by_chemical_formula":
        sampler.sample_by_chemical_formula(
            chemical_formula=sample_task.chemical_formula,
        )
    elif sample_task.name == "by_num_atoms":
        sampler.sample_by_num_atoms(
            num_atoms=sample_task.num_atoms,
        )
    elif sample_task.name == "by_dataloader":
        sampler.sample_by_dataloader()
    else:
        raise ValueError(f"Unknown mode: {sample_task}")

    logger.info("All tasks finished successfully.")


if __name__ == "__main__":
    main()
