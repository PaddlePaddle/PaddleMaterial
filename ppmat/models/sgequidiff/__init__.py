# Copyright (C) 2026 Suzhou National Laboratory and Baidu PaddlePaddle team
# This code was jointly developed by Suzhou National Laboratory and Baidu PaddlePaddle team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

try:
    from ppmat.models.sgequidiff.diffusion_model import (
        EquivariantDiffusionModel,
        EquivariantDiffusionModelConfig,
        NoiseScheduler,
    )
    from ppmat.models.sgequidiff.training_wrapper import (
        SGEQUITrainingWrapper,
    )
    from ppmat.models.sgequidiff.crystal_sampler import (
        CrystalSampler,
        CrystalSamplerConfig,
        SpaceGroupSampler,
    )
except ImportError:
    EquivariantDiffusionModel = None
    EquivariantDiffusionModelConfig = None
    NoiseScheduler = None
    SGEQUITrainingWrapper = None
    CrystalSampler = None
    CrystalSamplerConfig = None
    SpaceGroupSampler = None

try:
    from ppmat.models.sgequidiff.weight_utils import (
        download_weight_file,
        download_all_weights,
        load_pretrained_weights,
        PRETRAINED_WEIGHT_URLS,
    )
except ImportError:
    download_weight_file = None
    download_all_weights = None
    load_pretrained_weights = None
    PRETRAINED_WEIGHT_URLS = None

__all__ = [
    "EquivariantDiffusionModel",
    "EquivariantDiffusionModelConfig",
    "NoiseScheduler",
    "SGEQUITrainingWrapper",
    "CrystalSampler",
    "CrystalSamplerConfig",
    "SpaceGroupSampler",
    "download_weight_file",
    "download_all_weights",
    "load_pretrained_weights",
    "PRETRAINED_WEIGHT_URLS",
]
