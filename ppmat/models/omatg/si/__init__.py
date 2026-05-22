# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stochastic Interpolants (SI) module for OMatG.
"""

from ppmat.models.omatg.si.abstracts import (
    Corrector,
    Epsilon,
    Interpolant,
    LatentGamma,
    Sigma,
    StochasticInterpolant,
    StochasticInterpolantSpecies,
    TimeChecker,
    Tau,
)
from ppmat.models.omatg.si.corrector import (
    IdentityCorrector,
    PeriodicBoundaryConditionsCorrector,
)
from ppmat.models.omatg.si.gamma import (
    LatentGammaSqrt,
    LatentGammaEncoderDecoder,
)
from ppmat.models.omatg.si.epsilon import (
    ConstantEpsilon,
    VanishingEpsilon,
)
from ppmat.models.omatg.si.sigma import GeometricSigma
from ppmat.models.omatg.si.tau import (
    TauConstantSchedule,
    TauLinearSchedule,
    TauCosineSchedule,
)
from ppmat.models.omatg.si.interpolants import (
    EncoderDecoderInterpolant,
    ExponentialInterpolant,
    LinearInterpolant,
    PeriodicLinearInterpolant,
    PeriodicEncoderDecoderInterpolant,
    ScoreBasedDiffusionModelInterpolantVE,
    ScoreBasedDiffusionModelInterpolantVP,
    TrigonometricInterpolant,
)
from ppmat.models.omatg.si.discrete_flow_matching_mask import (
    DiscreteFlowMatchingMask,
)
from ppmat.models.omatg.si.single_stochastic_interpolant import (
    DifferentialEquationType,
    SingleStochasticInterpolant,
)
from ppmat.models.omatg.si.single_stochastic_interpolant_os import (
    SingleStochasticInterpolantOS,
)
from ppmat.models.omatg.si.single_stochastic_interpolant_identity import (
    SingleStochasticInterpolantIdentity,
)
from ppmat.models.omatg.si.stochastic_interpolants import (
    BIG_TIME,
    DataField,
    SMALL_TIME,
    reshape_t,
    StochasticInterpolants,
)
from ppmat.models.omatg.si.factory import build_si_from_cfg, build_sampler_from_cfg

__all__ = [
    "Corrector",
    "Epsilon",
    "Interpolant",
    "LatentGamma",
    "Sigma",
    "StochasticInterpolant",
    "StochasticInterpolantSpecies",
    "TimeChecker",
    "Tau",
    "IdentityCorrector",
    "PeriodicBoundaryConditionsCorrector",
    "LatentGammaSqrt",
    "LatentGammaEncoderDecoder",
    "ConstantEpsilon",
    "VanishingEpsilon",
    "GeometricSigma",
    "TauConstantSchedule",
    "TauLinearSchedule",
    "TauCosineSchedule",
    "EncoderDecoderInterpolant",
    "ExponentialInterpolant",
    "LinearInterpolant",
    "PeriodicLinearInterpolant",
    "PeriodicEncoderDecoderInterpolant",
    "ScoreBasedDiffusionModelInterpolantVE",
    "ScoreBasedDiffusionModelInterpolantVP",
    "TrigonometricInterpolant",
    "DiscreteFlowMatchingMask",
    "DifferentialEquationType",
    "SingleStochasticInterpolant",
    "SingleStochasticInterpolantOS",
    "SingleStochasticInterpolantIdentity",
    "BIG_TIME",
    "DataField",
    "SMALL_TIME",
    "reshape_t",
    "StochasticInterpolants",
    "build_si_from_cfg",
    "build_sampler_from_cfg",
]
