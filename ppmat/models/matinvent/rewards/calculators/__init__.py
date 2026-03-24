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

"""
Property calculators for reward system.

Available calculators:
- PyMatGen: Crystal density, HHI score, element price, crustal abundance
- SynScore: Synthesizability score prediction (requires model files)
- DFTCalc: DFT calculations (band gap, formation energy, etc.)
- FairChem: ML-based properties (bulk modulus, heat capacity)
"""

from ppmat.models.matinvent.rewards.base import Calculator
from ppmat.models.matinvent.rewards.calculators.dft import DFTCalc
from ppmat.models.matinvent.rewards.calculators.fairchem import FairChem
from ppmat.models.matinvent.rewards.calculators.pymatgen import PyMatGen
from ppmat.models.matinvent.rewards.calculators.syn_score import SynScore

__all__ = [
    "Calculator",
    "PyMatGen",
    "SynScore",
    "DFTCalc",
    "FairChem",
]
