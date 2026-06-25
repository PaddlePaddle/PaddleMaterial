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

import os
import numpy as np
from pymatgen.analysis.cost import CostAnalyzer, CostDBElements
from pymatgen.analysis.hhi import HHIModel
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smact import Element as SmactElement
from ppmat.models.matinvent.rewards.base import Calculator
from ppmat.models.matinvent.rewards.calculators.pymatgen import SUBSTRATE_PATH

SUB_MILLERS = {"Si": [(1, 0, 0)], "GaAs": [(1, 0, 0)], "InP": [(1, 0, 0)]}


def _abundance(s):
    try:
        wa = sum(wf * SmactElement(el).crustal_abundance
                 for el, wf in s.composition.to_weight_dict.items())
        return np.nan if wa <= 0.0 else wa
    except Exception:
        return np.nan


def _hhi(x):
    m = HHIModel()
    return np.array([m.get_hhi_reserve(s.composition) or np.nan for s in x], dtype=float)


def _price(x):
    ca = CostAnalyzer(CostDBElements())
    out = []
    for s in x:
        try:
            out.append(ca.get_cost_per_kg(s.composition))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)


def _mcia(x, sub, millers=None):
    sa = SubstrateAnalyzer(film_max_miller=1, substrate_max_miller=1)
    sub = SpacegroupAnalyzer(sub, symprec=0.1).get_conventional_standard_structure()
    rc = sub.composition.reduced_formula
    if millers is None and rc in SUB_MILLERS:
        millers = SUB_MILLERS[rc]
    out = []
    for s in x:
        try:
            film = SpacegroupAnalyzer(s, symprec=0.1).get_conventional_standard_structure()
            out.append(min(m.match_area for m in sa.calculate(film=film, substrate=sub,
                        substrate_millers=millers, lowest=True)))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)


class PyMatGen(Calculator):
    def __init__(self, root_dir, task="density", substrate="Si"):
        super().__init__(root_dir, task)
        self.substrate = Structure.from_file(os.path.join(SUBSTRATE_PATH, f"{substrate}.cif"))
        self._tasks = {
            "density": lambda x: np.array([s.density for s in x]),
            "hhi": _hhi,
            "price": _price,
            "abundance": lambda x: np.array([_abundance(s) for s in x]),
            "log_abundance": lambda x: np.log10(np.array([_abundance(s) for s in x])),
            "mcia": lambda x: _mcia(x, self.substrate),
            "num_atoms": lambda x: np.array([len(s) for s in x], dtype=float),
            "num_elements": lambda x: np.array([len(s.composition.elements) for s in x], dtype=float),
            "volume": lambda x: np.array([s.volume for s in x], dtype=float),
        }

    def calc(self, samples, label="tmp"):
        struc_list = samples[0]
        fn = self._tasks.get(self.task)
        if fn is None:
            raise ValueError(f"Unknown task: {self.task}")
        results = fn(struc_list)
        np.savetxt(os.path.abspath(os.path.join(self.root_dir, f"{label}.txt")), results, fmt="%.8f")
        return results
