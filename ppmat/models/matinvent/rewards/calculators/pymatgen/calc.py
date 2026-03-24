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
PyMatGen property calculators.

Calculates various crystallographic and materials properties using pymatgen.
"""

import os
from typing import List
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from pymatgen.analysis.cost import CostAnalyzer
from pymatgen.analysis.cost import CostDBElements
from pymatgen.analysis.hhi import HHIModel
from pymatgen.analysis.interfaces.substrate_analyzer import SubstrateAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from smact import Element as SmactElement

from ppmat.models.matinvent.rewards.base import Calculator
from ppmat.models.matinvent.rewards.calculators.pymatgen import SUBSTRATE_PATH

# 参考移植: matinvent/rewards/calculators/pymatgen/calc.py
SUB_MILLERS = {
    "Si": [(1, 0, 0)],
    "GaAs": [(1, 0, 0)],
    "InP": [(1, 0, 0)],
}


def abundance_crust(struc: Structure) -> float:
    """
    Given a pymatgen.Structure return the weighted average
    of the crustal abundance (in ppm)
    based on the mass fraction of each element in the structure.
    """
    comp = struc.composition
    weighted_abundance = 0.0
    for el, weight_frac in comp.to_weight_dict.items():
        try:
            crust_abundance = SmactElement(el).crustal_abundance
            assert isinstance(crust_abundance, float)
        except Exception:
            return np.nan
        weighted_abundance += weight_frac * crust_abundance
    if weighted_abundance <= 0.0:
        weighted_abundance = np.nan
    return weighted_abundance


def calc_density(struc_list: List[Structure]) -> np.ndarray:
    """
    Given a list of pymatgen.Structure and return their density (unit: g/cm^3).
    """
    density = np.array([struc.density for struc in struc_list])
    return density


def calc_hhi(struc_list: List[Structure]) -> np.ndarray:
    """
    Given a list of pymatgen.Structure and return their
    Herfindahl-Hirschman Index (HHI) score based on geological reserves,
    for evaluating their supply and demand risk.
    """
    calc = HHIModel()
    hhi_list = []
    for s in struc_list:
        _hhi = calc.get_hhi_reserve(s.composition)
        if _hhi is not None:
            hhi_list.append(_hhi)
        else:
            hhi_list.append(np.nan)
    hhi_arr = np.array(hhi_list, dtype=float)
    return hhi_arr


def calc_price(struc_list: List[Structure]) -> np.ndarray:
    """
    Given a list of pymatgen.Structure return their weighted
    average of prices (unit: USD/kg)
    based on the mass fraction and price of each element in a structure.
    """
    ca = CostAnalyzer(CostDBElements())
    price_list = []
    for struc in struc_list:
        try:
            price = ca.get_cost_per_kg(struc.composition)
            assert isinstance(price, float)
            price_list.append(price)
        except Exception:
            price_list.append(np.nan)
    price_arr = np.array(price_list, dtype=float)
    return price_arr


def calc_abundance_crust(struc_list: List[Structure]) -> np.ndarray:
    """
    Given a list of pymatgen.Structure return the weighted
    average of the crustal abundance (unit: ppm)
    based on the mass fraction of each element in the structure.
    """
    abundance_list = [abundance_crust(s) for s in struc_list]
    abundance_arr = np.array(abundance_list, dtype=float)
    return abundance_arr


def calc_log_abundance_crust(struc_list: List[Structure]) -> np.ndarray:
    """
    Given a list of pymatgen.Structure return their log10 crustal abundance (unit: ppm)
    based on the mass fraction of each element in the structure.
    """
    abundance_arr = calc_abundance_crust(struc_list)
    log_abundance_arr = np.log10(abundance_arr)
    return log_abundance_arr


# 参考移植 matinvent/rewards/calculators/pymatgen/calc.py
def calc_mcia(
    struc_list: List[Structure],
    substrate: Structure,
    substrate_millers: ArrayLike = None,
) -> np.ndarray:
    """Calculate the minimal co-incident area
    (MCIA, unit: Angstrom^2) between film and substrate
    https://docs.materialsproject.org/methodology/materials-methodology/suggested-substrates
    https://pubs.acs.org/doi/10.1021/acsami.6b01630
    https://www.sciencedirect.com/topics/engineering/silicon-single-crystal
    https://link.springer.com/chapter/10.1007/978-3-030-80135-9_1
    O'Mara, William C. (1990). Handbook of Semiconductor
    Silicon Technology. William Andrew Inc. pp. 349-352.

    Args:
        struc_list (List[Structure]): list of film structures
        substrate (Structure): substrate structure
        substrate_millers (ArrayLike): substrate facets to consider in search as
            defined by miller indices

    Returns:
        np.ndarray[float]: computed MCIA
    """
    sa = SubstrateAnalyzer(film_max_miller=1, substrate_max_miller=1)
    substrate = SpacegroupAnalyzer(
        substrate, symprec=0.1
    ).get_conventional_standard_structure()

    sub_comp = substrate.composition.reduced_formula
    if substrate_millers is None and sub_comp in SUB_MILLERS:
        substrate_millers = SUB_MILLERS[sub_comp]

    mcia_list = []
    for struc in struc_list:
        try:
            film = SpacegroupAnalyzer(
                struc, symprec=0.1
            ).get_conventional_standard_structure()
            matches = sa.calculate(
                film=film,
                substrate=substrate,
                substrate_millers=substrate_millers,
                lowest=True,
            )
            mcia = min([m.match_area for m in matches])
            assert isinstance(mcia, float)
            mcia_list.append(mcia)
        except Exception:
            mcia_list.append(np.nan)

    mcia_arr = np.array(mcia_list, dtype=float)
    return mcia_arr


class PyMatGen(Calculator):
    """PyMatGen property calculator.

    Supports calculating:
    - density: Crystal density (g/cm^3)
    - hhi: Herfindahl-Hirschman Index (supply risk)
    - price: Element price (USD/kg)
    - abundance: Crustal abundance (ppm)
    - log_abundance: Log10 crustal abundance
    - mcia: Minimal co-incident area (Å^2)
    - num_atoms: Number of atoms in structure
    - num_elements: Number of unique elements
    - volume: Unit cell volume (Å^3)
    """

    def __init__(
        self, root_dir: str, task: str = "density", substrate: str = "Si"
    ) -> None:
        """Initialize PyMatGen calculator.

        Args:
            root_dir: Directory for output files
            task: Property to calculate (density, hhi, price, abundance, log_abundance,
                  mcia, num_atoms, num_elements, volume)
            substrate: Substrate material (for MCIA calculation)
        """
        super().__init__(root_dir, task)
        self.substrate = Structure.from_file(
            os.path.join(SUBSTRATE_PATH, f"{substrate}.cif")
        )

    def calc(
        self, samples: Tuple[List[Structure], str], label: str = "tmp"
    ) -> np.ndarray:
        """Calculate properties for a list of structures.

        Args:
            samples: Tuple of (structure_list, path)
            label: Label for output file

        Returns:
            Array of calculated property values
        """
        struc_list = samples[0]
        out_path = os.path.join(self.root_dir, f"{label}.txt")
        out_path = os.path.abspath(out_path)

        if self.task == "density":
            results = calc_density(struc_list)
        elif self.task == "hhi":
            results = calc_hhi(struc_list)
        elif self.task == "price":
            results = calc_price(struc_list)
        elif self.task == "abundance":
            results = calc_abundance_crust(struc_list)
        elif self.task == "log_abundance":
            results = calc_log_abundance_crust(struc_list)
        elif self.task == "mcia":
            results = calc_mcia(struc_list, self.substrate)
        elif self.task == "num_atoms":
            results = np.array([len(struc) for struc in struc_list], dtype=float)
        elif self.task == "num_elements":
            results = np.array(
                [len(struc.composition.elements) for struc in struc_list],
                dtype=float,
            )
        elif self.task == "volume":
            results = np.array([struc.volume for struc in struc_list], dtype=float)
        else:
            raise ValueError(f"{self.task} is unknown task for PyMatGen calculator!")

        np.savetxt(out_path, results, fmt="%.8f")
        return results
