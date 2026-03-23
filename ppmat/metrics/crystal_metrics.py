# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Crystal Validity Metrics for CrystalLLM.

Ported from lantunes/CrystaLLM (MIT License).
Uses pymatgen for crystal structure analysis.
"""

import warnings

import numpy as np

try:
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.io.cif import CifParser
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError:
    CrystalNN = None
    warnings.warn(
        "pymatgen not installed. Crystal metrics will not be available. "
        "Install with: pip install pymatgen"
    )


def bond_length_reasonableness_score(structure):
    """Compute the fraction of bond lengths that are reasonable.

    For each bond in the structure, compute an expected bond length based on
    the average of covalent and ionic radii. A bond is 'reasonable' if its
    actual length is within 40% of the expected length.

    Args:
        structure: pymatgen Structure object.

    Returns:
        float: fraction of reasonable bonds (0.0 to 1.0).
    """
    if CrystalNN is None:
        raise ImportError("pymatgen is required for crystal metrics")

    try:
        nn = CrystalNN()
        all_bonds = 0
        reasonable_bonds = 0
        for i in range(len(structure)):
            neighbors = nn.get_nn_info(structure, i)
            element_i = structure[i].specie
            for neighbor in neighbors:
                element_j = neighbor["site"].specie
                distance = neighbor["site"].distance(structure[i])

                # Expected length from average of radii
                try:
                    r_cov_i = element_i.atomic_radius or 0
                    r_cov_j = element_j.atomic_radius or 0
                    r_ionic_i = element_i.average_ionic_radius or 0
                    r_ionic_j = element_j.average_ionic_radius or 0
                    expected = (r_cov_i + r_cov_j + r_ionic_i + r_ionic_j) / 2
                except Exception:
                    expected = 2.0  # fallback

                if expected > 0:
                    ratio = abs(distance - expected) / expected
                    if ratio <= 0.4:
                        reasonable_bonds += 1
                all_bonds += 1

        return reasonable_bonds / all_bonds if all_bonds > 0 else 0.0
    except Exception:
        return 0.0


def is_space_group_consistent(structure, declared_spacegroup):
    """Check if the detected space group matches the declared one.

    Args:
        structure: pymatgen Structure object.
        declared_spacegroup: declared Hermann-Mauguin symbol (string).

    Returns:
        bool: True if detected space group matches declared.
    """
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
        detected = analyzer.get_space_group_symbol()
        return detected == declared_spacegroup
    except Exception:
        return False


def is_valid(cif_string):
    """Check if a CIF string represents a valid crystal structure.

    Validity requires:
    1. Parseable as CIF → pymatgen Structure
    2. Sensible structure (reasonable bond lengths AND consistent space group)

    Args:
        cif_string: Raw CIF text.

    Returns:
        bool: True if valid.
    """
    try:
        parser = CifParser.from_str(cif_string)
        structures = parser.parse_structures()
        if not structures:
            return False
        structure = structures[0]

        # Extract declared space group from CIF
        cif_dict = parser.as_dict()
        key = list(cif_dict.keys())[0]
        declared_sg = cif_dict[key].get("_symmetry_space_group_name_H-M", "")

        bond_score = bond_length_reasonableness_score(structure)
        sg_consistent = is_space_group_consistent(structure, declared_sg)

        return bond_score > 0.5 and sg_consistent
    except Exception:
        return False


class CrystalMetrics:
    """Compute crystal generation quality metrics over a batch of CIF strings.

    Metrics reported:
        - validity_rate: fraction of valid CIF strings
        - avg_bond_score: average bond length reasonableness
        - sg_consistency_rate: fraction with consistent space groups
    """

    def __init__(self):
        pass

    def __call__(self, cif_strings):
        """Evaluate a list of generated CIF strings.

        Args:
            cif_strings: List of raw CIF text strings.

        Returns:
            dict with validity_rate, avg_bond_score, sg_consistency_rate.
        """
        valid_count = 0
        bond_scores = []
        sg_consistent_count = 0
        total = len(cif_strings)

        for cif_str in cif_strings:
            try:
                parser = CifParser.from_str(cif_str)
                structures = parser.parse_structures()
                if not structures:
                    continue
                structure = structures[0]

                cif_dict = parser.as_dict()
                key = list(cif_dict.keys())[0]
                declared_sg = cif_dict[key].get("_symmetry_space_group_name_H-M", "")

                score = bond_length_reasonableness_score(structure)
                bond_scores.append(score)
                sg_ok = is_space_group_consistent(structure, declared_sg)

                if sg_ok:
                    sg_consistent_count += 1
                if score > 0.5 and sg_ok:
                    valid_count += 1
            except Exception:
                continue

        return {
            "validity_rate": valid_count / total if total > 0 else 0.0,
            "avg_bond_score": float(np.mean(bond_scores)) if bond_scores else 0.0,
            "sg_consistency_rate": sg_consistent_count / total if total > 0 else 0.0,
        }
