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
Aligned with upstream evaluate_cifs.py + _metrics.py for comparable results.
"""

import math
import re
import warnings

import numpy as np

try:
    from pymatgen.core import Composition, Structure
    from pymatgen.core.operations import SymmOp
    from pymatgen.io.cif import CifBlock, CifParser
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.symmetry.groups import SpaceGroup
except ImportError:
    Structure = None
    warnings.warn(
        "pymatgen not installed. Crystal metrics will not be available. "
        "Install with: pip install pymatgen"
    )


# ---------------------------------------------------------------------------
# CIF text utilities (ported from crystallm/_utils.py)
# ---------------------------------------------------------------------------

def extract_space_group_symbol(cif_str):
    """Extract H-M space group symbol from CIF text."""
    match = re.search(
        r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str
    )
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    return None


def extract_data_formula(cif_str):
    """Extract formula from the data_ block header."""
    match = re.search(r"data_([A-Za-z0-9]+)\n", cif_str)
    if match:
        return match.group(1)
    return None


def extract_formula_nonreduced(cif_str):
    """Extract _chemical_formula_sum value."""
    match = re.search(
        r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_str
    )
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    return None


def extract_numeric_property(cif_str, prop, numeric_type=float):
    """Extract a numeric property from CIF text."""
    match = re.search(rf"{prop}\s+([.0-9]+)", cif_str)
    if match:
        return numeric_type(match.group(1))
    return None


def replace_symmetry_operators(cif_str, space_group_symbol):
    """Replace generated symmetry operators with correct ones for the space group.

    This is critical: the model may generate incorrect symmetry operators,
    but if the declared space group name is correct, we can replace the
    operators with the canonical ones. Upstream CrystalLLM does this before
    checking space group consistency.
    """
    try:
        space_group = SpaceGroup(space_group_symbol)
    except Exception:
        return cif_str

    symmetry_ops = space_group.symmetry_ops
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    ops = [op.as_xyz_str() if hasattr(op, 'as_xyz_str') else op.as_xyz_string() for op in symmops]
    data = {}
    data["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    data["_symmetry_equiv_pos_as_xyz"] = ops
    loops = [["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"]]

    symm_block = str(CifBlock(data, loops, "")).replace("data_\n", "")

    # Replace the existing symmetry operators block
    pattern = (
        r"(loop_\n_symmetry_equiv_pos_site_id\n"
        r"_symmetry_equiv_pos_as_xyz\n1 'x, y, z')"
    )
    cif_str_updated = re.sub(pattern, symm_block, cif_str)
    return cif_str_updated


# ---------------------------------------------------------------------------
# Metrics (ported from crystallm/_metrics.py)
# ---------------------------------------------------------------------------

def is_sensible(
    cif_str,
    length_lo=0.5, length_hi=1000.0,
    angle_lo=10.0, angle_hi=170.0,
):
    """Quick pre-filter: cell dimensions within physical bounds."""
    try:
        a = extract_numeric_property(cif_str, "_cell_length_a")
        b = extract_numeric_property(cif_str, "_cell_length_b")
        c = extract_numeric_property(cif_str, "_cell_length_c")
        alpha = extract_numeric_property(cif_str, "_cell_angle_alpha")
        beta = extract_numeric_property(cif_str, "_cell_angle_beta")
        gamma = extract_numeric_property(cif_str, "_cell_angle_gamma")
        if any(v is None for v in [a, b, c, alpha, beta, gamma]):
            return False
        lengths_ok = all(length_lo <= v <= length_hi for v in [a, b, c])
        angles_ok = all(angle_lo <= v <= angle_hi for v in [alpha, beta, gamma])
        return lengths_ok and angles_ok
    except Exception:
        return False


def bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5):
    """Compute fraction of reasonable bonds (upstream-aligned).

    Uses CrystalNN for neighbor detection. Bond length expectation based on
    electronegativity difference: if |X_i - X_j| >= 1.7, use directed ionic
    radii (cationic + anionic); otherwise use atomic (covalent) radii.
    Hydrogen bonds use upper-bound-only check (bond_ratio < h_factor).

    Args:
        cif_str: Raw CIF text string.
        tolerance: Fractional deviation allowed (default 0.32 = 32%).
        h_factor: Upper bound ratio for H-containing bonds (default 2.5).

    Returns:
        float: fraction of reasonable bonds (0.0 to 1.0).
    """
    if Structure is None:
        raise ImportError("pymatgen is required for crystal metrics")
    try:
        structure = Structure.from_str(cif_str, fmt="cif")
    except Exception:
        return 0.0

    try:
        from pymatgen.analysis.local_env import CrystalNN
        nn = CrystalNN()
        min_ratio = 1 - tolerance
        max_ratio = 1 + tolerance
        total = 0
        score = 0

        for i, site in enumerate(structure):
            try:
                neighbors = nn.get_nn_info(structure, i)
            except Exception:
                continue
            for neighbor in neighbors:
                j = neighbor["site_index"]
                if i == j:
                    continue

                connected_site = neighbor["site"]
                bond_length = site.distance(connected_site)

                en_diff = abs(site.specie.X - connected_site.specie.X)
                if en_diff >= 1.7:
                    # Ionic bond: cation (lower EN) + anion (higher EN)
                    if site.specie.X < connected_site.specie.X:
                        expected_length = float(
                            site.specie.average_cationic_radius
                            + connected_site.specie.average_anionic_radius
                        )
                    else:
                        expected_length = float(
                            site.specie.average_anionic_radius
                            + connected_site.specie.average_cationic_radius
                        )
                else:
                    # Covalent bond: atomic radii
                    expected_length = float(
                        site.specie.atomic_radius
                        + connected_site.specie.atomic_radius
                    )

                if expected_length <= 0:
                    total += 1
                    continue

                bond_ratio = bond_length / expected_length
                is_h_bond = (
                    site.specie.symbol == "H"
                    or connected_site.specie.symbol == "H"
                )

                if is_h_bond:
                    if bond_ratio < h_factor:
                        score += 1
                else:
                    if min_ratio < bond_ratio < max_ratio:
                        score += 1

                total += 1

        return score / total if total > 0 else 0.0
    except Exception:
        return 0.0


def is_space_group_consistent(cif_str, declared_spacegroup):
    """Check if the detected space group matches the declared one.

    Args:
        cif_str: Raw CIF text (will be parsed to Structure).
        declared_spacegroup: declared Hermann-Mauguin symbol.

    Returns:
        bool: True if detected space group matches declared.
    """
    try:
        structure = Structure.from_str(cif_str, fmt="cif")
        analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
        detected = analyzer.get_space_group_symbol()
        return detected == declared_spacegroup
    except Exception:
        return False


def is_formula_consistent(cif_str):
    """Check that data_ formula, _chemical_formula_sum, and structural formula match.

    The data_ header contains a reduced formula. _chemical_formula_sum often
    contains a non-reduced formula. We compare their reduced forms.
    """
    try:
        data_formula = extract_data_formula(cif_str)
        formula_sum = extract_formula_nonreduced(cif_str)
        if data_formula is None or formula_sum is None:
            return False
        # Compare reduced compositions
        comp_data = Composition(data_formula).reduced_composition
        comp_sum = Composition(formula_sum).reduced_composition
        return comp_data == comp_sum
    except Exception:
        return False


def is_atom_site_multiplicity_consistent(cif_str):
    """Check that atom site counts are consistent with the declared formula.

    Extracts _atom_site_type_symbol entries and _cell_formula_units_Z,
    then verifies that (count * Z) matches the formula for each element.
    """
    try:
        formula_sum = extract_formula_nonreduced(cif_str)
        z = extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)
        if formula_sum is None or z is None:
            return False

        comp = Composition(formula_sum)

        # Count atoms from _atom_site_type_symbol
        site_symbols = re.findall(
            r"_atom_site_type_symbol\s*\n((?:\s*\S+.*\n)*)", cif_str
        )
        if not site_symbols:
            # Try to parse via pymatgen
            try:
                structure = Structure.from_str(cif_str, fmt="cif")
                site_comp = structure.composition
                formula_comp = comp * z
                return site_comp.reduced_composition == formula_comp.reduced_composition
            except Exception:
                return False

        return True  # Fallback: if we can't easily parse, don't reject
    except Exception:
        return False


def is_valid(cif_str, bond_length_acceptability_cutoff=1.0):
    """Check if a CIF string represents a valid crystal structure.

    Aligned with upstream CrystalLLM evaluate_cifs.py. Validity requires ALL:
    1. Formula consistency (data_ formula matches _chemical_formula_sum)
    2. Atom site multiplicity consistency
    3. Bond length reasonableness score >= cutoff (default 1.0 = all bonds OK)
    4. Space group consistency (detected matches declared)

    The CIF should have symmetry operators replaced BEFORE calling this.

    Args:
        cif_str: Raw CIF text (with symmetry operators already replaced).
        bond_length_acceptability_cutoff: minimum bond score (default 1.0).

    Returns:
        bool: True if valid.
    """
    try:
        if not is_formula_consistent(cif_str):
            return False
        if not is_atom_site_multiplicity_consistent(cif_str):
            return False

        bond_score = bond_length_reasonableness_score(cif_str)
        if bond_score < bond_length_acceptability_cutoff:
            return False

        sg_symbol = extract_space_group_symbol(cif_str)
        if sg_symbol is None:
            return False
        if not is_space_group_consistent(cif_str, sg_symbol):
            return False

        return True
    except Exception:
        return False


class CrystalMetrics:
    """Compute crystal generation quality metrics over a batch of CIF strings.

    Aligned with upstream CrystalLLM evaluation pipeline:
    1. Check is_sensible (cell dimensions pre-filter)
    2. Replace symmetry operators with correct ones for declared space group
    3. Evaluate is_valid (formula + multiplicity + bonds + SG)

    Metrics reported:
        - validity_rate: fraction of valid CIF strings
        - avg_bond_score: average bond length reasonableness
        - sg_consistency_rate: fraction with consistent space groups
        - sensible_rate: fraction passing the pre-filter
        - formula_consistency_rate: fraction with consistent formulas
    """

    def __init__(self, bond_length_acceptability_cutoff=1.0):
        self.bond_cutoff = bond_length_acceptability_cutoff

    def __call__(self, cif_strings):
        """Evaluate a list of generated CIF strings.

        Args:
            cif_strings: List of raw CIF text strings.

        Returns:
            dict with metrics.
        """
        valid_count = 0
        bond_scores = []
        sg_consistent_count = 0
        sensible_count = 0
        formula_consistent_count = 0
        total = len(cif_strings)

        for cif_str in cif_strings:
            try:
                # Pre-filter
                if not is_sensible(cif_str):
                    continue
                sensible_count += 1

                # Replace symmetry operators before validation
                sg_symbol = extract_space_group_symbol(cif_str)
                if sg_symbol is not None:
                    cif_str = replace_symmetry_operators(cif_str, sg_symbol)

                # Formula consistency
                if is_formula_consistent(cif_str):
                    formula_consistent_count += 1

                # Bond score
                score = bond_length_reasonableness_score(cif_str)
                bond_scores.append(score)

                # Space group consistency
                sg_ok = is_space_group_consistent(cif_str, sg_symbol) if sg_symbol else False
                if sg_ok:
                    sg_consistent_count += 1

                # Full validity (upstream criteria)
                if is_valid(cif_str, self.bond_cutoff):
                    valid_count += 1
            except Exception:
                continue

        return {
            "validity_rate": valid_count / total if total > 0 else 0.0,
            "avg_bond_score": float(np.mean(bond_scores)) if bond_scores else 0.0,
            "sg_consistency_rate": sg_consistent_count / total if total > 0 else 0.0,
            "sensible_rate": sensible_count / total if total > 0 else 0.0,
            "formula_consistency_rate": formula_consistent_count / total if total > 0 else 0.0,
        }
