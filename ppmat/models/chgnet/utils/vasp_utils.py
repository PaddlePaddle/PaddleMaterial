from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure


def solve_charge_by_mag(
    structure: Structure,
    default_ox: (dict[str, float] | None) = None,
    ox_ranges: (dict[str, dict[tuple[float, float], int]] | None) = None,
) -> (Structure | None):
    """Solve oxidation states by magmom.

    Args:
        structure (Structure): pymatgen structure with magmoms in site_properties. Dict
            key must be either magmom or final_magmom.
        default_ox (dict[str, float]): default oxidation state for elements.
            Default = dict(Li=1, O=-2)
        ox_ranges (dict[str, dict[tuple[float, float], int]]): user-defined range to
            convert magmoms into formal valence.
            Example for Mn (Default):
                ("Mn": (
                    (0.5, 1.5): 2,
                    (1.5, 2.5): 3,
                    (2.5, 3.5): 4,
                    (3.5, 4.2): 3,
                    (4.2, 5): 2
                ))

    Returns:
        Structure: pymatgen Structure with oxidation states assigned based on magmoms.
    """
    out_structure = structure.copy()
    out_structure.remove_oxidation_states()
    ox_list = []
    solved_ox = True
    default_ox = default_ox or {"Li": 1, "O": -2}
    ox_ranges = ox_ranges or {
        "Mn": {(0.5, 1.5): 2, (1.5, 2.5): 3, (2.5, 3.5): 4, (3.5, 4.2): 3, (4.2, 5): 2}
    }
    magmoms = structure.site_properties.get(
        "final_magmom", structure.site_properties.get("magmom")
    )
    for idx, site in enumerate(out_structure):
        assigned = False
        if site.species_string in ox_ranges:
            for (min_mag, max_mag), mag_ox in ox_ranges[site.species_string].items():
                if min_mag <= magmoms[idx] < max_mag:
                    ox_list.append(mag_ox)
                    assigned = True
                    break
        elif site.species_string in default_ox:
            ox_list.append(default_ox[site.species_string])
            assigned = True
        if not assigned:
            solved_ox = False
    if solved_ox:
        total_charge = sum(ox_list)
        print(f"Solved oxidation state, total_charge={total_charge!r}")
        out_structure.add_oxidation_state_by_site(ox_list)
        return out_structure
    warnings.warn("Failed to solve oxidation state", stacklevel=2)
    return None
