from collections import defaultdict
from itertools import combinations
from typing import Any, Callable, Iterable, TypeVar

import numpy as np
from mattergen.evaluation.utils.globals import MAX_RMSD
from mattergen.evaluation.utils.structure_matcher import RMSDStructureMatcher
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

OptionalNumber = int | float | None
PropertyConstraint = tuple[OptionalNumber, OptionalNumber]


def generate_reduced_formula_dict(
    entries: Iterable[ComputedStructureEntry],
) -> dict[str, list[ComputedStructureEntry]]:
    """Generate a dictionary of entries with the same reduced formula."""

    def keyfunc(entry: ComputedStructureEntry) -> str:
        entry.structure.unset_charge()
        return entry.structure.remove_oxidation_states().composition.reduced_formula

    return group_list_items_into_dict(entries, keyfunc=keyfunc)


def generate_chemsys_dict(
    entries: Iterable[ComputedStructureEntry],
) -> dict[str, list[ComputedStructureEntry]]:
    """Generate a dictionary of entries with the same chemical system."""

    def keyfunc(entry: ComputedStructureEntry) -> str:
        return "-".join(sorted({el.symbol for el in entry.composition.elements}))

    return group_list_items_into_dict(entries, keyfunc=keyfunc)


T = TypeVar("T")


def group_list_items_into_dict(
    items: Iterable[T], keyfunc: Callable[[Any], str]
) -> dict[str, list[T]]:
    """Group a list of items into a dictionary with the same key."""
    result = defaultdict(list)
    for item in items:
        result[keyfunc(item)].append(item)
    return result


def compute_rmsd_angstrom(struc1: Structure, struc2: Structure) -> float:
    """Compute RMSD during relaxation in units of angstrom"""
    match = RMSDStructureMatcher().get_rms_dist(struc1, struc2)

    def av_lat(l1: Lattice, l2: Lattice):
        params = (np.array(l1.parameters) + np.array(l2.parameters)) / 2
        return Lattice.from_parameters(*params)

    avg_l = av_lat(struc1.lattice, struc2.lattice)
    normalization = (len(struc1) / avg_l.volume) ** (1 / 3)
    if match is None:
        return MAX_RMSD / normalization
    return match[0] / normalization


def expand_into_subsystems(chemical_system: str) -> list[tuple[str, ...]]:
    elements = chemical_system.split("-")
    list_combinations = []
    for n in range(1, len(elements) + 1):
        list_combinations += list(combinations(elements, n))
    return list_combinations


def preprocess_structure(structure: Structure) -> Structure:
    sga = SpacegroupAnalyzer(structure)
    return (
        sga.get_refined_structure()
        .get_primitive_structure()
        .get_reduced_structure(reduction_algo="LLL")
    )
