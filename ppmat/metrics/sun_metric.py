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
S.U.N. (Stability, Uniqueness, Novelty) metrics for crystal structure generation.

Implements the evaluation pipeline from MiAD paper for assessing generated
crystal structures across three dimensions:
- Stability: energy above hull below a threshold
- Uniqueness: no duplicates among generated structures
- Novelty: not present in the training set

Reference: https://arxiv.org/abs/2511.14426
"""

import logging
import os
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _composition_hash(structure: Structure) -> str:
    """Hash a structure by its sorted atomic numbers for quick grouping."""
    return str(sorted(list(structure.atomic_numbers)))


def _structures_from_cif_strings(cif_strings: List[str]) -> List[Optional[Structure]]:
    """Parse CIF strings into pymatgen Structures."""
    structures = []
    for cif in cif_strings:
        try:
            structures.append(Structure.from_str(cif, fmt="cif"))
        except Exception:
            structures.append(None)
    return structures


def _structures_from_dicts(structure_dicts: List[dict]) -> List[Optional[Structure]]:
    """Build Structures from dict lists with lattice, atom_types, frac_coords."""
    from pymatgen.core import Lattice

    structures = []
    for d in structure_dicts:
        try:
            lattice = Lattice(d["lattice"])
            struct = Structure(
                lattice,
                d["atom_types"],
                d["frac_coords"],
                coords_are_cartesian=False,
            )
            structures.append(struct)
        except Exception:
            structures.append(None)
    return structures


def compute_uniqueness(
    structures: List[Optional[Structure]],
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
    attempt_supercell: bool = True,
    symmetric: bool = True,
) -> List[bool]:
    """Compute uniqueness of generated structures.

    A structure is unique if it does not match any previously seen structure
    with the same composition.

    Args:
        structures: List of pymatgen Structures (None for invalid).
        stol: Site tolerance for StructureMatcher.
        angle_tol: Angle tolerance for StructureMatcher.
        ltol: Length tolerance for StructureMatcher.
        attempt_supercell: Whether to attempt supercell matching.
        symmetric: Whether to use symmetric matching.

    Returns:
        List of booleans indicating uniqueness.
    """
    matcher = StructureMatcher(
        stol=stol, angle_tol=angle_tol, ltol=ltol,
        attempt_supercell=attempt_supercell,
    )
    seen = defaultdict(list)
    uniqueness = []

    for struct in tqdm(structures, desc="Computing uniqueness"):
        if struct is None:
            uniqueness.append(False)
            continue
        h = _composition_hash(struct)
        if h not in seen:
            uniqueness.append(True)
        else:
            for existing in seen[h]:
                if matcher.fit(struct, existing, symmetric=symmetric):
                    uniqueness.append(False)
                    break
            else:
                uniqueness.append(True)
        seen[h].append(struct)

    return uniqueness


def compute_novelty(
    structures: List[Optional[Structure]],
    reference_structures: List[Structure],
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
    attempt_supercell: bool = True,
    symmetric: bool = True,
) -> List[bool]:
    """Compute novelty of generated structures against a reference set.

    A structure is novel if it does not match any structure in the reference
    set (typically the training set).

    Args:
        structures: List of pymatgen Structures (None for invalid).
        reference_structures: Reference structures (training set).
        stol: Site tolerance for StructureMatcher.
        angle_tol: Angle tolerance for StructureMatcher.
        ltol: Length tolerance for StructureMatcher.
        attempt_supercell: Whether to attempt supercell matching.
        symmetric: Whether to use symmetric matching.

    Returns:
        List of booleans indicating novelty.
    """
    matcher = StructureMatcher(
        stol=stol, angle_tol=angle_tol, ltol=ltol,
        attempt_supercell=attempt_supercell,
    )

    reference_by_comp = defaultdict(list)
    for ref in reference_structures:
        h = _composition_hash(ref)
        reference_by_comp[h].append(ref)

    novelty = []
    for struct in tqdm(structures, desc="Computing novelty"):
        if struct is None:
            novelty.append(False)
            continue
        h = _composition_hash(struct)
        if h not in reference_by_comp:
            novelty.append(True)
        else:
            for ref in reference_by_comp[h]:
                if matcher.fit(struct, ref, symmetric=symmetric):
                    novelty.append(False)
                    break
            else:
                novelty.append(True)

    return novelty


def compute_stability(
    energy_above_hull: List[Optional[float]],
    threshold: float = 0.0,
) -> List[bool]:
    """Compute stability from energy above hull values.

    Args:
        energy_above_hull: Energy above hull per atom (None for invalid).
        threshold: Energy threshold in eV/atom. 0.0 for stable, 0.1 for metastable.

    Returns:
        List of booleans indicating stability.
    """
    return [
        (eah is not None and not np.isnan(eah) and eah < threshold)
        for eah in energy_above_hull
    ]


def compute_sun(
    stability: List[bool],
    uniqueness: List[bool],
    novelty: List[bool],
    structures: List[Optional[Structure]],
    min_elements: int = 2,
) -> Dict[str, float]:
    """Compute combined S.U.N. metrics.

    S.U.N. = structure is Stable AND Unique AND Novel AND non-trivial
    (more than one element type).

    Args:
        stability: Stability flags.
        uniqueness: Uniqueness flags.
        novelty: Novelty flags.
        structures: Original structures (for composition check).
        min_elements: Minimum number of element types for non-trivial.

    Returns:
        Dictionary with metric rates.
    """
    n = len(stability)
    assert len(uniqueness) == n and len(novelty) == n

    non_trivial = []
    for s in structures:
        if s is not None and len(set(s.composition)) >= min_elements:
            non_trivial.append(True)
        else:
            non_trivial.append(False)

    sun = [s and u and nv and nt for s, u, nv, nt in
           zip(stability, uniqueness, novelty, non_trivial)]

    def rate(flags):
        return round(100 * sum(flags) / max(len(flags), 1), 2)

    return {
        "total": n,
        "valid": sum(1 for s in structures if s is not None),
        "non_trivial": sum(non_trivial),
        "stability_rate": rate(stability),
        "uniqueness_rate": rate(uniqueness),
        "novelty_rate": rate(novelty),
        "sun_rate": rate(sun),
        "sun_count": sum(sun),
    }


class SUNMetric:
    """S.U.N. metric for evaluating generated crystal structures.

    Computes Stability (energy above hull), Uniqueness (no duplicates),
    Novelty (not in training set), and their combination.

    Args:
        gt_file_path: Path to ground truth CSV (with 'cif' column) for novelty ref.
        stol: Site tolerance for structure matching.
        angle_tol: Angle tolerance for structure matching.
        ltol: Length tolerance for structure matching.
        stability_threshold: Energy above hull threshold in eV/atom.
        attempt_supercell: Whether to attempt supercell matching.
    """

    def __init__(
        self,
        gt_file_path: Optional[str] = None,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        ltol: float = 0.3,
        stability_threshold: float = 0.0,
        attempt_supercell: bool = True,
        use_chgnet: bool = False,
        chgnet_relax: bool = True,
        chgnet_relax_steps: int = 200,
    ):
        self.gt_file_path = gt_file_path
        self.stol = stol
        self.angle_tol = angle_tol
        self.ltol = ltol
        self.stability_threshold = stability_threshold
        self.attempt_supercell = attempt_supercell
        self.use_chgnet = use_chgnet
        self.chgnet_relax = chgnet_relax
        self.chgnet_relax_steps = chgnet_relax_steps
        self._reference_structures = None

    def _load_reference(self):
        """Load reference structures from ground truth CSV."""
        if self._reference_structures is not None:
            return
        if self.gt_file_path is None:
            return
        if not os.path.exists(self.gt_file_path):
            logger.warning(f"Reference file not found: {self.gt_file_path}")
            return

        df = pd.read_csv(self.gt_file_path)
        if "cif" not in df.columns:
            logger.warning("Reference CSV must have a 'cif' column")
            return

        self._reference_structures = []
        for cif_str in tqdm(df["cif"], desc="Loading reference structures"):
            try:
                self._reference_structures.append(
                    Structure.from_str(cif_str, fmt="cif")
                )
            except Exception:
                pass
        logger.info(f"Loaded {len(self._reference_structures)} reference structures")

    def __call__(
        self,
        generated: Union[List[dict], List[str], pd.DataFrame],
        energy_above_hull: Optional[List[Optional[float]]] = None,
    ) -> Dict[str, float]:
        """Run the full S.U.N. evaluation pipeline.

        Args:
            generated: Generated structures. Can be:
                - List of dicts with 'lattice', 'atom_types', 'frac_coords'
                - List of CIF strings
                - DataFrame with a 'cif' column
            energy_above_hull: Optional pre-computed energy above hull values.
                If None, stability is not computed (all marked False).

        Returns:
            Dictionary with S.U.N. metrics.
        """
        # Parse structures
        if isinstance(generated, pd.DataFrame):
            if "cif" in generated.columns:
                structures = _structures_from_cif_strings(generated["cif"].tolist())
            elif "structure" in generated.columns:
                structures = generated["structure"].tolist()
            else:
                raise ValueError("DataFrame must have 'cif' or 'structure' column")
        elif isinstance(generated, list) and len(generated) > 0:
            if isinstance(generated[0], str):
                structures = _structures_from_cif_strings(generated)
            elif isinstance(generated[0], dict):
                structures = _structures_from_dicts(generated)
            else:
                structures = generated
        else:
            raise ValueError("generated must be a list or DataFrame")

        n = len(structures)
        logger.info(f"Evaluating {n} generated structures")

        # Uniqueness
        uniqueness = compute_uniqueness(
            structures,
            stol=self.stol, angle_tol=self.angle_tol, ltol=self.ltol,
            attempt_supercell=self.attempt_supercell,
        )

        # Novelty
        novelty_list = [True] * n  # default: all novel
        self._load_reference()
        if self._reference_structures:
            novelty_list = compute_novelty(
                structures, self._reference_structures,
                stol=self.stol, angle_tol=self.angle_tol, ltol=self.ltol,
                attempt_supercell=self.attempt_supercell,
            )

        # Stability
        if energy_above_hull is not None:
            stability = compute_stability(
                energy_above_hull, threshold=self.stability_threshold
            )
        elif self.use_chgnet:
            from ppmat.metrics.prerelax_chgnet import predict_energy

            energies = predict_energy(structures)
            stability = [
                (e is not None and e < self.stability_threshold)
                for e in energies
            ]
        else:
            stability = [False] * n
            energies = None
            logger.warning(
                "No energy_above_hull provided and use_chgnet=False, "
                "stability metrics will be 0%"
            )

        # Combined S.U.N.
        results = compute_sun(stability, uniqueness, novelty_list, structures)

        if self.use_chgnet and energies is not None:
            valid_energies = [e for e in energies if e is not None]
            if valid_energies:
                results["avg_energy_per_atom"] = float(np.mean(valid_energies))

        # Also compute metastable variant
        if energy_above_hull is not None:
            metastability = compute_stability(
                energy_above_hull, threshold=0.1
            )
            msun_results = compute_sun(
                metastability, uniqueness, novelty_list, structures
            )
            results["metastable_rate"] = msun_results["stability_rate"]
            results["msun_rate"] = msun_results["sun_rate"]
            results["msun_count"] = msun_results["sun_count"]

        # Log results
        logger.info(
            f"S.U.N. Results: "
            f"Stability={results['stability_rate']:.2f}%, "
            f"Uniqueness={results['uniqueness_rate']:.2f}%, "
            f"Novelty={results['novelty_rate']:.2f}%, "
            f"S.U.N.={results['sun_rate']:.2f}%"
        )

        return results
