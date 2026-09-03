# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import hashlib
import os
from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from ppmat.metrics.streaming_base import StreamingMetricBase
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.crystal import lattices_to_params_shape_numpy

if TYPE_CHECKING:
    import pandas as pd

# StructureMatcher tolerances (fractional / degrees / lattice)
_MATCHER_STOL = 0.5
_MATCHER_ANGLE_TOL = 10.0
_MATCHER_LTOL = 0.3
# Match the full unit cell only, without supercell or symmetry reduction
_MATCHER_ATTEMPT_SUPERCELL = False
_MATCHER_SYMMETRIC = False
# e_above_hull below these bounds counts as stable / metastable
_STABILITY_THRESHOLD = 0.0
_METASTABILITY_THRESHOLD = 0.1


def _composition_hash(structure) -> str:
    # Composition key used to bucket structures before StructureMatcher.
    return str(sorted(list(structure.atomic_numbers)))


def _structure_from_array(crystal_data):
    """Build a pymatgen Structure from an array-style dict.

    Supported keys: ``frac_coords``, ``atom_types``, plus either
    ``lengths`` + ``angles`` or ``lattice`` (3x3 matrix).

    NOTE: metrics layer must not depend on ppmat.datasets (see
    .agents/rules/metrics_layer_independence.md), so this intentionally
    re-implements the ``array`` branch of ``BuildStructure.build_one``;
    keep both in sync when either changes.
    """
    from pymatgen.core import Lattice
    from pymatgen.core import Structure

    frac_coords = crystal_data["frac_coords"]
    atom_types = crystal_data["atom_types"]
    if "lengths" in crystal_data and "angles" in crystal_data:
        lengths = crystal_data["lengths"]
        angles = crystal_data["angles"]
    else:
        lattice = crystal_data["lattice"]
        if isinstance(lattice, list):
            lattice = np.asarray(lattice)
        lengths, angles = lattices_to_params_shape_numpy(lattice)
    if isinstance(lengths, np.ndarray):
        lengths = lengths.tolist()
    if isinstance(angles, np.ndarray):
        angles = angles.tolist()
    return Structure(
        lattice=Lattice.from_parameters(*(lengths + angles)),
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    )


def _parse_structures(raw_list, format_str):
    from pymatgen.core import Structure

    results = []
    for item in raw_list:
        try:
            if format_str == "cif_str":
                results.append(Structure.from_str(item, fmt="cif"))
            elif format_str == "array":
                results.append(_structure_from_array(item))
            else:
                raise ValueError(f"Invalid format specified: {format_str}")
        except Exception as exc:
            logger.debug("Failed to parse structure: %s", exc)
            results.append(None)
    return results


def _match_against(
    structures: List[Optional["Structure"]],
    reference_by_comp: dict,
    matcher,
    symmetric: bool = False,
    dynamic: bool = False,
) -> List[bool]:
    from tqdm import tqdm

    results = []
    for struct in tqdm(structures, desc="Matching structures"):
        if struct is None:
            results.append(False)
            continue
        h = _composition_hash(struct)
        refs = reference_by_comp.get(h)
        if refs is None:
            results.append(True)
        else:
            for ref in refs:
                if matcher.fit(struct, ref, symmetric=symmetric):
                    results.append(False)
                    break
            else:
                results.append(True)
        if dynamic and results[-1]:
            reference_by_comp[h].append(struct)
    return results


def _make_matcher(stol, angle_tol, ltol, attempt_supercell):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(
        stol=stol,
        angle_tol=angle_tol,
        ltol=ltol,
        attempt_supercell=attempt_supercell,
    )


def compute_uniqueness(
    structures: List[Optional["Structure"]],
    stol: float = _MATCHER_STOL,
    angle_tol: float = _MATCHER_ANGLE_TOL,
    ltol: float = _MATCHER_LTOL,
    attempt_supercell: bool = _MATCHER_ATTEMPT_SUPERCELL,
    symmetric: bool = _MATCHER_SYMMETRIC,
) -> List[bool]:
    matcher = _make_matcher(stol, angle_tol, ltol, attempt_supercell)
    return _match_against(
        structures, defaultdict(list), matcher, symmetric=symmetric, dynamic=True
    )


def compute_novelty(
    structures: List[Optional["Structure"]],
    reference_structures: List["Structure"],
    stol: float = _MATCHER_STOL,
    angle_tol: float = _MATCHER_ANGLE_TOL,
    ltol: float = _MATCHER_LTOL,
    attempt_supercell: bool = _MATCHER_ATTEMPT_SUPERCELL,
    symmetric: bool = _MATCHER_SYMMETRIC,
) -> List[bool]:
    matcher = _make_matcher(stol, angle_tol, ltol, attempt_supercell)
    reference_by_comp = defaultdict(list)
    for ref in reference_structures:
        reference_by_comp[_composition_hash(ref)].append(ref)
    return _match_against(structures, reference_by_comp, matcher, symmetric=symmetric)


def compute_stability(
    energy_above_hull: List[Optional[float]],
    threshold: float = _STABILITY_THRESHOLD,
) -> List[bool]:
    return [
        (eah is not None and not np.isnan(eah) and eah < threshold)
        for eah in energy_above_hull
    ]


def compute_sun(
    stability: List[bool],
    uniqueness: List[bool],
    novelty: List[bool],
    structures: List[Optional["Structure"]],
    min_elements: int = 2,
) -> Dict[str, float]:
    n = len(stability)
    non_trivial = []
    for s in structures:
        if s is not None and len(set(s.composition)) >= min_elements:
            non_trivial.append(True)
        else:
            non_trivial.append(False)

    sun = [
        s and u and nv and nt
        for s, u, nv, nt in zip(stability, uniqueness, novelty, non_trivial)
    ]

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


class SUNMetric(StreamingMetricBase):
    """S.U.N. (Stability / Uniqueness / Novelty) metric for crystal generation.

    Novelty references the training split (``reference_file_path``), and
    stability requires convex-hull energies (``e_above_hull < 0``). Stability
    is only computed when ``energy_above_hull`` is passed explicitly
    (prerelaxation + hull reference not yet migrated); otherwise
    ``stability_rate`` reports 0.0 with a warning.

    Supports one-shot evaluation (``__call__`` over the full generated set)
    and streaming evaluation through the ``StreamingMetricBase`` hooks
    (``update_step`` / ``compute_epoch`` / ``reset``). The uniqueness pool is
    shared across batch boundaries, so both entry points compute the exact
    same global two-by-two semantics as the one-shot path.
    """

    def __init__(
        self,
        gt_file_path: Optional[str] = None,
        reference_file_path: Optional[str] = None,
        stol: float = _MATCHER_STOL,
        angle_tol: float = _MATCHER_ANGLE_TOL,
        ltol: float = _MATCHER_LTOL,
        stability_threshold: float = _STABILITY_THRESHOLD,
        attempt_supercell: bool = _MATCHER_ATTEMPT_SUPERCELL,
    ):
        self.gt_file_path = gt_file_path
        self.reference_file_path = reference_file_path or gt_file_path
        self.stol = stol
        self.angle_tol = angle_tol
        self.ltol = ltol
        self.stability_threshold = stability_threshold
        self.attempt_supercell = attempt_supercell
        self._reference_structures = None
        self.reset()

    # ---- streaming lifecycle (StreamingMetricBase) ----
    def reset(self):
        """Clear the per-evaluation uniqueness pool and accumulated flags."""
        self._unique_pool: Dict[str, List] = defaultdict(list)
        self._structures: List[Optional["Structure"]] = []
        self._unique: List[bool] = []
        self._novelty: List[bool] = []
        self._stability: List[bool] = []
        self._energy_above_hull: List[Optional[float]] = []
        self._stability_given = False
        self._warned_no_eah = False

    def update_step(self, *, result: Dict, batch, stage: str):
        """Accumulate one evaluation step's generated structures.

        ``result`` must be a dict (per ``StreamingMetricBase``) carrying the
        generated materials under ``"result"``/``"samples"``/``"structures"``
        (sample output), and optionally ``"energy_above_hull"`` for stability.
        """
        if stage not in ("eval", "sample"):
            return
        generated = (
            result.get("result") or result.get("samples") or result.get("structures")
        )
        if not generated:
            return
        self._ingest(generated, result.get("energy_above_hull"))

    def compute_epoch(self, *, stage: str) -> Dict[str, float]:
        """Finalize S.U.N. rates over everything accumulated so far."""
        if not self._structures:
            return {}
        return self._finalize_dict()

    # ---- one-shot evaluation (backwards compatible) ----
    def __call__(
        self,
        generated: Union[List[dict], List[str], "pd.DataFrame"],
        energy_above_hull: Optional[List[Optional[float]]] = None,
    ) -> Dict[str, float]:
        self.reset()
        self._ingest(generated, energy_above_hull)
        return self._finalize_dict()

    # ---- internals ----
    def _ingest(
        self,
        generated: Union[List[dict], List[str], "pd.DataFrame"],
        energy_above_hull: Optional[List[Optional[float]]],
    ):
        import pandas as pd

        if isinstance(generated, pd.DataFrame):
            if "cif" in generated.columns:
                generated = generated["cif"].tolist()
            elif "structure" in generated.columns:
                generated = generated["structure"].tolist()
            else:
                raise ValueError("DataFrame must have 'cif' or 'structure' column")

        if isinstance(generated, list) and len(generated) > 0:
            fmt = None
            if isinstance(generated[0], str):
                fmt = "cif_str"
            elif isinstance(generated[0], dict):
                fmt = "array"
            structures = _parse_structures(generated, fmt) if fmt else list(generated)
        else:
            raise ValueError("generated must be a non-empty list or DataFrame")

        n = len(structures)
        logger.info(f"Evaluating {n} generated structures")

        self._structures.extend(structures)
        self._unique.extend(self._match_unique(structures))

        self._load_reference()
        if self._reference_structures:
            self._novelty.extend(
                compute_novelty(
                    structures,
                    self._reference_structures,
                    stol=self.stol,
                    angle_tol=self.angle_tol,
                    ltol=self.ltol,
                    attempt_supercell=self.attempt_supercell,
                )
            )
        else:
            logger.warning("No reference set loaded; novelty defaults to True")
            self._novelty.extend([True] * n)

        if energy_above_hull is not None:
            self._stability.extend(
                compute_stability(energy_above_hull, threshold=self.stability_threshold)
            )
            self._energy_above_hull.extend(list(energy_above_hull))
            self._stability_given = True
        else:
            self._stability.extend([False] * n)

    def _match_unique(self, structures: List[Optional["Structure"]]) -> List[bool]:
        """Incremental uniqueness against the shared (cross-batch) pool.

        Semantically identical to ``compute_uniqueness`` (dynamic=True) over
        the concatenation of all ingested batches.
        """
        matcher = _make_matcher(
            self.stol, self.angle_tol, self.ltol, self.attempt_supercell
        )
        results = []
        for struct in structures:
            if struct is None:
                results.append(False)
                continue
            h = _composition_hash(struct)
            refs = self._unique_pool.get(h)
            if refs is None:
                self._unique_pool[h] = [struct]
                results.append(True)
                continue
            matched = any(matcher.fit(struct, ref) for ref in refs)
            results.append(not matched)
            if not matched:
                refs.append(struct)
        return results

    def _finalize_dict(self) -> Dict[str, float]:
        if not self._stability_given and not self._warned_no_eah:
            logger.warning(
                "No energy_above_hull provided; "
                "stability_rate will be 0.0 (prerelaxation + hull not migrated yet)."
            )
            self._warned_no_eah = True

        results = compute_sun(
            self._stability, self._unique, self._novelty, self._structures
        )

        if self._stability_given:
            metastable = compute_stability(
                self._energy_above_hull, threshold=_METASTABILITY_THRESHOLD
            )
            msun = compute_sun(
                metastable, self._unique, self._novelty, self._structures
            )
            results["metastable_rate"] = msun["stability_rate"]
            results["msun_rate"] = msun["sun_rate"]
            results["msun_count"] = msun["sun_count"]

        logger.info(
            f"S.U.N.: Stability={results['stability_rate']:.2f}%, "
            f"Uniqueness={results['uniqueness_rate']:.2f}%, "
            f"Novelty={results['novelty_rate']:.2f}%, "
            f"S.U.N.={results['sun_rate']:.2f}%"
        )
        return results

    def _load_reference(self):
        """Load the novelty reference set (the training split).

        Parsed structures are cached under ``DATASETS_HOME/miad/`` (keyed by a
        hash of the reference file path) so repeated evaluations skip the
        expensive CIF parsing without writing into the dataset directory.
        """
        import pickle

        import pandas as pd
        from pymatgen.core import Structure
        from tqdm import tqdm

        if self._reference_structures is not None:
            return
        if self.reference_file_path is None:
            return
        if not os.path.exists(self.reference_file_path):
            logger.warning(f"Reference file not found: {self.reference_file_path}")
            return

        cache_dir = os.path.join(download.DATASETS_HOME, "miad")
        os.makedirs(cache_dir, exist_ok=True)
        path_hash = hashlib.md5(
            os.path.abspath(self.reference_file_path).encode("utf-8")
        ).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"novelty_reference_{path_hash}.pkl")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached reference structures from {cache_path}")
            with open(cache_path, "rb") as f:
                self._reference_structures = pickle.load(f)
            logger.info(
                f"Loaded {len(self._reference_structures)} cached reference structures"
            )
            return

        df = pd.read_csv(self.reference_file_path)
        if "cif" not in df.columns:
            logger.warning("Reference CSV must have a 'cif' column")
            return
        self._reference_structures = []
        for cif_str in tqdm(df["cif"], desc="Loading reference structures"):
            try:
                self._reference_structures.append(
                    Structure.from_str(cif_str, fmt="cif")
                )
            except Exception as exc:
                logger.debug("Failed to parse reference cif: %s", exc)
        logger.info(f"Loaded {len(self._reference_structures)} reference structures")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self._reference_structures, f)
            logger.info(f"Cached reference structures to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache reference structures: {e}")
