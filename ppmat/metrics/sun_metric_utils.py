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

from ppmat.datasets.build_structure import BuildStructure

logger = logging.getLogger(__name__)

_chgnet_model = None
_graph_converter = None


def _load_chgnet(model_name: str = "chgnet_mptrj"):
    global _chgnet_model, _graph_converter
    if _chgnet_model is not None:
        return _chgnet_model, _graph_converter
    from ppmat.models import build_model_from_name
    from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter
    _graph_converter = CHGNetGraphConverter(atom_graph_cutoff=6.0, bond_graph_cutoff=3.0)
    model, _ = build_model_from_name(model_name)
    model.eval()
    _chgnet_model = model
    return _chgnet_model, _graph_converter


def predict_energy(
    structures: List[Optional[Structure]],
    model_name: str = "chgnet_mptrj",
) -> List[Optional[float]]:
    model, converter = _load_chgnet(model_name)
    energies = []
    for struct in tqdm(structures, desc="Predicting energies (CHGNet)"):
        if struct is None:
            energies.append(None)
            continue
        try:
            graph = converter(struct)
            pred = model.predict(graph)
            e_per_atom = float(np.array(pred["energy_per_atom"]).flatten()[0])
            energies.append(e_per_atom)
        except Exception as e:
            logger.debug(f"Energy prediction failed: {e}")
            energies.append(None)
    return energies


def _composition_hash(structure: Structure) -> str:
    return str(sorted(list(structure.atomic_numbers)))


def _parse_structures(raw_list, format_str):
    results = []
    for item in raw_list:
        try:
            results.append(BuildStructure.build_one(item, format_str, niggli=False, canocial=False))
        except Exception:
            results.append(None)
    return results


def _match_against(
    structures: List[Optional[Structure]],
    reference_by_comp: dict,
    matcher: StructureMatcher,
    symmetric: bool = True,
    dynamic: bool = False,
) -> List[bool]:
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


def compute_uniqueness(
    structures: List[Optional[Structure]],
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
    attempt_supercell: bool = True,
    symmetric: bool = True,
) -> List[bool]:
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol, attempt_supercell=attempt_supercell)
    return _match_against(structures, defaultdict(list), matcher, symmetric=symmetric, dynamic=True)


def compute_novelty(
    structures: List[Optional[Structure]],
    reference_structures: List[Structure],
    stol: float = 0.5,
    angle_tol: float = 10.0,
    ltol: float = 0.3,
    attempt_supercell: bool = True,
    symmetric: bool = True,
) -> List[bool]:
    matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol, attempt_supercell=attempt_supercell)
    reference_by_comp = defaultdict(list)
    for ref in reference_structures:
        reference_by_comp[_composition_hash(ref)].append(ref)
    return _match_against(structures, reference_by_comp, matcher, symmetric=symmetric)


def compute_stability(
    energy_above_hull: List[Optional[float]],
    threshold: float = 0.0,
) -> List[bool]:
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
    n = len(stability)
    non_trivial = []
    for s in structures:
        if s is not None and len(set(s.composition)) >= min_elements:
            non_trivial.append(True)
        else:
            non_trivial.append(False)

    sun = [s and u and nv and nt for s, u, nv, nt in zip(stability, uniqueness, novelty, non_trivial)]

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
    def __init__(
        self,
        gt_file_path: Optional[str] = None,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        ltol: float = 0.3,
        stability_threshold: float = 0.0,
        attempt_supercell: bool = True,
        use_chgnet: bool = False,
    ):
        self.gt_file_path = gt_file_path
        self.stol = stol
        self.angle_tol = angle_tol
        self.ltol = ltol
        self.stability_threshold = stability_threshold
        self.attempt_supercell = attempt_supercell
        self.use_chgnet = use_chgnet
        self._reference_structures = None

    def _load_reference(self):
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
                self._reference_structures.append(BuildStructure.build_one(cif_str, "cif_str", niggli=False, canocial=False))
            except Exception:
                pass
        logger.info(f"Loaded {len(self._reference_structures)} reference structures")

    def __call__(
        self,
        generated: Union[List[dict], List[str], pd.DataFrame],
        energy_above_hull: Optional[List[Optional[float]]] = None,
    ) -> Dict[str, float]:
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
            structures = _parse_structures(generated, fmt) if fmt else generated
        else:
            raise ValueError("generated must be a non-empty list or DataFrame")

        n = len(structures)
        logger.info(f"Evaluating {n} generated structures")

        uniqueness = compute_uniqueness(
            structures, stol=self.stol, angle_tol=self.angle_tol, ltol=self.ltol,
            attempt_supercell=self.attempt_supercell,
        )

        novelty_list = [True] * n
        self._load_reference()
        if self._reference_structures:
            novelty_list = compute_novelty(
                structures, self._reference_structures,
                stol=self.stol, angle_tol=self.angle_tol, ltol=self.ltol,
                attempt_supercell=self.attempt_supercell,
            )

        if energy_above_hull is not None:
            stability = compute_stability(energy_above_hull, threshold=self.stability_threshold)
            energies = None
        elif self.use_chgnet:
            energies = predict_energy(structures)
            stability = [(e is not None and e < self.stability_threshold) for e in energies]
        else:
            stability = [False] * n
            energies = None
            logger.warning("No energy_above_hull and use_chgnet=False, stability will be 0%")

        results = compute_sun(stability, uniqueness, novelty_list, structures)

        if energies is not None:
            valid = [e for e in energies if e is not None]
            if valid:
                results["avg_energy_per_atom"] = float(np.mean(valid))

        if energy_above_hull is not None:
            meta = compute_stability(energy_above_hull, threshold=0.1)
            msun = compute_sun(meta, uniqueness, novelty_list, structures)
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
