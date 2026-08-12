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

"""SGEQuiDiff generation quality metrics (validity / uniqueness /
novelty / coverage)."""

from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher

from ppmat.metrics.utils import Crystal
from ppmat.metrics.utils import compute_cov
from ppmat.metrics.utils import get_crys_from_cif
from ppmat.metrics.utils import get_novel_structures
from ppmat.metrics.utils import get_unique_structures


class SGEQuiDiffMetric:
    """Generation-quality metrics for SGEQuiDiff.

    Args:
        gt_file_path: Optional CSV with a ``"cif"`` column used as ground truth
            (used when ``gt_data`` is not passed to ``__call__``).
        struc_cutoff: Structural fingerprint distance cutoff for coverage.
        comp_cutoff: Composition fingerprint distance cutoff for coverage.
        n_structures: Optional cap on the number of generated structures scored.
    """

    def __init__(
        self,
        gt_file_path: Optional[str] = None,
        struc_cutoff: float = 0.4,
        comp_cutoff: float = 10.0,
        n_structures: Optional[int] = None,
    ):
        self.gt_file_path = gt_file_path
        self.matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
        self.struc_cutoff = struc_cutoff
        self.comp_cutoff = comp_cutoff
        self.n_structures = n_structures
        self._gt_crys: Optional[List[Crystal]] = None

    def _load_gt_crys(self) -> List[Crystal]:
        if self._gt_crys is None:
            assert self.gt_file_path, "gt_file_path is required when gt_data is None"
            csv = pd.read_csv(self.gt_file_path)
            self._gt_crys = [get_crys_from_cif(cif) for cif in csv["cif"].tolist()]
        return self._gt_crys

    def __call__(self, pred_data: Any, gt_data: Any = None) -> dict:
        """Score generated structures against ground truth.

        Args:
            pred_data: list of dicts (``num_atoms`` / ``atom_types`` /
                ``frac_coords`` / ``lengths`` / ``angles``).
            gt_data: optional list of dicts; falls back to ``gt_file_path``.
        """
        pred_crys = [Crystal(d) for d in pred_data]
        if self.n_structures is not None:
            pred_crys = pred_crys[: self.n_structures]

        if gt_data is not None:
            gt_crys = [Crystal(d) for d in gt_data]
        else:
            gt_crys = self._load_gt_crys()

        validity = float(np.mean([c.valid for c in pred_crys]))

        valid_structs = [c.structure for c in pred_crys if c.valid]
        n_valid = len(valid_structs)

        uniqueness = 0.0
        novelty = 0.0
        if n_valid > 0:
            unique, _ = get_unique_structures(valid_structs, self.matcher)
            uniqueness = len(unique) / n_valid

            gt_valid_structs = [c.structure for c in gt_crys if c.valid]
            novel, _ = get_novel_structures(
                valid_structs, gt_valid_structs, self.matcher
            )
            novelty = len(novel) / n_valid

        cov_metrics = compute_cov(
            pred_crys,
            gt_crys,
            self.struc_cutoff,
            self.comp_cutoff,
        )

        return {
            "validity": validity,
            "uniqueness": uniqueness,
            "novelty": novelty,
            **cov_metrics,
        }
