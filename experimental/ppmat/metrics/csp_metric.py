# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

from ppmat.metrics.utils import Crystal
from ppmat.metrics.utils import get_crys_from_cif

# Warning: the smact package version is 2.5.5,
# different version may cause slight differences in accuracy.


class CSPMetric:
    # the crystal structure prediction metrics class
    def __init__(self, gt_file_path, stol=0.5, angle_tol=10, ltol=0.3):

        assert gt_file_path.endswith(".csv"), "gt_file_path should be a CSV file"
        self.gt_file_path = gt_file_path
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.gt_crys = None

    def get_match_rate_and_rms(self, pred_crys, gt_crys):
        assert len(pred_crys) == len(gt_crys)

        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        validity = [(c1.valid and c2.valid) for c1, c2 in zip(pred_crys, gt_crys)]
        rms_dists = []
        for i in tqdm(range(len(pred_crys)), desc="Computing RMS distance"):
            rms_dists.append(process_one(pred_crys[i], gt_crys[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(pred_crys)  # noqa
        mean_rms_dist = rms_dists[rms_dists != None].mean()  # noqa
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def __call__(self, pred_data, gt_data=None) -> Any:

        # pred_crys = p_map(lambda x: Crystal(x), pred_data, desc="Loading predictions")
        # the following line is equivalent to the above line, but it is slower,
        # it is used for debugging purposes
        pred_crys = []
        for i in tqdm(range(len(pred_data))):
            pred_crys.append(Crystal(pred_data[i]))

        if gt_data is not None:
            gt_crys = p_map(
                lambda x: Crystal(x), gt_data, desc="Loading ground truth from data"
            )
        else:
            if self.gt_crys is None:
                # read the ground truth from csv file
                csv = pd.read_csv(self.gt_file_path)
                self.gt_crys = p_map(
                    get_crys_from_cif, csv["cif"], desc="Loading ground truth from CSV"
                )
            gt_crys = self.gt_crys

        metrics = self.get_match_rate_and_rms(pred_crys, gt_crys)
        return metrics
