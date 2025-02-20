from typing import Any

import numpy as np
import pandas as pd
from metrics.metric_utils import Crystal
from metrics.metric_utils import get_gt_crys_ori
from metrics.metric_utils import load_jsonline
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

# Warning: the smact package version is 2.5.5,
# different version may cause slight differences in accuracy.


class RecMetrics:
    # the reconstruct metrics class
    def __init__(self, gt_file_path=None, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

        self.gt_file_path = gt_file_path
        if gt_file_path is not None:
            csv = pd.read_csv(self.gt_file_path)
            self.gt_crys = p_map(get_gt_crys_ori, csv["cif"])
        else:
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
        for i in tqdm(range(len(pred_crys))):
            rms_dists.append(process_one(pred_crys[i], gt_crys[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(pred_crys)  # noqa
        mean_rms_dist = rms_dists[rms_dists != None].mean()  # noqa
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def __call__(self, file_path_or_data) -> Any:
        if isinstance(file_path_or_data, str):
            data = load_jsonline(file_path_or_data)
        else:
            data = file_path_or_data

        prediction = [d["prediction"] for d in data]
        pred_crys = p_map(lambda x: Crystal(x), prediction)

        if self.gt_crys is None:
            ground_truth = [d["ground_truth"] for d in data]
            gt_crys = p_map(lambda x: Crystal(x), ground_truth)
        else:
            gt_crys = self.gt_crys

        recon_metrics = self.get_match_rate_and_rms(pred_crys, gt_crys)
        return recon_metrics


if __name__ == "__main__":
    metircs = RecMetrics("structure_prediction/data/mp_20/test.csv")
    print(
        metircs(
            "structure_prediction/data/test_metric_data/diffcsp_mp_20_sample_1/output.jsonl"
        )
    )
