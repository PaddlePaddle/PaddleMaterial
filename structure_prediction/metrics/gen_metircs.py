import numpy as np
import pandas as pd
from metrics.scaler import CompScalerMeans
from metrics.scaler import CompScalerStds
from metrics.scaler import scale_data
from metrics.utils import Crystal
from metrics.utils import get_gt_crys_ori
from metrics.utils import load_jsonline
from p_tqdm import p_map
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

# Warning: the smact package version is 2.5.5,
# different version may cause slight differences in accuracy.


def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


def compute_cov(crys, gt_crys, struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    # comp_fps = CompScaler.transform(comp_fps)
    comp_fps = scale_data(comp_fps, CompScalerMeans, CompScalerStds)
    gt_comp_fps = scale_data(gt_comp_fps, CompScalerMeans, CompScalerStds)
    # gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(
        np.logical_and(
            struc_recall_dist <= struc_cutoff, comp_recall_dist <= comp_cutoff
        )
    )
    cov_precision = (
        np.sum(
            np.logical_and(
                struc_precision_dist <= struc_cutoff, comp_precision_dist <= comp_cutoff
            )
        )
        / num_gen_crystals
    )

    metrics_dict = {
        "cov_recall": cov_recall,
        "cov_precision": cov_precision,
        "amsd_recall": np.mean(struc_recall_dist),
        "amsd_precision": np.mean(struc_precision_dist),
        "amcd_recall": np.mean(comp_recall_dist),
        "amcd_precision": np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        "struc_recall_dist": struc_recall_dist.tolist(),
        "struc_precision_dist": struc_precision_dist.tolist(),
        "comp_recall_dist": comp_recall_dist.tolist(),
        "comp_precision_dist": comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict


class GenMetrics:
    def __init__(
        self, gt_file_path, n_samples=1000, seed=42, struc_cutoff=0.4, comp_cutoff=10
    ):
        self.gt_file_path = gt_file_path
        if gt_file_path is not None:
            csv = pd.read_csv(self.gt_file_path)
            self.gt_crys = p_map(get_gt_crys_ori, csv["cif"])
        else:
            self.gt_crys = None

        self.n_samples = n_samples
        self.seed = seed
        self.struc_cutoff = struc_cutoff
        self.comp_cutoff = comp_cutoff

    def get_validity(self, pred_crys):
        comp_valid = np.array([c.comp_valid for c in pred_crys]).mean()
        struct_valid = np.array([c.struct_valid for c in pred_crys]).mean()
        valid = np.array([c.valid for c in pred_crys]).mean()
        return {"comp_valid": comp_valid, "struct_valid": struct_valid, "valid": valid}

    def get_density_wdist(self, valid_samples, gt_crys):
        pred_densities = [c.structure.density for c in valid_samples]
        gt_densities = [c.structure.density for c in gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {"wdist_density": wdist_density}

    def get_num_elem_wdist(self, valid_samples, gt_crys):
        pred_nelems = [len(set(c.structure.species)) for c in valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {"wdist_num_elems": wdist_num_elems}

    def get_prop_wdist(self, valid_samples, gt_crys):
        # TODO: implement it
        return {"wdist_prop": None}
        # if self.eval_model_name is not None:
        #     pred_props = prop_model_eval(self.eval_model_name, [
        #                                  c.dict for c in self.valid_samples])
        #     gt_props = prop_model_eval(self.eval_model_name, [
        #                                c.dict for c in self.gt_crys])
        #     wdist_prop = wasserstein_distance(pred_props, gt_props)
        #     return {'wdist_prop': wdist_prop}
        # else:
        #     return {'wdist_prop': None}

    def get_coverage(self, pred_crys, gt_crys):
        # cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            pred_crys,
            gt_crys,
            struc_cutoff=self.struc_cutoff,
            comp_cutoff=self.comp_cutoff,
        )
        return cov_metrics_dict

    def __call__(self, file_path_or_data):
        if self.seed is not None:
            np.random.seed(self.seed)

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

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= self.n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), self.n_samples, replace=False
            )
            valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                "Not enough valid crystals in the predicted set:"
                f" {len(valid_crys)}/{self.n_samples}"
            )

        metrics = {}
        metrics.update(self.get_validity(pred_crys))
        metrics.update(self.get_density_wdist(valid_samples, gt_crys))
        metrics.update(self.get_prop_wdist(valid_samples, gt_crys))
        metrics.update(self.get_num_elem_wdist(valid_samples, gt_crys))
        metrics.update(self.get_coverage(pred_crys, gt_crys))
        return metrics


if __name__ == "__main__":
    pred_file_path = (
        "structure_prediction/data/test_metric_data/diffcsp_mp_20_gen/output_gen.jsonl"
    )
    metric = GenMetrics(
        "structure_prediction/data/mp_20/test.csv",
    )
    print(metric(pred_file_path))
