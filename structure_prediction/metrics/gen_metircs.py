import numpy as np
import paddle
import pandas as pd
import yaml
from dataset.cryst_dataset import TensorCrystDataset
from metrics.metric_utils import Crystal
from metrics.metric_utils import get_gt_crys_ori
from metrics.metric_utils import load_jsonline
from metrics.scaler import CompScalerMeans
from metrics.scaler import CompScalerStds
from metrics.scaler import scale_data
from models.dimenet import DimeNetPlusPlusWrap
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


def get_model(cfg, pretrained=None):
    # setup the architecture of MEGNet model
    model_cfg = cfg["model"]
    model_name = model_cfg.pop("__name__", None)
    if model_name == "DimeNetPlusPlusWrap":
        model = DimeNetPlusPlusWrap(**model_cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if pretrained is not None:
        model.set_state_dict(paddle.load(pretrained))
    return model


def collate_fn_graph(batch):
    new_batch = {}
    keys = [
        "edge_index",
        "y",
        "batch",
        "ptr",
        "frac_coords",
        "atom_types",
        "lengths",
        "angles",
        "to_jimages",
        "num_atoms",
        "num_bonds",
        "num_nodes",
        "prop",
        "anchor_index",
        "ops_inv",
        "ops",
        "spacegroup",
    ]
    for key in keys:
        if key not in batch[0]:
            continue
        if key in ["edge_index"]:
            cumulative_length = 0
            result_arrays_edge_index = []
            for x in batch:
                new_array = x[key] + cumulative_length
                result_arrays_edge_index.append(new_array)
                cumulative_length += x["num_atoms"]
            new_batch[key] = np.concatenate(result_arrays_edge_index, axis=1)
        elif key in [
            "frac_coords",
            "atom_types",
            "lengths",
            "angles",
            "to_jimages",
            "prop",
            "ops",
            "ops_inv",
            "spacegroup",
        ]:
            new_batch[key] = np.concatenate([x[key] for x in batch], axis=0)
        elif key in [
            "anchor_index",
        ]:
            cumulative_length = 0
            result_arrays_anchor_index = []
            for x in batch:
                new_array = x[key] + cumulative_length
                result_arrays_anchor_index.append(new_array)
                cumulative_length += len(x[key])
            new_batch[key] = np.concatenate(result_arrays_anchor_index, axis=0)
        elif key in [
            "num_atoms",
            "num_bonds",
        ]:
            new_batch[key] = np.array([x[key] for x in batch])
        elif key in ["num_nodes"]:
            new_batch[key] = np.array([x[key] for x in batch]).sum()

    graph_idxs = []
    for i in range(len(batch)):
        graph_idxs.extend([i] * batch[i]["num_atoms"])
    new_batch["batch"] = np.array(graph_idxs, dtype="int64")
    new_batch["num_graphs"] = len(batch)

    return new_batch


def prop_model_eval(cfg_path, weights_path, crystal_array_list):

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = get_model(cfg, weights_path)

    dataset = TensorCrystDataset(
        crystal_array_list,
        cfg["data"]["niggli"],
        cfg["data"]["primitive"],
        cfg["data"]["graph_method"],
        cfg["data"]["preprocess_workers"],
        cfg["data"]["lattice_scale_method"],
    )

    loader = paddle.io.DataLoader(
        dataset,
        batch_sampler=paddle.io.DistributedBatchSampler(
            dataset,
            batch_size=256,
            shuffle=False,
        ),
        collate_fn=collate_fn_graph,
        num_workers=0,
    )

    model.eval()
    means = -1.219802737236023
    stds = 1.0293837785720825
    all_preds = []

    for batch in loader:
        preds = model(batch)

        scaled_preds = preds * stds + means
        all_preds.append(scaled_preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(1)
    return all_preds.tolist()


class GenMetrics:
    def __init__(
        self,
        gt_file_path,
        n_samples=1,
        struc_cutoff=0.4,
        comp_cutoff=10,
        property_model_cfg_path=None,
        property_model_weights_path=None,
        seed=42,
    ):
        self.gt_file_path = gt_file_path
        if gt_file_path is not None:
            csv = pd.read_csv(self.gt_file_path)
            self.gt_crys = p_map(get_gt_crys_ori, csv["cif"])
        else:
            self.gt_crys = None

        self.n_samples = n_samples
        self.struc_cutoff = struc_cutoff
        self.comp_cutoff = comp_cutoff
        self.seed = seed

        if property_model_cfg_path is not None:
            assert property_model_weights_path is not None

        self.property_model_cfg_path = property_model_cfg_path
        self.property_model_weights_path = property_model_weights_path

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
        if self.property_model_cfg_path is not None:
            pred_props = prop_model_eval(
                self.property_model_cfg_path,
                self.property_model_weights_path,
                [c.dict for c in valid_samples],
            )
            gt_props = prop_model_eval(
                self.property_model_cfg_path,
                self.property_model_weights_path,
                [c.dict for c in gt_crys],
            )
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {"wdist_prop": wdist_prop}
        else:
            return {"wdist_prop": None}

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
        # prediction = prediction[:10]
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
    pred_file_path = "./data/test_metric_data/diffcsp_mp_20_gen/output_gen.jsonl"
    metric = GenMetrics(
        "./data/mp_20/test.csv",
        property_model_cfg_path="./data/prop_models/mp20/hparams_paddle.yaml",
        property_model_weights_path="./data/prop_models/mp20/epoch=839-step=89039_paddle.pdparams",
    )
    print(metric(pred_file_path))
