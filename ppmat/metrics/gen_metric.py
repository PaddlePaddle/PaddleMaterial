from collections import defaultdict

import numpy as np
import paddle
import pandas as pd
from omegaconf import OmegaConf
from p_tqdm import p_map
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets.transform import build_post_process
from ppmat.metrics.scaler import CompScalerMeans
from ppmat.metrics.scaler import CompScalerStds
from ppmat.metrics.scaler import scale_data
from ppmat.metrics.utils import Crystal
from ppmat.metrics.utils import get_crys_from_cif
from ppmat.models import build_model
from ppmat.utils import save_load

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


class PropPredictor:
    def __init__(self, config_file, pretrained_model_path):
        self.config_file = config_file
        self.pretrained_model_path = pretrained_model_path
        config = OmegaConf.load(self.config_file)
        self.config = OmegaConf.to_container(config, resolve=True)

        # build model from config
        model_cfg = self.config["Model"]
        self.model = build_model(model_cfg)
        save_load.load_pretrain(self.model, pretrained_model_path)
        self.model.eval()

        # build dataloader from config
        set_signal_handlers()
        self.predict_data_cfg = self.config["Dataset"]["predict"]
        self.predict_data_cfg["dataset"] = {
            "__name__": "TensorDataset",
            "niggli": False,
            "converter_cfg": {"method": "crystalnn"},
        }
        # build post processing from config
        post_process_cfg = config["PostProcess"]
        self.post_process_class = build_post_process(post_process_cfg)

    def __call__(self, crystal_list):

        """Eval program for one epoch.

        Args:
            epoch_id (int): Epoch id.
        """
        self.predict_data_cfg["dataset"]["crystal_list"] = crystal_list
        dataloader = build_dataloader(self.predict_data_cfg)

        all_preds = defaultdict(list)
        for _, batch_data in enumerate(dataloader):

            with paddle.no_grad():
                pred_data = self.model(batch_data)

            if self.post_process_class is not None:
                # since the label data may be not in batch_data, we need to pass it to
                # post_process_class
                pred_data, _ = self.post_process_class(pred_data)

            for key, value in pred_data.items():
                all_preds[key].append(value.detach().cpu().numpy())

        for key, value in all_preds.items():
            all_preds[key] = np.concatenate(all_preds[key], axis=0)
        return all_preds


class GenMetric:
    def __init__(
        self,
        gt_file_path,
        n_samples=[1000, 5000],
        struc_cutoff=0.4,
        comp_cutoff=10,
        property_model_cfg_path=None,
        property_model_weights_path=None,
        polar_decompose=False,
        seed=42,
    ):
        self.gt_file_path = gt_file_path
        self.gt_crys = None

        self.n_samples = n_samples
        self.struc_cutoff = struc_cutoff
        self.comp_cutoff = comp_cutoff
        self.polar_decompose = polar_decompose
        self.seed = seed

        if property_model_cfg_path is not None:
            assert property_model_weights_path is not None

        if property_model_cfg_path is None and property_model_weights_path is None:

            property_model_cfg_path = "./ppmat/metrics/prop_models/dimenet_mp20.yaml"
            property_model_weights_path = (
                "./ppmat/metrics/prop_models/epoch=839-step=89039_paddle.pdparams"
            )

        self.property_model_cfg_path = property_model_cfg_path
        self.property_model_weights_path = property_model_weights_path

        self.prop_predictor = PropPredictor(
            config_file=property_model_cfg_path,
            pretrained_model_path=property_model_weights_path,
        )

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
        pred_props = self.prop_predictor([c.dict for c in valid_samples])
        pred_props = pred_props["formation_energy_per_atom"].squeeze(1).tolist()
        gt_props = self.prop_predictor([c.dict for c in gt_crys])
        gt_props = gt_props["formation_energy_per_atom"].squeeze(1).tolist()
        wdist_prop = wasserstein_distance(pred_props, gt_props)
        return {"wdist_prop": wdist_prop}

    def get_coverage(self, pred_crys, gt_crys):
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            pred_crys,
            gt_crys,
            struc_cutoff=self.struc_cutoff,
            comp_cutoff=self.comp_cutoff,
        )
        return cov_metrics_dict

    def __call__(self, pred_data, gt_data=None):
        if self.seed is not None:
            np.random.seed(self.seed)

        pred_crys = p_map(lambda x: Crystal(x), pred_data, desc="Loading predictions")
        # the following line is equivalent to the above line, but it is slower,
        # it is used for debugging purposes
        # from tqdm import tqdm
        # pred_crys = []
        # for i in tqdm(range(len(pred_data))):
        #     pred_crys.append(Crystal(pred_data[i]))

        if gt_data is not None:
            gt_crys = p_map(
                lambda x: Crystal(x), gt_data, desc="Loading ground truth from data"
            )
        else:
            if self.gt_crys is None:
                # read the ground truth from csv file
                csv = pd.read_csv(self.gt_file_path)
                self.gt_crys = p_map(
                    get_crys_from_cif,
                    csv["cif"],
                    [self.polar_decompose] * len(csv["cif"]),
                    desc="Loading ground truth from CSV",
                )
            gt_crys = self.gt_crys

        metrics = {}
        metrics.update(self.get_validity(pred_crys))
        metrics.update(self.get_coverage(pred_crys, gt_crys))

        for n_sample in self.n_samples:
            metrics[f"n_sample_{n_sample}"] = {}
            valid_crys = [c for c in pred_crys if c.valid]
            if len(valid_crys) >= n_sample:
                sampled_indices = np.random.choice(
                    len(valid_crys), n_sample, replace=False
                )
                valid_samples = [valid_crys[i] for i in sampled_indices]
            else:
                raise Exception(
                    "Not enough valid crystals in the predicted set:"
                    f" {len(valid_crys)}/{n_sample}"
                )

            metrics[f"n_sample_{n_sample}"].update(
                self.get_density_wdist(valid_samples, gt_crys)
            )
            metrics[f"n_sample_{n_sample}"].update(
                self.get_prop_wdist(valid_samples, gt_crys)
            )
            metrics[f"n_sample_{n_sample}"].update(
                self.get_num_elem_wdist(valid_samples, gt_crys)
            )
        return metrics
