# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Optional
from zipfile import ZipFile

import numpy as np
try:
    import paddle
except ModuleNotFoundError:
    paddle = None
try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    OmegaConf = None
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

_IMPORT_ERROR = None
try:
    from ppmat.datasets import build_dataloader
    from ppmat.datasets.build_structure import BuildStructure
    from ppmat.datasets.transform import build_post_transforms
    from ppmat.metrics import build_metric
    from ppmat.models import build_model
    from ppmat.models import build_model_from_name
    from ppmat.utils import logger
    from ppmat.utils import save_load
except Exception as exc:  # noqa: BLE001
    _IMPORT_ERROR = exc

    class _FallbackLogger:
        @staticmethod
        def info(msg):
            print(msg)

        @staticmethod
        def warning(msg):
            print(msg)

    logger = _FallbackLogger()
    build_dataloader = None
    BuildStructure = None
    build_post_transforms = None
    build_metric = None
    build_model = None
    build_model_from_name = None
    save_load = None


class StructureSampler:
    """Structure Sampler.

    This class provides an interface for sampling structures using pre-trained deep
    learning models. Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional): Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry. When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional): Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional): Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.
        checkpoint_path (Optional[str], optional): Path to model checkpoint file
            (.pdparams) for custom models. Required when not using predefined
            `model_name`. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        if OmegaConf is None:
            raise ImportError(
                "OmegaConf is required for sampling. Please install 'omegaconf' or "
                "run with --mode compute_metric_SUN to use evaluation only."
            )
        if paddle is None:
            raise ImportError(
                "PaddlePaddle is required for sampling. Please install 'paddlepaddle' "
                "or run with --mode compute_metric_SUN to use evaluation only."
            )
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Failed to import PaddleMaterials sampling dependencies."
            ) from _IMPORT_ERROR
        # if model_name is not None, then config_path and checkpoint_path must be
        # provided
        if model_name is None:
            assert (
                config_path is not None and checkpoint_path is not None
            ), "config_path and checkpoint_path must be provided when model_name is "
            "None."

            logger.info(f"Loading model from {config_path} and {checkpoint_path}.")

            config = OmegaConf.load(config_path)
            config = OmegaConf.to_container(config, resolve=True)

            model_config = config.get("Model", None)
            assert model_config is not None, "Model config must be provided."
            model = build_model(model_config)
            save_load.load_pretrain(model, checkpoint_path)

        else:
            logger.info("Since model_name is given, downloading it...")
            model, config = build_model_from_name(model_name, weights_name)

        self.model = model
        self.config = config

        self.model.eval()

        # sample config
        sample_config = config.get("Sample", None)
        self.sample_config = sample_config

        self.post_transforms_cfg = self.sample_config.get("post_transforms", None)
        if self.post_transforms_cfg is not None:
            self.post_transforms = build_post_transforms(self.post_transforms_cfg)
        else:
            self.post_transforms = None

    def post_process(self, data):
        if self.post_transforms is None:
            return data
        return self.post_transforms(data)

    def sample(self, data, sample_params=None):
        if sample_params is None:
            sample_params = {}
        assert isinstance(sample_params, dict), "sample_params must be a dict or None."
        pred_data = self.model.sample(data, **sample_params)
        pred_data = self.post_process(pred_data)
        return pred_data

    def sample_by_dataloader(
        self,
        save_path=None,
    ):
        dataset_cfg = self.sample_config["data"]
        data_loader = build_dataloader(dataset_cfg)

        build_structure_cfg = self.sample_config["build_structure_cfg"]
        structure_converter = BuildStructure(**build_structure_cfg)

        logger.info(f"Total iterations: {len(data_loader)}")
        logger.info("Start sampling process...\n")

        total_results = []
        for iter_id, batch_data in enumerate(data_loader):
            pred_data = self.model.sample(batch_data)
            structures = structure_converter(pred_data["result"])
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                for i, structure in enumerate(structures):
                    formula = structure.formula.replace(" ", "-")
                    tar_file = os.path.join(
                        save_path, f"{formula}_{iter_id + 1}_{i + 1}.cif"
                    )
                    if structure is not None:
                        writer = CifWriter(structure)
                        writer.write_file(tar_file)
                    else:
                        logger.info(
                            f"No structure generated for iteration {iter_id}, index {i}"
                        )
            total_results.extend(pred_data["result"])
        return total_results

    def sample_by_num_atoms(self, num_atoms, save_path=None, sample_params=None):
        assert isinstance(num_atoms, int), "num_atoms must be an integer."
        data = {
            "structure_array": {
                "num_atoms": paddle.to_tensor(np.array([num_atoms]).astype("int64")),
            }
        }

        result = self.sample(data, sample_params=sample_params)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Save results to {save_path}")
            build_structure_cfg = self.sample_config["build_structure_cfg"]
            structure_converter = BuildStructure(**build_structure_cfg)
            structures = structure_converter(result["result"])
            for i, structure in enumerate(structures):
                formula = structure.formula.replace(" ", "-")
                tar_file = os.path.join(save_path, f"{formula}_{i + 1}.cif")
                if structure is not None:
                    writer = CifWriter(structure)
                    writer.write_file(tar_file)
                else:
                    logger.info(f"No structure generated for index {i}")

        return result

    def sample_by_chemical_formula(
        self, chemical_formula, save_path=None, sample_params=None
    ):
        assert isinstance(chemical_formula, str), "chemical_formula must be a string."
        composition = Composition(chemical_formula)
        atom_types = []
        for elem, num in composition.items():
            atom_types.extend([elem.Z] * int(num))
        atom_types = np.array(atom_types).astype("int64")

        data = {
            "structure_array": {
                "atom_types": paddle.to_tensor(atom_types),
                "num_atoms": paddle.to_tensor(
                    np.array([atom_types.shape[0]]).astype("int64")
                ),
            }
        }
        result = self.sample(data, sample_params=sample_params)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Save results to {save_path}")
            build_structure_cfg = self.sample_config["build_structure_cfg"]
            structure_converter = BuildStructure(**build_structure_cfg)
            structures = structure_converter(result["result"])
            for i, structure in enumerate(structures):
                formula = structure.formula.replace(" ", "-")
                tar_file = os.path.join(save_path, f"{formula}_{i + 1}.cif")
                if structure is not None:
                    writer = CifWriter(structure)
                    writer.write_file(tar_file)
                else:
                    logger.info(f"No structure generated for index {i}")

        return result

    def sample_by_condition(self, composition, save_path=None, sample_params=None):
        # todo: implement this function
        pass

    def compute_metric(
        self,
        save_path=None,
    ):
        metrics_cfg = self.sample_config.get("metrics")
        assert metrics_cfg is not None, "metrics config must be provided."
        metrics_fn = build_metric(metrics_cfg)

        total_results = self.sample_by_dataloader(save_path)

        metric = metrics_fn(total_results)
        return metric

def _extract_structures_from_folder(dirname: str) -> list[Structure]:
    structures: list[Structure] = []
    if not os.path.isdir(dirname):
        raise ValueError(f"Directory {dirname} does not exist.")
    for filename in os.listdir(dirname):
        full_path = os.path.join(dirname, filename)
        if filename.endswith(".cif"):
            try:
                structures.append(Structure.from_file(full_path))
            except ValueError as exc:
                logger.warning(f"Failed to read {filename} as a CIF file: {exc}")
        elif filename.endswith(".extxyz") or filename.endswith(".xyz"):
            try:
                import ase.io
                from pymatgen.io.ase import AseAtomsAdaptor
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Reading .xyz/.extxyz requires the 'ase' package. Please install "
                    "it or convert files to CIF."
                ) from exc
            ase_atoms = ase.io.read(full_path, 0)
            structures.append(AseAtomsAdaptor.get_structure(ase_atoms))
    return structures


def _load_structures_local(input_path: Path) -> list[Structure]:
    """Minimal loader for structures supporting dir, .zip, .xyz/.extxyz."""
    if input_path.suffix in {".xyz", ".extxyz"}:
        try:
            import ase.io
            from pymatgen.io.ase import AseAtomsAdaptor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading .xyz/.extxyz requires the 'ase' package. Please install it "
                "or convert files to CIF."
            ) from exc
        ase_atoms = ase.io.read(input_path, ":")
        return [AseAtomsAdaptor.get_structure(x) for x in ase_atoms]
    if input_path.suffix == ".zip":
        with TemporaryDirectory() as tmpdirname:
            with ZipFile(input_path, "r") as zip_obj:
                zip_obj.extractall(tmpdirname)
            return _extract_structures_from_folder(tmpdirname)
    if input_path.is_dir():
        return _extract_structures_from_folder(str(input_path))
    raise ValueError(f"Invalid input path {input_path}")


def compute_metric_SUN(
    structures_path: str,
    relaxed_structures_path: Optional[str] = None,
    relax: bool = False,
    energies_path: Optional[str] = None,
    structure_matcher: Literal["ordered", "disordered"] = "disordered",
    save_as: Optional[str] = None,
    metrics_cfg_path: Optional[str] = None,
):
    """Compute evaluation metrics with element filtering for SUN experiments."""
    if structures_path is None:
        raise ValueError("structures_path must be provided.")
    if relax:
        logger.warning("Relaxation inside compute_metric_SUN is not supported; ignoring.")

    structures = _load_structures_local(Path(structures_path))
    relaxed_structures = None
    if relaxed_structures_path is not None:
        relaxed_structures = _load_structures_local(Path(relaxed_structures_path))
    energies = np.load(energies_path) if energies_path else None
    matcher = StructureMatcher(
        stol=0.5,
        angle_tol=5,
        ltol=0.2,
        attempt_supercell=False,
        primitive_cell=False,
        scale=False,
    )

    reference_elements = {
        "Sc",
        "F",
        "Pd",
        "Ti",
        "Nd",
        "P",
        "Ca",
        "Ru",
        "Sn",
        "Sm",
        "As",
        "O",
        "Be",
        "Au",
        "Cd",
        "Pt",
        "Bi",
        "Y",
        "Si",
        "Se",
        "Cu",
        "Sb",
        "In",
        "Br",
        "Hf",
        "I",
        "Ir",
        "La",
        "Ba",
        "Er",
        "Lu",
        "W",
        "Mo",
        "Li",
        "Ge",
        "Pb",
        "Hg",
        "Tl",
        "Ho",
        "Ta",
        "Co",
        "Ga",
        "Nb",
        "Fe",
        "Mg",
        "B",
        "N",
        "Cr",
        "Sr",
        "Rh",
        "Yb",
        "Ce",
        "Ni",
        "Re",
        "V",
        "Os",
        "H",
        "Rb",
        "Pr",
        "Al",
        "Eu",
        "Cl",
        "Gd",
        "S",
        "Ag",
        "Mn",
        "Na",
        "K",
        "Zn",
        "Cs",
        "C",
        "Te",
        "Tb",
        "Dy",
        "Tm",
        "Zr",
    }

    filtered_structures: list[Structure] = []
    kept_indices: list[int] = []
    for idx, structure in enumerate(structures):
        if all(site.specie.symbol in reference_elements for site in structure):
            filtered_structures.append(structure)
            kept_indices.append(idx)

    logger.info(f"{len(structures)} -> {len(filtered_structures)}")
    n_failed_jobs = len(structures) - len(filtered_structures)
    structures = filtered_structures
    if energies is not None:
        energies = [float(energies[idx]) for idx in kept_indices]
    if relaxed_structures is not None:
        if len(relaxed_structures) != len(structures):
            logger.warning(
                "relaxed_structures count does not match filtered structures; "
                "truncating to the shorter length."
            )
        relaxed_structures = relaxed_structures[: len(structures)]

    # base metrics following compute_metric pattern
    custom_metrics = None
    if metrics_cfg_path is not None:
        if OmegaConf is None or build_metric is None:
            logger.warning(
                "metrics_cfg_path provided but OmegaConf/build_metric unavailable; "
                "skipping custom metric computation."
            )
        else:
            try:
                cfg = OmegaConf.to_container(OmegaConf.load(metrics_cfg_path), resolve=True)
                metrics_fn = build_metric(cfg)
                custom_metrics = metrics_fn(structures)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to run custom metrics: {exc}")

    unique_structures: list[Structure] = []
    for structure in structures:
        if not any(matcher.fit(structure, uniq) for uniq in unique_structures):
            unique_structures.append(structure)

    rms_values: list[float] = []
    if relaxed_structures is not None:
        for pred, ref in zip(structures, relaxed_structures):
            try:
                rms_dist = matcher.get_rms_dist(pred, ref)
                if rms_dist is None:
                    continue
                if isinstance(rms_dist, (list, tuple, np.ndarray)):
                    rms_dist = rms_dist[0]
                rms_values.append(float(rms_dist))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to compute RMS distance: {exc}")

    metrics = {
        "total_structures": len(structures) + n_failed_jobs,
        "filtered_structures": len(structures),
        "failed_jobs": n_failed_jobs,
        "unique_count": len(unique_structures),
        "unique_fraction": float(len(unique_structures) / len(structures))
        if structures
        else 0.0,
    }

    if rms_values:
        metrics["rms_mean_relaxed"] = float(np.mean(rms_values))
        metrics["rms_match_rate_relaxed"] = float(len(rms_values) / len(structures))
    else:
        metrics["rms_mean_relaxed"] = None
        metrics["rms_match_rate_relaxed"] = 0.0

    if energies is not None:
        metrics["energies_count"] = len(energies)
        metrics["energies_mean"] = float(np.mean(energies)) if len(energies) else None
    if custom_metrics is not None:
        metrics.update(custom_metrics)

    if save_as is not None:
        save_path = Path(save_as)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    logger.info(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()

    argparse.add_argument("--model_name", type=str, default=None)
    argparse.add_argument(
        "--weights_name",
        type=str,
        default=None,
        help="Weights name, e.g., best.pdparams, latest.pdparams.",
    )
    argparse.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )
    argparse.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file.",
    )
    argparse.add_argument("--save_path", type=str, default="results")
    argparse.add_argument("--chemical_formula", type=str, default="LiMnO2")
    argparse.add_argument("--num_atoms", type=int, default=4)
    argparse.add_argument(
        "--mode",
        type=str,
        choices=[
            "by_chemical_formula",
            "by_num_atoms",
            "by_dataloader",
            "compute_metric",
            "compute_metric_SUN",
        ],
        default="by_chemical_formula",
    )
    argparse.add_argument(
        "--structures_path",
        type=str,
        default=None,
        help="Path to generated structures for compute_metric_SUN.",
    )
    argparse.add_argument(
        "--relaxed_structures_path",
        type=str,
        default=None,
        help="Optional path to relaxed structures for compute_metric_SUN.",
    )
    argparse.add_argument(
        "--relax",
        action="store_true",
        help="Relax structures before evaluation in compute_metric_SUN (currently ignored).",
    )
    argparse.add_argument(
        "--energies_path",
        type=str,
        default=None,
        help="Path to energies array for compute_metric_SUN.",
    )
    argparse.add_argument(
        "--structure_matcher",
        type=str,
        choices=["ordered", "disordered"],
        default="disordered",
        help="Structure matcher type for compute_metric_SUN.",
    )
    argparse.add_argument(
        "--save_as",
        type=str,
        default=None,
        help="Optional save path for computed metrics.",
    )
    argparse.add_argument(
        "--metrics_cfg_path",
        type=str,
        default=None,
        help="Optional metrics config path; will be built with build_metric similar to compute_metric.",
    )

    args = argparse.parse_args()

    if args.mode == "compute_metric_SUN":
        metrics = compute_metric_SUN(
            structures_path=args.structures_path,
            relaxed_structures_path=args.relaxed_structures_path,
            relax=args.relax,
            energies_path=args.energies_path,
            structure_matcher=args.structure_matcher,
            save_as=args.save_as,
        )
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value}")
        sys.exit(0)

    sampler = StructureSampler(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    if args.mode == "compute_metric":
        metric_result = sampler.compute_metric(save_path=args.save_path)
        for metric_name, metric_value in metric_result.items():
            logger.info(f"{metric_name}: {metric_value}")
    elif args.mode == "by_chemical_formula":
        result = sampler.sample_by_chemical_formula(
            chemical_formula=args.chemical_formula,
            save_path=args.save_path,
        )
    elif args.mode == "by_num_atoms":
        result = sampler.sample_by_num_atoms(
            num_atoms=args.num_atoms,
            save_path=args.save_path,
        )
    elif args.mode == "by_dataloader":
        result = sampler.sample_by_dataloader(
            save_path=args.save_path,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
