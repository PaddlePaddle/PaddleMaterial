from __future__ import annotations

import argparse
import os
import os.path as osp
import warnings

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import pandas as pd
from omegaconf import OmegaConf
from pymatgen.core import Structure

import interatomic_potentials.eager_comp_setting as eager_comp_setting
from ppmat.models import build_model
from ppmat.models.chgnet_v2.model import StructOptimizer
from ppmat.utils import logger
from ppmat.utils import misc

eager_comp_setting.setting_eager_mode(enable=True)

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

if dist.get_world_size() > 1:
    fleet.init(is_collective=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./interatomic_potentials/configs/chgnet_2d_lessatom20_st_v2.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--cif_dir",
        type=str,
        default="./data/2d_1k/1000",
        help="Path to cif directory",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default=None,  # "./data/2d_1k/1000/relax.csv",
        help="Path to label path",
    )

    args, dynamic_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

    if dist.get_rank() == 0:
        os.makedirs(config["Global"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(config, osp.join(config["Global"]["output_dir"], config_name))

    config = OmegaConf.to_container(config, resolve=True)
    logger.init_logger(
        log_file=osp.join(config["Global"]["output_dir"], "structure_optimization.log")
    )
    seed = config["Global"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # build model from config
    model_cfg = config["Model"]
    model = build_model(model_cfg)
    model.set_state_dict(paddle.load(config["Global"]["pretrained_model_path"]))
    logger.info(f"Loaded model from {config['Global']['pretrained_model_path']}")

    relaxer = StructOptimizer(model=model)

    cif_dir = args.cif_dir
    cif_files = [
        osp.join(cif_dir, f) for f in os.listdir(cif_dir) if f.endswith(".cif")
    ]
    logger.info(f"Loaded {len(cif_files)} structures from {cif_dir}")

    if args.label_path is not None:
        label = pd.read_csv(args.label_path)
        label_dict = {
            key: value
            for key, value in zip(label["cif"].tolist(), label["energy"].tolist())
        }
    else:
        label_dict = None

    cif_dir = os.path.join(config["Global"]["output_dir"], "relaxed_cifs")
    os.makedirs(cif_dir, exist_ok=True)

    preds = {
        "cif_name": [],
        "pred_energy_per_atom": [],
        "true_energy_per_atom": [],
    }
    diff = []

    for idx, cif_file in enumerate(cif_files):

        cif_name = os.path.basename(cif_file)

        cif_name = os.path.basename(cif_file)
        if label_dict is not None and cif_name not in label_dict:
            continue
        structure = Structure.from_file(cif_file)
        if max(structure.atomic_numbers) - 1 > 93:
            continue

        # Relax the structure
        logger.info(f"Start relaxing structure: {cif_name}")
        result = relaxer.relax(structure, verbose=False)

        relaxed_structure = result["final_structure"]
        final_energy = result["trajectory"].energies[-1]
        final_energy_per_atom = final_energy / len(structure)

        optimized_cif_file_path = os.path.join(cif_dir, cif_name)
        relaxed_structure.to(fmt="cif", filename=optimized_cif_file_path)
        logger.info(f"Relaxed structure saved to {optimized_cif_file_path}")

        preds["cif_name"].append(cif_name)
        preds["pred_energy_per_atom"].append(final_energy_per_atom)

        if label_dict is not None:
            true_energy_per_atom = label_dict.get(cif_name)
            if true_energy_per_atom is not None:
                true_energy_per_atom = true_energy_per_atom / len(structure)
        else:
            true_energy_per_atom = None
        preds["true_energy_per_atom"].append(true_energy_per_atom)

        if true_energy_per_atom is not None:
            diff.append(abs(final_energy_per_atom - true_energy_per_atom))

        msg = (
            f"idx: {idx}, cif name: {cif_name}, "
            f"predicted energy per atom: {final_energy_per_atom}"
        )
        if true_energy_per_atom is not None:
            msg += f", true energy per atom: {true_energy_per_atom}"
            msg += f", diff: {abs(final_energy_per_atom - true_energy_per_atom)}"
            msg += f", mae: {sum(diff)/len(diff)}"
        logger.info(msg)

        df = pd.DataFrame(preds)
        df.to_csv(
            osp.join(config["Global"]["output_dir"], "predictions.csv"), index=False
        )

        logger.info(
            f"Prediction results saved to "
            f"{osp.join(config['Global']['output_dir'], 'predictions.csv')}"
        )
