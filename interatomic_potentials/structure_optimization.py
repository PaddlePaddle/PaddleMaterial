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
from ppmat.models.chgnet.model import StructOptimizer
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
        default="./interatomic_potentials/configs/chgnet_2d_lessatom20.yaml",
        help="Path to config file",
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

    relaxer = StructOptimizer(model=model)

    cif_dir = "./data/2d_1k/1000"
    cif_files = [
        osp.join(cif_dir, f) for f in os.listdir(cif_dir) if f.endswith(".cif")
    ]

    print(len(cif_files))

    label = pd.read_csv("./data/2d_1k/1000/relax.csv")
    label_dict = {
        key: value
        for key, value in zip(label["cif"].tolist(), label["energy"].tolist())
    }

    preds = {}
    diff = []
    index = 0
    for idx, cif_file in enumerate(cif_files):

        base_name = os.path.basename(cif_file)
        if base_name not in label_dict:
            continue

        structure = Structure.from_file(cif_file)

        if max(structure.atomic_numbers) - 1 > 93:
            continue
        # Relax the structure
        result = relaxer.relax(structure, verbose=False)

        # print("Relaxed structure:\n")
        # print(result["final_structure"])

        # print(result["trajectory"].energies)
        preds[os.path.basename(cif_file)] = result["trajectory"].energies[-1]

        print(
            "preds and labels: ",
            result["trajectory"].energies[-1],
            label_dict[os.path.basename(cif_file)],
            structure.n_elems,
            len(structure.sites),
        )

        diff.append(
            abs(
                result["trajectory"].energies[-1]
                - label_dict[os.path.basename(cif_file)]
            )
        )
        logger.info(f"idx: {idx}, mae: {sum(diff)/len(diff)}")
