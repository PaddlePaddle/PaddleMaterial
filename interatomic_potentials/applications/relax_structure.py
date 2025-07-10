from __future__ import annotations

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))  # ruff: noqa

import argparse
import os.path as osp

from ase import filters
from ase.io import write
from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.optimize import LBFGS
from ase.optimize import BFGSLineSearch
from ase.optimize import GPMin
from ase.optimize import LBFGSLineSearch
from ase.optimize import MDMin
from ppmat.calculators import PPMatCalculator
from ppmat.calculators import PPMatPredictor

from ppmat.utils import logger

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "MDMin": MDMin,
    "GPMin": GPMin,
    "LBFGSLineSearch": LBFGSLineSearch,
    "BFGSLineSearch": BFGSLineSearch,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Relax crystal structures")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name.",
    )
    parser.add_argument(
        "--weights_name",
        type=str,
        default=None,
        help="Weights name, e.g., best.pdparams, latest.pdparams.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--cif_file_path",
        type=str,
        default="interatomic_potentials/example_data/cifs",
        help="Path to the CIF file whose material properties you want to predict.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the prediction result.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="LBFGS",
        choices=list(OPTIMIZERS.keys()),
        help="Optimizer name.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="FrechetCellFilter",
        choices=[
            "none",
            "Filter",
            "StrainFilter",
            "UnitCellFilter",
            "FrechetCellFilter",
            "ExpCellFilter",
        ],
        help="Filter name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device name, e.g., gpu, cpu.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting relaxation...")
    args = parse_args()

    # set save directory
    if args.save_path == None:
        base_dir = (
            args.cif_file_path
            if osp.isdir(args.cif_file_path)
            else osp.dirname(args.cif_file_path)
        )
        args.save_path = osp.join(base_dir, "results_opt")
    os.makedirs(args.save_path, exist_ok=True)
    logger.info(f"Saved relaxed structure in directory: {args.save_path}")

    # predict
    predictor = PPMatPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    predictor.load_inference_model(ase_calc=True, device=args.device)

    # ase calculator
    calc = PPMatCalculator(predictor)

    # choose filter / optimizer
    ase_optimizer_cls = OPTIMIZERS[args.optimizer]
    logger.info(f"Using optimizer: {args.optimizer}")
    if args.filter == "none":
        ase_filter_cls = None
        logger.info("No filter specified. Will relax atomic positions only.")
    else:
        ase_filter_cls = getattr(filters, args.filter)
        logger.info(f"Using filter: {args.filter}")

    # load structures
    structures = calc.from_cif_file(args.cif_file_path)
    logger.info(f"Loaded {len(structures)} structures from {args.cif_file_path}")

    # relax
    for idx, structure in enumerate(structures):
        formula = structure.get_chemical_formula()
        logger.info(f"[{idx+1}/{len(structures)}] Relaxing structure: {formula}")
        structure.calc = calc
        opt = (
            ase_optimizer_cls(ase_filter_cls(structure))
            if ase_filter_cls
            else ase_optimizer_cls(structure)
        )
        opt.run(fmax=0.05, steps=100)

        outfile = osp.join(args.save_path, f"{formula}_{idx}.xyz")
        write(outfile, structure)
        logger.info(f"Relaxed structure saved: {outfile}")
    logger.info(f"Finished all relaxations.")
