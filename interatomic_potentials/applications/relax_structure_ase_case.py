from __future__ import annotations

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))  # ruff: noqa

import argparse

from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.optimize import LBFGS
from ase.optimize import BFGSLineSearch
from ase.optimize import GPMin
from ase.optimize import LBFGSLineSearch
from ase.optimize import MDMin
from pymatgen.io.ase import AseAtomsAdaptor
from ppmat.calculators import PPMatCalculator
from ppmat.calculators import PPMatPredictor


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
        "--device",
        type=str,
        default="cpu",
        help="Device name, e.g., gpu, cpu.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize the predictor and load weights for inference
    predictor = PPMatPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    predictor.load_inference_model(ase_calc=True, device=args.device)

    # Initialize the ASE-compatible calculator
    calc = PPMatCalculator(predictor)

    # Build a bulk Cu structure
    atoms = bulk("Cu")

    # Convert the ASE Atoms object to pymatgen Structure 
    # and build the graph
    structure = AseAtomsAdaptor().get_structure(atoms)
    graph = predictor.graph_converter(structure)
    atoms.info["graph"] = graph

    # Set the calculator
    atoms.calc = calc

    # Initialize the optimizer
    # opt = LBFGS(atoms)
    opt = LBFGS(FrechetCellFilter(atoms))

    opt.run(fmax=0.05, steps=100)
