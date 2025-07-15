from __future__ import annotations

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))  # ruff: noqa

from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.predict import ASECalculator
from ppmat.predict import PPMatPredictor
from ppmat.predict import parse_args
from ppmat.utils import logger

# Option A: Load structures from file (default)
structures = None

# Option B: Manually generate a single structure using ASE
structures = bulk("Fe")
structures = bulk("Fe").repeat((4, 4, 4))

# Option C: Generate multiple structures using ASE
# structures = [bulk("Cu"), bulk("Al")]


def prepare_structures(args):
    global structures

    if args.file_path:
        if structures is None:
            logger.info(f"Loading structures from {args.file_path}.")
            _, structures = predictor.collect_structures(file_path=args.file_path)
        else:
            raise ValueError(
                "Both `file_path` and `structures` are provided. "
                "Please provide only one source of input structures."
            )
    else:
        if structures is not None:
            if not isinstance(structures, list):
                structures = [structures]
            structures = [
                AseAtomsAdaptor().get_structure(structure) for structure in structures
            ]
            logger.info(f"Using ASE provided structures (count: {len(structures)})")
        else:
            raise ValueError(
                "No `file_path` provided and no `structures` defined. "
                "Please specify input structures via `file_path` or ASE Atoms."
            )
    return structures


def run_relaxation(calc: ASECalculator, args, structures):
    logger.info("Relax structure.")
    calc.run_opt(
        structures=structures,
        file_path=args.file_path,
        save_path=args.save_path,
        optimizer=args.optimizer,
        filter=args.filter,
        fmax=args.fmax,
        steps=args.steps,
    )
    logger.info("All relaxations finished successfully.")


def run_md_simulation(calc: ASECalculator, args, structures):
    logger.info("Run MD simulation.")
    calc.run_md(
        structures=structures,
        file_path=args.file_path,
        save_path=args.save_path,
        temperature=args.temperature,
        timestep=args.timestep,
        steps=args.steps,
        interval=args.interval,
    )
    logger.info("All MD simulations finished successfully.")


if __name__ == "__main__":
    args = parse_args()
    logger.info("Arguments:")
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")

    # Initialize the predictor and load weights for inference
    predictor = PPMatPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    predictor.load_inference_model(ase_calc=args.ase_calc, device=args.device)

    # Load structures from file or use provided structures
    structures = prepare_structures(args)

    # Initialize the ASE calculator
    calc = ASECalculator(predictor)

    if args.command == "ase_opt":
        run_relaxation(calc, args, structures)
    elif args.command == "ase_md":
        run_md_simulation(calc, args, structures)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    # To convert the trajectory (.traj) file to XYZ format, run:
    # ase convert <trajectory_file>.traj <output_file>.xyz
