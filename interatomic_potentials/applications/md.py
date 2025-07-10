from __future__ import annotations

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))  # ruff: noqa

import argparse
import os.path as osp

from ase import units
from ase.io import Trajectory
from ase.io import write
from ase.md.langevin import Langevin
from ppmat.calculators import PPMatCalculator
from ppmat.calculators import PPMatPredictor

from ppmat.utils import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run MD simulation")
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
        "--device",
        type=str,
        default="cpu",
        help="Device name, e.g., gpu, cpu.",
    )
    parser.add_argument(
        "--temperature", type=float, default=300, help="Temperature in Kelvin."
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of MD steps.")
    parser.add_argument(
        "--interval", type=int, default=1, help="Interval to save trajectory."
    )

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("MD simulation workflow launched.")
    args = parse_args()

    # set save directory
    if args.save_path == None:
        base_dir = (
            args.cif_file_path
            if osp.isdir(args.cif_file_path)
            else osp.dirname(args.cif_file_path)
        )
        args.save_path = osp.join(base_dir, "results_md")
    os.makedirs(args.save_path, exist_ok=True)
    logger.info(f"Trajectories will be stored in: {args.save_path}")

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

    # load structures
    structures = calc.from_cif_file(args.cif_file_path)
    logger.info(f"Loaded {len(structures)} structures from {args.cif_file_path}")

    # run MD
    for idx, atoms in enumerate(structures):
        formula = atoms.get_chemical_formula()
        logger.info(
            f"[{idx+1}/{len(structures)}] Running MD: {formula}, "
            f"at T = {args.temperature} K"
        )
        atoms.calc = calc

        dyn = Langevin(
            atoms,
            timestep=0.1 * units.fs,
            temperature_K=args.temperature,
            friction=0.001 / units.fs,
        )

        traj_out = osp.join(args.save_path, f"{idx}_{formula}.traj")
        trajectory = Trajectory(traj_out, "w", atoms)
        dyn.attach(trajectory.write, interval=args.interval)
        dyn.run(steps=args.steps)
        logger.info(f"Saved MD trajectory: {traj_out}")
        xyz_out = osp.join(args.save_path, f"{idx}_{formula}_final.xyz")
        write(xyz_out, atoms)
        logger.info(f"Saved final frame to: {xyz_out}")
    logger.info(f"All MD simulations finished successfully.")

    # To convert the trajectory (.traj) file to XYZ format, run:
    # ase convert <trajectory_file>.traj <output_file>.xyz
