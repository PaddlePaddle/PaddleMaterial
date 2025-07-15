from __future__ import annotations

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))  # ruff: noqa
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))  # ruff: noqa

from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

from ppmat.predict import PropertyPredictor
from ppmat.predict import parse_args
from ppmat.utils import logger

# Option A: Load structures from file (default behavior if `--file_path` is set)
structures = None

# Option B: Manually generate a single structure using ASE
# structures = bulk("Fe")
# structures = bulk("Fe").repeat((4, 4, 4))

# Option C: Generate multiple structures using ASE
structures = [bulk("Cu"), bulk("Al")]


def prepare_structures(args, predictor):
    global structures

    if args.file_path:
        if structures is None:
            logger.info(f"Loading structures from {args.file_path}.")
            files, structures = predictor.collect_structures(file_path=args.file_path)
        else:
            raise ValueError(
                "Both `file_path` and `structures` are provided. "
                "Please provide only one source of input structures."
            )
    else:
        if structures is not None:
            if not isinstance(structures, list):
                structures = [structures]
            # Convert ASE Atoms to pymatgen Structure
            structures_list, files = [], []
            for i, structure in enumerate(structures):
                _structure = AseAtomsAdaptor().get_structure(structure)
                _formula = structure.get_chemical_formula()  # Get chemical formula
                structures_list.append(_structure)
                files.append(f"structure_{i}_{_formula}")
            structures = structures_list  # Update structures
            logger.info(f"Using ASE provided structures (count: {len(structures)})")
        else:
            raise ValueError(
                "No `file_path` provided and no `structures` defined. "
                "Please specify input structures via `file_path` or ASE Atoms."
            )
    return files, structures


if __name__ == "__main__":
    logger.info("[PPMaterial] Start property prediction.")
    args = parse_args()

    logger.info("Arguments:")
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")

    # Initialize the predictor and load weights for inference
    predictor = PropertyPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    predictor.load_inference_model(device=args.device)

    # Load structures
    files, structures = prepare_structures(args, predictor)

    # Predict property
    predictor.get_predict(
        files, structures, file_path=args.file_path, save_path=args.save_path
    )
    logger.info("All property predictions finished successfully.")
