# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Note:
    `ase_opt` is used for structure relaxation (optimization)
    `ase_md` is used for molecular dynamics simulation


Example usage:

1. Run ASE molecular dynamics with a given model name:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python ppmat/tasks/ase_run.py \
        ase_md \
        --model_name chgnet_mptrj

2. Run ASE optimization with specified config, checkpoint, and input structures:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python ppmat/tasks/ase_run.py \
        ase_opt \
        --config_path "output/chgnet_mptrj/chgnet_mptrj.yaml" \
        --checkpoint_path "output/chgnet_mptrj/checkpoints" \
        --file_path "interatomic_potentials/example_data/cifs"

3. Run ASE optimization with manually defined ASE structures.
   Note: Modify the `structures` variable inside the script before running:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python ppmat/tasks/ase_run.py \
        ase_opt \
        --config_path "output/chgnet_mptrj/chgnet_mptrj.yaml" \
        --checkpoint_path "output/chgnet_mptrj/checkpoints"

"""

from __future__ import annotations

from ppmat.predict import ASECalculator
from ppmat.predict import PPMatPredictor
from ppmat.predict import parse_args
from ppmat.utils import logger

# Option A: Load structures from file (default)
structures = None

# Option B: Manually generate a single structure using ASE
# from ase.build import bulk
# structures = bulk("Fe")
# structures = bulk("Fe").repeat((4, 4, 4))

# Option C: Generate multiple structures using ASE
# from ase.build import bulk
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
            from pymatgen.io.ase import AseAtomsAdaptor

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
