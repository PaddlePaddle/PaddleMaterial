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
Example usage:

1. Run prediction with a given model name:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python ppmat/tasks/property.py \
        predict \
        --model_name chgnet_mptrj

2. Run prediction with specified config, checkpoint and input structures:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python ppmat/tasks/property.py \
        predict \
        --config_path "output/chgnet_mptrj/chgnet_mptrj.yaml" \
        --checkpoint_path "output/chgnet_mptrj/checkpoints" \
        --file_path "interatomic_potentials/example_data/cifs" \

3. Run prediction with manually defined ASE structures,
need to modify `structures` variable in the script:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python ppmat/tasks/property.py \
        predict \
        --config_path "output/chgnet_mptrj/chgnet_mptrj.yaml" \
        --checkpoint_path "output/chgnet_mptrj/checkpoints"
"""

from __future__ import annotations

from ppmat.predict import PropertyPredictor
from ppmat.predict import parse_args
from ppmat.utils import logger

# Option A: Load structures from file (default behavior if `--file_path` is set)
structures = None

# Option B: Manually generate a single structure using ASE
# from ase.build import bulk
# structures = bulk("Fe")
# structures = bulk("Fe").repeat((4, 4, 4))

# Option C: Generate multiple structures using ASE
# from ase.build import bulk
# structures = [bulk("Cu"), bulk("Al")]


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
            from pymatgen.io.ase import AseAtomsAdaptor

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
