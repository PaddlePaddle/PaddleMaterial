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

"""
FairChem property calculator.


Uses FairChem machine learning models to compute materials properties
like bulk modulus and heat capacity.

This is a simplified implementation. For full functionality:
1. Install FairChem package
2. Configure conda environment
3. Set up elastic.py and phonon.py prediction scripts
"""

import os
from typing import List
from typing import Tuple

import numpy as np
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rewards.base import Calculator


def fairchem_predict_stub(xyz_path, out_path, task, num_workers=1):
    """Stub function for FairChem prediction.

    In production, this would run FairChem models to predict properties.

    Args:
        xyz_path: Path to input XYZ file
        out_path: Path to output file
        task: Type of prediction (bulk_modulus, heat_capacity)
        num_workers: Number of parallel workers

    Returns:
        Predicted property values
    """
    # This is a stub - implement actual FairChem prediction
    # For now, return NaN to indicate prediction not performed
    return np.array([np.nan])


class FairChem(Calculator):
    """FairChem property calculator.

    Uses FairChem machine learning models to predict:
    - bulk_modulus: Bulk modulus (GPa)
    - heat_capacity: Heat capacity at various temperatures
    - And other materials properties

    Requires:
    - FairChem package installation
    - Pre-trained FairChem models
    - Prediction scripts (elastic.py, phonon.py)
    """

    def __init__(
        self,
        root_dir: str,
        task: str = "bulk_modulus",
        env_name: str = "fair-chem-v1",
        worker: int = 1,
    ) -> None:
        """Initialize FairChem calculator.

        Args:
            root_dir: Directory for output files
            task: Type of property (bulk_modulus, heat_capacity)
            env_name: Conda environment name with FairChem installed
            worker: Number of parallel workers
        """
        super().__init__(root_dir, task)
        self.env_name = env_name
        self.worker = worker

    def calc(
        self, samples: Tuple[List[Structure], str], label: str = "tmp"
    ) -> np.ndarray:
        """Calculate FairChem properties.

        Args:
            samples: Tuple of (structure_list, xyz_path)
            label: Label for output files

        Returns:
            Array of calculated property values
        """
        xyz_path = samples[1]
        out_path = os.path.join(self.root_dir, f"{label}.txt")
        xyz_path = os.path.abspath(xyz_path)
        out_path = os.path.abspath(out_path)

        if self.task == "bulk_modulus":
            pass
        elif self.task == "heat_capacity":
            pass
        else:
            raise ValueError(f"{self.task} is unknown task for FairChem calculator!")

        try:
            results = fairchem_predict_stub(xyz_path, out_path, self.task, self.worker)
            if os.path.isfile(out_path):
                results = np.genfromtxt(out_path)
        except Exception as e:
            import warnings

            warnings.warn(
                f"FairChem prediction failed: {e}\n"
                "FairChem calculator requires proper configuration and models."
            )
            results = np.full(len(samples[0]), np.nan)

        return results
