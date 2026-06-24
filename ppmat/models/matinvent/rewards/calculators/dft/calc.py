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
DFT property calculator.


Runs DFT calculations (e.g., VASP, Quantum ESPRESSO) to compute
electronic properties like band gap, formation energy, etc.

This is a simplified implementation. For full functionality:
1. Configure DFT software (VASP, QE, etc.)
2. Set up remote job queue system
3. Configure dft_config.yaml
"""

import os
from typing import List
from typing import Tuple

import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

from ppmat.models.matinvent.rewards.base import Calculator
from ppmat.models.matinvent.rewards.calculators.dft import DFT_CONFIG_PATH


def dft_run_stub(task, work_dir, cif_path, config_path):
    """Stub function for DFT calculation.

    In production, this would submit a DFT job to a queue system
    and wait for results.

    Args:
        task: Type of calculation (e.g., 'band_gap', 'formation_energy')
        work_dir: Working directory for the calculation
        cif_path: Path to CIF file
        config_path: Path to DFT configuration

    Returns:
        Calculated property value (or NaN if not implemented)
    """
    return np.nan


class DFTCalc(Calculator):
    """DFT property calculator.

    Supports running DFT calculations for:
    - band_gap: Electronic band gap
    - formation_energy: Formation energy
    - energy_per_atom: Energy per atom
    - And other DFT-computed properties

    Requires:
    - DFT software (VASP, Quantum ESPRESSO, etc.)
    - Configuration file (dft_config.yaml)
    - Access to compute cluster (if remote)
    """

    def __init__(
        self,
        root_dir: str,
        task: str = "band_gap",
        max_node: int = 8,
        config_path: str = None,
    ) -> None:
        """Initialize DFT calculator.

        Args:
            root_dir: Directory for output files
            task: Type of DFT calculation (band_gap, formation_energy, etc.)
            max_node: Maximum number of parallel DFT jobs
            config_path: Path to DFT configuration file
        """
        super().__init__(root_dir, task)
        self.max_node = max_node
        if config_path is None:
            self.config_path = DFT_CONFIG_PATH
        else:
            self.config_path = os.path.abspath(config_path)

    def calc(
        self, samples: Tuple[List[Structure], str], label: str = "tmp"
    ) -> np.ndarray:
        """Calculate DFT properties.

        Args:
            samples: Tuple of (structure_list, path)
            label: Label for output files

        Returns:
            Array of calculated property values
        """
        struc_list = samples[0]
        cif_dir = os.path.join(self.root_dir, label)
        os.makedirs(cif_dir, exist_ok=True)

        results = []
        for i, struc in enumerate(struc_list):
            cif_writer = CifWriter(struc)
            cif_path = os.path.join(cif_dir, f"{i}.cif")
            cif_path = os.path.abspath(cif_path)
            cif_writer.write_file(cif_path)

            work_dir = os.path.join(self.root_dir, label, f"{i:02d}")
            os.makedirs(work_dir, exist_ok=True)

            try:
                result = dft_run_stub(self.task, work_dir, cif_path, self.config_path)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"DFT calculation failed for structure {i}: {e}\n"
                    "DFT calculator requires proper configuration and DFT software."
                )
                result = np.nan

            results.append(result)

        results = np.array(results, dtype=float)
        out_path = os.path.join(self.root_dir, f"{label}.txt")
        np.savetxt(out_path, results, fmt="%.6f")
        return results


"""
machine: remote  # or 'local' for local calculations

remote:
  host: cluster.example.com
  username: user
  key_file: ~/.ssh/id_rsa
  work_dir: /scratch/dft/calculations
  queue_system: slurm  # or pbs, cobalt, etc.

dft:
  software: vasp  # or quantum_espresso, abinit, etc.
  encut: 520
  ediff: 1e-6
  kpoints: [4, 4, 4]
  potcar_path: /path/to/potcar

band_gap:
  nbands: 100
  icharg: 11

formation_energy:
  reference_states: true
"""
