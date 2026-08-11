# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified ASU data/resource directory resolution shared by datasets and models."""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TARGET_FILE = "wyckoff_positions/clean_wyckoffs_in_asu_v6.json"

# Supported ASU dataset names. Each dataset stores preprocessed asymmetric-unit
# data as NPZ archives under ``<data_dir>/<name>/{train,val,test}.npz``.
SUPPORTED_DATASETS = ("mp_20", "mp_20_assumeP1", "mpts_52")

# Per-dataset metadata used by models for sampling / construction.
lattice_parameter_ranges = {
    "mp_20": {
        "min_lattice_length": 2.0,
        "max_lattice_length": 133.0,
        "min_lattice_angle": 60.0,
        "max_lattice_angle": 135.0,
    },
    "mpts_52": {
        "min_lattice_length": 0.98,
        "max_lattice_length": 189.5,
        "min_lattice_angle": 60.0,
        "max_lattice_angle": 135.0,
    },
}
# Maximum number of atoms in the full unit cell for each dataset. Used to bound
# autoregressive Wyckoff/element generation and to size padded tensors.
max_atoms_per_dataset = {
    "mp_20": 20,
    "mpts_52": 52,
}

_CANDIDATE_DIRS = [
    _PROJECT_ROOT / "data/data",
    Path("~").expanduser() / ".asu_data",
]


def resolve_asu_data_dir() -> Path:
    """Locate the ASU data directory.

    Priority: $ASU_DATA_DIR > project data dirs > ~/.asu_data.
    A directory is accepted if it contains either the Wyckoff position data
    (model resources) or one of the ASU dataset directories (npz files).
    """
    env = os.getenv("ASU_DATA_DIR")
    if env:
        return Path(env)

    for candidate in _CANDIDATE_DIRS:
        if (candidate / _TARGET_FILE).exists():
            return candidate
        if any((candidate / d).is_dir() for d in SUPPORTED_DATASETS):
            return candidate

    _candidate_paths = "\n".join([f"  * {d}" for d in _CANDIDATE_DIRS])
    raise FileNotFoundError(
        f"Cannot find ASU data directory.\n"
        f"\nTried paths:\n{_candidate_paths}\n"
        f"\nSolutions:\n"
        f"  1. Copy data to: {str(_CANDIDATE_DIRS[0])}\n"
        f"  2. Set environment variable: export ASU_DATA_DIR=/your/data/path\n"
        f"\nRequired: {_TARGET_FILE} or one of dataset dirs {list(SUPPORTED_DATASETS)}"
    )
