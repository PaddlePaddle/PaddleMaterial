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

"""ASU (Asymmetric Unit) dataset metadata and data-directory resolution.

A dependency-free leaf module shared by the data layer
(``ppmat.datasets.asu_dataset``) and the model layer
(``ppmat.models.sgequidiff``). Keeping it here lets both layers depend on it
without depending on each other.
"""

import os
from pathlib import Path

from pymatgen.core import Element

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Supported ASU dataset names. Each dataset stores preprocessed asymmetric-unit
# data as NPZ archives under ``<data_dir>/<name>/{train,val,test}.npz``.
SUPPORTED_DATASETS = ("mp_20", "mp_20_assumeP1", "mpts_52")

# Element-encoding table size. Not the periodic-table total (118); covers the
# elements in the supported ASU datasets (max Z=94) with margin. Index 0 is a
# placeholder. Used by the ASU data layout and the model embeddings.
ELEMENT_ENCODING_SIZE: int = 98

# Symbol table for 0-indexed atomic numbers in [0, ELEMENT_ENCODING_SIZE);
# built from pymatgen to avoid a hand-written table.
chemical_symbols = ["X"] + [
    Element.from_Z(atomic_number).symbol
    for atomic_number in range(1, ELEMENT_ENCODING_SIZE + 1)
]
assert len(chemical_symbols) == ELEMENT_ENCODING_SIZE + 1

_CANDIDATE_DIRS = [
    _PROJECT_ROOT / "data/data",
    Path("~").expanduser() / ".asu_data",
]


def resolve_asu_data_dir() -> Path:
    """Locate the ASU data directory.

    Priority: $ASU_DATA_DIR > project data dirs > ~/.asu_data.
    A directory is accepted if it contains any of the supported ASU
    dataset directories (npz files). Model-specific resources (e.g.
    ``ppmat.models.sgequidiff.vocabs``) live in their own package and
    resolve via ``_resolve_data_file`` rather than this directory.
    """
    env = os.getenv("ASU_DATA_DIR")
    if env:
        return Path(env)

    for candidate in _CANDIDATE_DIRS:
        if any((candidate / d).is_dir() for d in SUPPORTED_DATASETS):
            return candidate

    _candidate_paths = "\n".join([f"  * {d}" for d in _CANDIDATE_DIRS])
    raise FileNotFoundError(
        f"Cannot find ASU data directory.\n"
        f"\nTried paths:\n{_candidate_paths}\n"
        f"\nSolutions:\n"
        f"  1. Copy data to: {str(_CANDIDATE_DIRS[0])}\n"
        f"  2. Set environment variable: export ASU_DATA_DIR=/your/data/path\n"
        f"\nRequired: at least one of dataset dirs {list(SUPPORTED_DATASETS)}"
    )
