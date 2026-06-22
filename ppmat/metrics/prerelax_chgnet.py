# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CHGNet energy prediction utility for S.U.N. Stability metric.
"""

import logging
from typing import List
from typing import Optional

import numpy as np
from pymatgen.core import Structure
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Lazy-loaded globals to avoid import cost at module level
_chgnet_model = None
_graph_converter = None


def _load_chgnet(model_name: str = "chgnet_mptrj"):
    """Lazy-load CHGNet model and graph converter."""
    global _chgnet_model, _graph_converter
    if _chgnet_model is not None:
        return _chgnet_model, _graph_converter

    from ppmat.models import build_model_from_name
    from ppmat.models.chgnet.chgnet_graph_converter import CHGNetGraphConverter

    _graph_converter = CHGNetGraphConverter(
        atom_graph_cutoff=6.0, bond_graph_cutoff=3.0
    )
    model, _ = build_model_from_name(model_name)
    model.eval()
    _chgnet_model = model
    return _chgnet_model, _graph_converter


def predict_energy(
    structures: List[Optional[Structure]],
    model_name: str = "chgnet_mptrj",
    batch_size: int = 1,
) -> List[Optional[float]]:
    """Predict energy per atom for each structure using CHGNet.

    Args:
        structures: List of pymatgen Structures (None for invalid).
        model_name: CHGNet model name in ppmat registry.
        batch_size: Prediction batch size.

    Returns:
        List of energy per atom (eV/atom), None for invalid structures.
    """
    model, converter = _load_chgnet(model_name)
    energies = []

    for struct in tqdm(structures, desc="Predicting energies (CHGNet)"):
        if struct is None:
            energies.append(None)
            continue
        try:
            graph = converter(struct)
            pred = model.predict(graph)
            e_per_atom = float(np.array(pred["energy_per_atom"]).flatten()[0])
            energies.append(e_per_atom)
        except Exception as e:
            logger.debug(f"Energy prediction failed: {e}")
            energies.append(None)

    return energies
