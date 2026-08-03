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

import numpy as np
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.models import UMA
from ppmat.models import FindPointsInSpheres


def test_uma_forward_with_default_collator():
    converter = FindPointsInSpheres(
        cutoff=4.0,
        max_neighbors=8,
        num_cpus=1,
    )
    structures = [
        Structure(
            Lattice.cubic(3.5),
            ["Si", "Si"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            Lattice.cubic(4.0),
            ["C", "C", "C"],
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]],
        ),
    ]

    samples = []
    for structure in structures:
        graph = converter(structure)
        samples.append(
            {
                "graph": graph,
                "energy": np.zeros([1], dtype="float32"),
                "forces": ConcatNumpyWarper(
                    np.zeros([len(structure), 3], dtype="float32")
                ),
            }
        )
    batch = DefaultCollator()(samples)

    model = UMA(
        sphere_channels=16,
        edge_channels=16,
        hidden_channels=16,
        num_distance_basis=16,
        num_layers=1,
    )
    output = model(batch)

    assert output["pred_dict"]["energy"].shape == [2, 1]
    assert output["pred_dict"]["forces"].shape == [5, 3]
    output["loss_dict"]["loss"].backward()

    prediction = model.predict(batch["graph"])
    assert prediction["energy"].shape == (2, 1)
    assert prediction["forces"].shape == (5, 3)
