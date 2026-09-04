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

import json
import zlib

import lmdb
import numpy as np
from pymatgen.core import Lattice
from pymatgen.core import Structure

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.datasets.uma_dataset import UMAOC20Dataset
from ppmat.datasets.uma_dataset import UMAOMat24Dataset
from ppmat.models import UMA
from ppmat.models import FindPointsInSpheres


def test_uma_dataset_resolves_only_requested_split(tmp_path):
    extracted_root = tmp_path / "uma_datasets"
    oc20_train = extracted_root / "oc20" / "uma_aselmdb" / "train"
    omat24_val = extracted_root / "omat24" / "val" / "rattled-500"
    oc20_train.mkdir(parents=True)
    omat24_val.mkdir(parents=True)

    oc20_dataset = UMAOC20Dataset.__new__(UMAOC20Dataset)
    omat24_dataset = UMAOMat24Dataset.__new__(UMAOMat24Dataset)

    assert oc20_dataset._get_downloaded_split_path(tmp_path, "train") == str(oc20_train)
    assert omat24_dataset._get_downloaded_split_path(tmp_path, "val") == str(omat24_val)


def test_uma_oc20_dataset_reads_one_split(tmp_path):
    database_path = tmp_path / "train.aselmdb"
    database = lmdb.open(str(database_path), subdir=False, map_size=1024**2)
    record = {
        "numbers": [14, 14],
        "positions": [[0.0, 0.0, 0.0], [0.875, 0.875, 0.875]],
        "cell": [[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]],
        "pbc": [True, True, True],
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }
    with database.begin(write=True) as transaction:
        transaction.put(b"length", b"1")
        transaction.put(b"1", zlib.compress(json.dumps(record).encode()))
    database.close()

    dataset = UMAOC20Dataset(
        path=str(database_path),
        split="train",
        build_graph_cfg={
            "__class_name__": "FindPointsInSpheres",
            "__init_params__": {"cutoff": 4.0, "max_neighbors": 8, "num_cpus": 1},
        },
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["energy"].shape == (1,)
    assert sample["forces"].shape == (2, 3)
    assert sample["graph"].num_nodes == 2


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
