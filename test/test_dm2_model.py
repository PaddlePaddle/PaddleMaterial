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

import numpy as np
import paddle

from ppmat.datasets.geometric_data_type.batch import Batch
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.models.dm2 import DM2


def _toy_graph(offset=0.0):
    pos = paddle.to_tensor(
        [
            [0.0 + offset, 0.0, 0.0],
            [0.8 + offset, 0.0, 0.0],
            [0.0 + offset, 0.8, 0.0],
        ],
        dtype="float32",
    )
    edge_index = paddle.to_tensor(
        [
            [0, 1, 2, 1, 2, 0],
            [1, 2, 0, 0, 1, 2],
        ],
        dtype="int64",
    )
    src, dst = edge_index[0], edge_index[1]
    return Data(
        x=paddle.to_tensor([0, 1, 0], dtype="int64"),
        pos=pos,
        edge_index=edge_index,
        edge_attr=pos[dst] - pos[src],
        atomic_numbers=paddle.to_tensor([8, 14, 8], dtype="int64"),
        lattice=paddle.eye(3, dtype="float32").unsqueeze(axis=0) * 5.0,
        cooling_rate=paddle.to_tensor([1.0], dtype="float32"),
        num_nodes=3,
    )


def test_dm2_forward_and_sample_smoke():
    paddle.seed(42)
    model = DM2(
        cutoff=2.0,
        sigma_min=0.001,
        sigma_max=0.01,
        denoiser_cfg={
            "num_species": 2,
            "node_embedding_dim": 4,
            "edge_basis_size": 4,
            "irreps_hidden": "8x0e + 4x1e",
            "irreps_edge": "2x0e + 1x1e",
            "irreps_out": "1x1e",
            "num_convs": 2,
            "radial_neurons": [4, 8],
            "num_neighbors": 3,
            "use_condition": True,
        },
    )
    batch = Batch.from_data_list([_toy_graph(), _toy_graph(offset=1.0)])
    result = model(batch)
    assert "loss" in result["loss_dict"]
    assert np.isfinite(float(result["loss_dict"]["loss"]))

    sampled = model.sample(batch, num_inference_steps=2, final_relax_steps=1)
    assert len(sampled["result"]) == 2
    assert sampled["result"][0]["frac_coords"].shape == (3, 3)
