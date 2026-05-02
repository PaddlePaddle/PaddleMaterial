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
import pytest

from ppmat.schedulers.scheduling_dm2 import DM2DenoisingScheduler
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
        scheduler_cfg={
            "sigma_min": 0.001,
            "sigma_max": 0.01,
            "default_num_inference_steps": 2,
            "default_final_relax_steps": 1,
        },
    )
    batch = Batch.from_data_list([_toy_graph(), _toy_graph(offset=1.0)])
    result = model(batch)
    assert "loss" in result["loss_dict"]
    assert np.isfinite(float(result["loss_dict"]["loss"]))

    sampled = model.sample(batch)
    assert len(sampled["result"]) == 2
    assert sampled["result"][0]["frac_coords"].shape == (3, 3)


def test_dm2_scheduler_fixed_sigma():
    graph = _toy_graph()
    scheduler = DM2DenoisingScheduler(sigma_min=0.1, sigma_max=0.1)
    noisy_graph = scheduler.add_noise(graph, sigma=0.1)
    assert noisy_graph.dx.shape == noisy_graph.pos.shape
    assert noisy_graph.sigma.shape == [3, 1]
    assert paddle.allclose(
        noisy_graph.pos,
        _toy_graph().pos + noisy_graph.dx,
        atol=1e-6,
    )


def test_dm2_metric_smoke(tmp_path):
    pytest.importorskip("ase")
    pytest.importorskip("pandas")
    pytest.importorskip("p_tqdm")
    pytest.importorskip("pymatgen")
    pytest.importorskip("smact")
    pytest.importorskip("matminer")
    pytest.importorskip("scipy")

    from ase import Atoms
    import ase.io
    from ppmat.metrics.dm2_metric import DM2AmorphousGenerationMetric

    atoms = Atoms(
        numbers=[14, 8, 8],
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0]],
        cell=np.eye(3) * 6.0,
        pbc=True,
    )
    ref_path = tmp_path / "ref.extxyz"
    ase.io.write(ref_path, atoms, format="extxyz")

    metric = DM2AmorphousGenerationMetric(
        reference_paths=str(ref_path),
        file_format="extxyz",
        rdf_cutoff=4.0,
        rdf_bins=16,
        coordination_cutoffs=[
            {
                "name": "si_o_coordination",
                "center_atomic_number": 14,
                "neighbor_atomic_number": 8,
                "cutoff": 2.0,
            }
        ],
    )
    result = metric(
        [
            {
                "frac_coords": atoms.get_scaled_positions(),
                "atom_types": atoms.numbers,
                "lattice": atoms.cell.array,
            }
        ]
    )
    assert result["rdf_wasserstein"] == 0.0
    assert result["si_o_coordination"] == 2.0
