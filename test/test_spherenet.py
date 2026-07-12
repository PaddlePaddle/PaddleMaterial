# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import math

import numpy as np
import paddle
from rdkit import Chem
from scipy import special

from ppmat.datasets.collate_fn import RadiusGraphCollator
from ppmat.models.common.graph_converter import RadiusGraphConverter
from ppmat.models.common.spherical_fourier_bessel import RealSphericalHarmonics
from ppmat.models.common.spherical_fourier_bessel import SphericalBesselBasis
from ppmat.models.common.spherical_fourier_bessel import (
    SphericalFourierBesselEmbedding,
)
from ppmat.models.common.spherical_fourier_bessel import _build_basis_constants
from ppmat.models.spherenet.geometry import compute_geometry
from ppmat.models.spherenet.spherenet import SphereNet


def setup_module():
    paddle.set_device("cpu")


def test_basis_constants_are_cached_by_shape():
    _build_basis_constants.cache_clear()
    constants_7_6 = _build_basis_constants(7, 6)
    constants_3_4 = _build_basis_constants(3, 4)

    assert constants_7_6 is _build_basis_constants(7, 6)
    assert constants_7_6[0].shape == (7, 6)
    assert constants_3_4[0].shape == (3, 4)
    assert not constants_7_6[0].flags.writeable
    assert _build_basis_constants.cache_info().currsize == 2


def test_spherical_bessel_matches_scipy_value_and_gradient():
    num_spherical = 7
    num_radial = 6
    distances = np.array(
        [1e-4, 0.03, 0.15, 0.7, 1.3, 2.6, 4.9],
        dtype=np.float32,
    )
    dist = paddle.to_tensor(distances, stop_gradient=False)
    actual = SphericalBesselBasis(num_spherical, num_radial)(dist)

    zeros, normalizers, _ = _build_basis_constants(
        num_spherical, num_radial
    )
    arguments = distances[:, None, None].astype(np.float64) / 5.0
    arguments = arguments * zeros[None]
    expected = np.empty_like(arguments)
    expected_gradient = np.zeros(len(distances), dtype=np.float64)
    for degree in range(num_spherical):
        expected[:, degree, :] = (
            special.spherical_jn(degree, arguments[:, degree, :])
            * normalizers[degree]
        )
        expected_gradient += np.sum(
            special.spherical_jn(
                degree,
                arguments[:, degree, :],
                derivative=True,
            )
            * normalizers[degree]
            * zeros[degree]
            / 5.0,
            axis=1,
        )

    np.testing.assert_allclose(
        actual.numpy(), expected, rtol=1e-4, atol=1e-5
    )
    actual_gradient = paddle.grad(paddle.sum(actual), dist)[0]
    np.testing.assert_allclose(
        actual_gradient.numpy(),
        expected_gradient,
        rtol=1e-4,
        atol=3e-5,
    )


def test_real_spherical_harmonics_preserve_spherenet_order():
    angle = paddle.to_tensor([math.pi / 2], dtype="float32")
    torsion = paddle.zeros([1], dtype="float32")
    harmonics = RealSphericalHarmonics(2)(angle, torsion)

    scale = math.sqrt(3.0 / (4.0 * math.pi))
    expected = np.array(
        [[1.0 / math.sqrt(4.0 * math.pi), 0.0, -scale, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        harmonics.numpy(), expected, rtol=1e-6, atol=1e-6
    )


def test_embedding_supports_dynamic_shapes_and_empty_triplets():
    for num_spherical, num_radial in ((1, 1), (3, 4), (7, 6)):
        embedding = SphericalFourierBesselEmbedding(
            num_spherical, num_radial
        )
        dist = paddle.to_tensor([0.8, 1.2], dtype="float32")
        angle = paddle.to_tensor([0.5], dtype="float32")
        torsion = paddle.to_tensor([0.2], dtype="float32")
        idx_kj = paddle.to_tensor([1], dtype="int64")
        angle_embedding, torsion_embedding = embedding(
            dist, angle, torsion, idx_kj
        )
        assert angle_embedding.shape == [
            1, num_spherical * num_radial
        ]
        assert torsion_embedding.shape == [
            1,
            num_spherical * num_spherical * num_radial,
        ]

    empty = paddle.empty([0], dtype="float32")
    empty_index = paddle.empty([0], dtype="int64")
    angle_embedding, torsion_embedding = embedding(
        dist, empty, empty, empty_index
    )
    assert angle_embedding.shape == [0, 42]
    assert torsion_embedding.shape == [0, 294]


def test_radius_graph_uses_edges_as_endpoint_indices():
    xyz_blocks = [
        "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n",
        (
            "5\nmethane\nC 0 0 0\nH 0.63 0.63 0.63\n"
            "H -0.63 -0.63 0.63\nH -0.63 0.63 -0.63\n"
            "H 0.63 -0.63 -0.63\n"
        ),
    ]
    converter = RadiusGraphConverter(
        cutoff=5.0, return_triplet_indices=True
    )
    graphs = converter(
        [Chem.MolFromXYZBlock(block) for block in xyz_blocks]
    )
    for graph in graphs:
        assert "ti_i" not in graph.edge_feat
        assert "ti_j" not in graph.edge_feat

    batch = RadiusGraphCollator()(
        [
            {"graph": graph, "id": index}
            for index, graph in enumerate(graphs)
        ]
    )
    graph = batch["graph"].tensor()
    edge_index = paddle.transpose(graph.edges.astype("int64"), [1, 0])
    triplet_indices = {
        "idx_kj": graph.edge_feat["ti_idx_kj"].astype("int64"),
        "idx_ji": graph.edge_feat["ti_idx_ji"].astype("int64"),
        "idx_lk": graph.edge_feat["ti_idx_lk"].astype("int64"),
        "idx_triplet": graph.edge_feat["ti_idx_triplet"].astype("int64"),
    }
    result = compute_geometry(
        graph.node_feat["pos"], edge_index, triplet_indices
    )
    np.testing.assert_array_equal(result[3].numpy(), edge_index[0].numpy())
    np.testing.assert_array_equal(result[4].numpy(), edge_index[1].numpy())



def test_predict_returns_energy_and_force():
    molecule = Chem.MolFromXYZBlock(
        "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
    )
    graph = RadiusGraphConverter(
        cutoff=5.0, return_triplet_indices=True
    )(molecule)
    model = SphereNet(
        energy_and_force=True,
        property_name="energy",
        num_layers=0,
        hidden_channels=16,
        int_emb_size=8,
        basis_emb_size_dist=4,
        basis_emb_size_angle=4,
        basis_emb_size_torsion=4,
        out_emb_channels=16,
        num_spherical=2,
        num_radial=2,
        num_output_layers=1,
    )

    prediction = model.predict(graph)

    assert prediction["energy"].shape == (1, 1)
    assert prediction["force"].shape == (3, 3)
    assert np.isfinite(prediction["energy"]).all()
    assert np.isfinite(prediction["force"]).all()
