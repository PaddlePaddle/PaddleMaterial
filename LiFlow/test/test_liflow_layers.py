import numpy as np
import paddle

from ppmat.models.liflow.layers import BesselBasis, CosineCutoff, GaussianFourierBasis


def test_basis_shapes_and_cutoff():
    distance = paddle.to_tensor([0.5, 6.0], dtype="float32")
    assert BesselBasis(4, 5.0)(distance).shape == [2, 4]
    # GaussianFourierBasis requires an even num_basis and outputs num_basis channels
    assert GaussianFourierBasis(4)(distance).shape == [2, 4]
    np.testing.assert_allclose(CosineCutoff(5.0)(distance).numpy(), [0.97552824, 0.0], rtol=1e-5)
