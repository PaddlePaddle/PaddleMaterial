import paddle

from ppmat.models.liflow import LiFlow


def test_liflow_forward_backward():
    model = LiFlow(num_features=8, num_radial_basis=4, num_layers=1, num_elements=10)
    batch = {
        "positions_1": paddle.randn([3, 3]),
        "positions_2": paddle.randn([3, 3]),
        "prior": paddle.zeros([3, 3]),
        "elements": paddle.to_tensor([1, 2, 3]),
        "num_atoms": paddle.to_tensor([3]),
        "time": paddle.to_tensor([0.5]),
        "temp": paddle.to_tensor([800.0]),
    }
    result = model(batch)
    assert result["pred_dict"]["target"].shape == [3, 3]
    result["loss_dict"]["loss"].backward()
