import paddle
from mattergen.diffusion.data.batched_data import (SimpleBatchedData,
                                                   _batch_edge_index,
                                                   collate_fn)


def test_collate_fn():
    state1 = dict(foo=paddle.ones(shape=[2, 3]), bar=paddle.ones(shape=[5, 2]))
    state2 = dict(foo=paddle.zeros(shape=[3, 3]), bar=paddle.zeros(shape=[2, 2]))
    batch = collate_fn([state1, state2])
    field_names = list(state1.keys())
    expected = SimpleBatchedData(
        data=dict(
            foo=paddle.to_tensor(
                data=[
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            bar=paddle.to_tensor(
                data=[
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype="float32",
            ),
        ),
        batch_idx={
            "foo": paddle.to_tensor(data=[0, 0, 1, 1, 1], dtype="int64"),
            "bar": paddle.to_tensor(data=[0, 0, 0, 0, 0, 1, 1], dtype="int64"),
        },
    )
    for k in field_names:
        assert paddle.equal_all(x=batch[k], y=expected[k]).item()
        assert paddle.equal_all(
            x=batch.get_batch_idx(k), y=expected.get_batch_idx(k)
        ).item()
    assert batch.get_batch_size() == 2


def test_batch_edge_index():
    edge_index = paddle.to_tensor(
        data=[[0, 1], [0, 2], [1, 2], [0, 1], [0, 3], [1, 2], [2, 3], [0, 1], [1, 3]]
    )
    atom_batch_idx = paddle.to_tensor(data=[0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    edge_batch_idx = paddle.to_tensor(data=[0, 0, 0, 1, 1, 1, 1, 2, 2])
    assert paddle.allclose(
        x=_batch_edge_index(edge_index, atom_batch_idx, edge_batch_idx),
        y=paddle.to_tensor(
            data=[
                [0, 1],
                [0, 2],
                [1, 2],
                [2, 3],
                [2, 5],
                [3, 4],
                [4, 5],
                [7, 8],
                [8, 10],
            ]
        ),
    ).item(), ""
