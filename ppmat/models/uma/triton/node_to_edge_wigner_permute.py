from __future__ import annotations

import paddle


class NodeToEdgeWignerPermuteFunction:
    """Paddle fallback for node->edge Wigner permutation op."""

    @staticmethod
    def apply(x: paddle.Tensor, edge_index: paddle.Tensor, wigner: paddle.Tensor) -> paddle.Tensor:
        x_source = x[edge_index[0]]
        x_target = x[edge_index[1]]
        x_message = paddle.concat((x_source, x_target), axis=2)
        return paddle.bmm(wigner, x_message)
