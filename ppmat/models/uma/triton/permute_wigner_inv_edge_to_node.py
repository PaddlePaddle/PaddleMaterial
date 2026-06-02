from __future__ import annotations

import paddle


class PermuteWignerInvEdgeToNodeFunction:
    """Paddle fallback for inverse Wigner edge->node op."""

    @staticmethod
    def apply(x: paddle.Tensor, wigner: paddle.Tensor) -> paddle.Tensor:
        return paddle.bmm(wigner, x)
