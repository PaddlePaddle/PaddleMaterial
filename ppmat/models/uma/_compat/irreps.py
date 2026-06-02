from __future__ import annotations

import paddle


def cg_change_mat(ang_mom: int, place=None, dtype: str | paddle.dtype = "float32") -> paddle.Tensor:
    if ang_mom != 2:
        raise NotImplementedError
    change_mat = paddle.to_tensor(
        [
            [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
            [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
            [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
            [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
            [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
            [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
            [-(6 ** (-0.5)), 0, 0, 0, 2 * 6 ** (-0.5), 0, 0, 0, -(6 ** (-0.5))],
            [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
            [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
        ],
        dtype=dtype,
        place=place,
    )
    return change_mat.detach()


def irreps_sum(ang_mom: int) -> int:
    total = 0
    for i in range(ang_mom + 1):
        total += 2 * i + 1
    return total
