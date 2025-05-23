from typing import Tuple

import paddle


def direct_sum(*matrices):
    """Direct sum of matrices, put them in the diagonal"""
    front_indices = tuple(matrices[0].shape)[:-2]
    m = sum(x.shape[-2] for x in matrices)
    n = sum(x.shape[-1] for x in matrices)
    total_shape = list(front_indices) + [m, n]
    out = paddle.zeros(shape=total_shape, dtype=matrices[0].dtype)
    i, j = 0, 0
    for x in matrices:
        m, n = tuple(x.shape)[-2:]
        out[..., i : i + m, j : j + n] = x
        i += m
        j += n
    return out


def orthonormalize(
    original: paddle.Tensor, eps: float = 1e-09
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """orthonomalize vectors

    Parameters
    ----------
    original : `torch.Tensor`
        list of the original vectors :math:`x`

    eps : float
        a small number

    Returns
    -------
    final : `torch.Tensor`
        list of orthonomalized vectors :math:`y`

    matrix : `torch.Tensor`
        the matrix :math:`A` such that :math:`y = A x`
    """
    assert original.dim() == 2
    dim = tuple(original.shape)[1]
    final = []
    matrix = []
    for i, x in enumerate(original):
        cx = paddle.zeros(shape=len(original), dtype=x.dtype)
        cx[i] = 1
        for j, y in enumerate(final):
            c = paddle.dot(x=x, y=y)
            x = x - c * y
            cx = cx - c * matrix[j]
        if x.norm() > 2 * eps:
            c = 1 / x.norm()
            x = c * x
            cx = c * cx
            x[x.abs() < eps] = 0
            cx[cx.abs() < eps] = 0
            c = x[x.nonzero()[0, 0]].sign()
            x = c * x
            cx = c * cx
            final += [x]
            matrix += [cx]
    final = (
        paddle.stack(x=final)
        if len(final) > 0
        else paddle.zeros(shape=(0, dim), dtype=original.dtype)
    )
    matrix = (
        paddle.stack(x=matrix)
        if len(matrix) > 0
        else paddle.zeros(shape=(0, len(original)), dtype=original.dtype)
    )
    return final, matrix


def complete_basis(vecs: paddle.Tensor, eps: float = 1e-09) -> paddle.Tensor:
    assert vecs.dim() == 2
    dim = tuple(vecs.shape)[1]
    base = [(x / x.norm()) for x in vecs]
    expand = []
    for x in paddle.eye(num_rows=dim, dtype=vecs.dtype):
        for y in base + expand:
            x -= paddle.dot(x=x, y=y) * y
        if x.norm() > 2 * eps:
            x /= x.norm()
            x[x.abs() < eps] = paddle.zeros(shape=(), dtype=x.dtype)
            x *= x[x.nonzero()[0, 0]].sign()
            expand += [x]
    expand = (
        paddle.stack(x=expand)
        if len(expand) > 0
        else paddle.zeros(shape=[0, dim], dtype=vecs.dtype)
    )
    return expand
