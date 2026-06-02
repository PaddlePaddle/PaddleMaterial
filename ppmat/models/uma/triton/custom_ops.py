from __future__ import annotations

import paddle


"""Paddle fallback custom op wrappers used by UMA triton compatibility layer."""


def _kernel_node_to_edge_wigner_permute(
    x: paddle.Tensor,
    edge_index: paddle.Tensor,
    wigner: paddle.Tensor,
    out: paddle.Tensor,
    x_edge: paddle.Tensor,
) -> None:
    src = x[edge_index[0]]
    tgt = x[edge_index[1]]
    tmp = paddle.concat((src, tgt), axis=2)
    x_edge[...] = tmp
    out[...] = paddle.bmm(wigner, tmp)


def _kernel_permute_wigner_inv_edge_to_node(
    x: paddle.Tensor, wigner: paddle.Tensor, out: paddle.Tensor, x_l: paddle.Tensor
) -> None:
    x_l[...] = x
    out[...] = paddle.bmm(wigner, x)


def _kernel_node_to_edge_wigner_permute_bwd_dx(
    grad_out: paddle.Tensor, wigner: paddle.Tensor, grad_edge: paddle.Tensor
) -> None:
    grad_edge[...] = paddle.bmm(paddle.transpose(wigner, [0, 2, 1]), grad_out)


def _kernel_permute_wigner_inv_edge_to_node_bwd_dx(
    grad_out: paddle.Tensor, wigner: paddle.Tensor, grad_x: paddle.Tensor
) -> None:
    grad_x[...] = paddle.bmm(paddle.transpose(wigner, [0, 2, 1]), grad_out)


def _kernel_permute_wigner_inv_edge_to_node_bwd_dw(
    grad_out: paddle.Tensor, x_l: paddle.Tensor, grad_wigner_flat: paddle.Tensor
) -> None:
    grad_w = paddle.bmm(grad_out, paddle.transpose(x_l, [0, 2, 1]))
    grad_wigner_flat[...] = grad_w.reshape([grad_w.shape[0], -1])
