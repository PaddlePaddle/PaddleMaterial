from __future__ import annotations

import logging

import paddle

from .rotation import eulers_to_wigner


class RotMatWignerCudaGraph:
    """Compatibility fallback: keep API but use eager Paddle path."""

    def __init__(self):
        self.graph_mod = None
        self.graph_capture_count = 0
        self.max_edge_size = None
        logging.info("rotation_cuda_graph fallback: using eager Paddle implementation")

    def _capture_graph(self, edge_dist_vec: paddle.Tensor, jds: list[paddle.Tensor]):
        self.max_edge_size = edge_dist_vec.shape[0]
        self.graph_capture_count += 1

    def get_rotmat_and_wigner(
        self, edge_dist_vec: paddle.Tensor, jds: list[paddle.Tensor]
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        if self.max_edge_size is None or edge_dist_vec.shape[0] > self.max_edge_size:
            self._capture_graph(edge_dist_vec, jds)
        return edge_rot_and_wigner_graph_capture_region(edge_dist_vec, jds)


def capture_rotmat_and_wigner_with_make_graph_callable(
    edge_dist_vec: paddle.Tensor, jds: list[paddle.Tensor]
):
    del edge_dist_vec, jds
    return edge_rot_and_wigner_graph_capture_region


def edge_rot_and_wigner_graph_capture_region(
    edge_distance_vecs: paddle.Tensor, Jd_buffers: list[paddle.Tensor]
):
    lmax = len(Jd_buffers) - 1
    _, alpha, beta, gamma = init_edge_rot_euler_angles_wigner_cuda_graph(
        edge_distance_vecs
    )
    wigner = eulers_to_wigner((alpha, beta, gamma), 0, lmax, Jd_buffers)
    wigner_inv = paddle.transpose(wigner, [0, 2, 1]).contiguous()
    return wigner, wigner_inv


def init_edge_rot_euler_angles_wigner_cuda_graph(edge_distance_vec):
    edge_vec_0_distance = paddle.sqrt(paddle.sum(edge_distance_vec**2, axis=1))
    xyz = edge_distance_vec / edge_vec_0_distance.reshape([-1, 1])
    mask = paddle.isclose(paddle.abs(xyz[:, 1]), paddle.ones([1], dtype=xyz.dtype))
    beta = paddle.acos(xyz[:, 1])
    alpha = paddle.atan2(x=xyz[:, 0], y=xyz[:, 2])
    gamma = paddle.rand_like(alpha) * 2 * paddle.pi
    return mask, -gamma, -beta, -alpha
