from __future__ import annotations

import paddle

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Wigner D matrices via hybrid approach (fastest method per l).

This module provides Wigner D computation using the optimal method for each l:
- l=0: Trivial (identity)
- l=1: Direct quaternion to rotation matrix (fastest for 3x3)
- l=2: Quaternion einsum tensor contraction (~20x faster on GPU)
- l=3,4: Batched quaternion matmul (single kernel dispatch)
- l>=5: Ra/Rb polynomial (faster than matrix_exp on GPU)

Entry point:
- axis_angle_wigner_hybrid: Main function using real arithmetic throughout
"""

import math
from typing import TYPE_CHECKING

from .quaternion_utils import (
    quaternion_edge_to_y_stable, quaternion_multiply, quaternion_y_rotation)
from .quaternion_wigner_utils import (
    WignerCoefficients, quaternion_to_ra_rb_real, wigner_d_matrix_real,
    wigner_d_pair_to_real)
from .wigner_d_custom_kernels import (
    quaternion_to_rotation_matrix, quaternion_to_wigner_d_l2_einsum,
    quaternion_to_wigner_d_l3l4_batched, quaternion_to_wigner_d_matmul)

if TYPE_CHECKING:
    from .wigner_d_custom_kernels import CustomKernelModule


def wigner_d_from_quaternion_hybrid(
    q: paddle.Tensor,
    lmax: int,
    coeffs: WignerCoefficients,
    U_blocks: list[tuple[paddle.Tensor, paddle.Tensor]],
    custom_kernels: CustomKernelModule,
) -> paddle.Tensor:
    """
    Compute Wigner D matrices from quaternion using hybrid approach.

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion to Wigner D via degree-4 polynomial einsum
    - l=3: Quaternion matmul (used when lmax=3)
    - l=3,4: Batched quaternion matmul (used when lmax>=4)
    - l>=5: Ra/Rb polynomial

    Uses real-pair arithmetic throughout for torch.compile compatibility.

    Args:
        q: Quaternions of shape (N, 4) in (w, x, y, z) convention
        lmax: Maximum angular momentum
        coeffs: Pre-computed WignerCoefficients for l>=5 Ra/Rb path.
        U_blocks: Pre-computed U transformation blocks for l>=5.
        custom_kernels: CustomKernelModule holding l=2,3,4 coefficient buffers.

    Returns:
        Block-diagonal Wigner D matrices of shape (N, size, size)
        where size = (lmax+1)^2
    """
    N = q.shape[0]
    device = q.device
    dtype = q.dtype
    size = (lmax + 1) ** 2
    D = paddle.zeros(N, size, size, dtype=dtype, device=device)
    D[:, 0, 0] = 1.0
    if lmax >= 1:
        D[:, 1:4, 1:4] = quaternion_to_rotation_matrix(q)
    if lmax >= 2:
        D[:, 4:9, 4:9] = quaternion_to_wigner_d_l2_einsum(q, custom_kernels.C_l2)
    if lmax >= 4:
        D_l3, D_l4 = quaternion_to_wigner_d_l3l4_batched(
            q, custom_kernels.C_combined_l3l4, custom_kernels.monomials_l4
        )
        D[:, 9:16, 9:16] = D_l3
        D[:, 16:25, 16:25] = D_l4
    elif lmax >= 3:
        D[:, 9:16, 9:16] = quaternion_to_wigner_d_matmul(
            q, 3, custom_kernels.C_l3, custom_kernels.monomials_l3
        )
    lmin = 5
    if lmax >= lmin:
        ra_re, ra_im, rb_re, rb_im = quaternion_to_ra_rb_real(q)
        D_re, D_im = wigner_d_matrix_real(ra_re, ra_im, rb_re, rb_im, coeffs)
        D_range = wigner_d_pair_to_real(D_re, D_im, U_blocks, lmin=lmin, lmax=lmax)
        block_offset = lmin * lmin
        D[:, block_offset:, block_offset:] = D_range
    return D


def axis_angle_wigner_hybrid(
    edge_distance_vec: paddle.Tensor,
    lmax: int,
    gamma: (paddle.Tensor | None) = None,
    coeffs: (WignerCoefficients | None) = None,
    U_blocks: (list[tuple[paddle.Tensor, paddle.Tensor]] | None) = None,
    custom_kernels: (CustomKernelModule | None) = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    Compute Wigner D using hybrid approach (optimal method per l).

    Uses the fastest method for each l:
    - l=0: Trivial (identity)
    - l=1: Quaternion to rotation matrix (fastest for 3x3, already Cartesian)
    - l=2: Quaternion einsum tensor contraction
    - l=3,4: Batched quaternion matmul (single kernel dispatch)
    - l>=5: Ra/Rb polynomial

    Combines the edge->Y and gamma rotations into a single quaternion before
    computing the Wigner D, avoiding the overhead of computing two separate
    Wigner D matrices and multiplying them.

    Uses real-pair arithmetic throughout for torch.compile compatibility.

    Args:
        edge_distance_vec: Edge vectors of shape (N, 3)
        lmax: Maximum angular momentum
        gamma: Optional roll angles of shape (N,).
               If None, uses random gamma (for SO(2) equivariance during training).
        coeffs: Pre-computed WignerCoefficients for l>=5 Ra/Rb path.
        U_blocks: Pre-computed U transformation blocks for l>=5.
        custom_kernels: CustomKernelModule holding l=2,3,4 coefficient buffers.

    Returns:
        Tuple of (wigner_edge_to_y, wigner_y_to_edge) where each has shape
        (N, size, size) and size = (lmax+1)^2.
    """
    if edge_distance_vec.dim() == 1:
        edge_distance_vec = edge_distance_vec.unsqueeze(0)
    N = edge_distance_vec.shape[0]
    device = edge_distance_vec.device
    dtype = edge_distance_vec.dtype
    edge_normalized = paddle.nn.functional.normalize(edge_distance_vec, dim=-1)
    if gamma is None:
        gamma = paddle.rand(N, dtype=dtype, device=device) * 2 * math.pi
    q_edge_to_y = quaternion_edge_to_y_stable(edge_normalized)
    q_gamma = quaternion_y_rotation(gamma)
    q_combined = quaternion_multiply(q_gamma, q_edge_to_y)
    D = wigner_d_from_quaternion_hybrid(
        q_combined,
        lmax,
        coeffs=coeffs,
        U_blocks=U_blocks,
        custom_kernels=custom_kernels,
    )
    D_inv = D.transpose(1, 2).contiguous()
    return D, D_inv
