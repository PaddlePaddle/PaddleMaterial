from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Constants for Triton kernel operations.
"""

BLOCK_C = 128
GRID_E_STRIDE = 2048
M_TO_L_GATHER_IDX = [0, 5, 1, 3, 8, 6, 2, 4, 7]
