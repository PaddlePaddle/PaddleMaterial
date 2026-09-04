# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""Cutoff neighbor-list construction shared by the dataset and the integrator.

Faithful semantics to liflow.utils.geometry.get_neighbor_list_batch:
shifts are returned in Cartesian coordinates (integer lattice shift @ lattice).
"""

import numpy as np


def get_neighbor_list(positions, lattice, cutoff, pbc=True):
    """Build a cutoff neighbor list.

    Args:
        positions (np.ndarray): [n_atoms, 3] Cartesian coordinates.
        lattice (np.ndarray | None): [3, 3] lattice matrix (required for pbc).
        cutoff (float): neighbor cutoff.
        pbc (bool): whether periodic images are considered.

    Returns:
        (np.ndarray, np.ndarray): edge_index [2, E] int64 and shifts [E, 3]
        float32 (Cartesian).
    """
    positions = np.asarray(positions, dtype=np.float32)
    n = len(positions)
    if not pbc or lattice is None:
        image_shifts = np.zeros((1, 3), dtype=np.float32)
        lattice = None
    else:
        lattice = np.asarray(lattice, dtype=np.float32).reshape(3, 3)
        inv = np.linalg.inv(lattice)
        span = np.ceil(float(cutoff) * np.linalg.norm(inv, axis=0)).astype(int) + 1
        image_shifts = np.asarray(
            list(np.ndindex(*(2 * span + 1))), dtype=np.float32
        ) - span

    source = positions[:, None, :]
    target = positions[None, :, :]
    all_edges = []
    all_shifts = []
    for image_shift in image_shifts:
        cart_shift = (image_shift @ lattice).astype(np.float32) if lattice is not None else np.zeros(3, dtype=np.float32)
        distances = np.linalg.norm(target + cart_shift - source, axis=-1)
        mask = distances <= cutoff
        if lattice is None:
            np.fill_diagonal(mask, False)
        elif np.all(image_shift == 0):
            np.fill_diagonal(mask, False)
        rows, cols = np.nonzero(mask)
        if len(rows):
            all_edges.append(np.stack([rows, cols], axis=0))
            all_shifts.append(np.repeat(cart_shift[None, :], len(rows), axis=0))
    if not all_edges:
        return np.empty((2, 0), np.int64), np.empty((0, 3), np.float32)
    return np.concatenate(all_edges, axis=1).astype(np.int64), np.concatenate(all_shifts, axis=0).astype(np.float32)
