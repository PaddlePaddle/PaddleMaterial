from typing import Optional
from typing import Tuple

import numpy as np
import paddle


def radius_graph(
    x: paddle.Tensor,
    r: float,
    batch: Optional[paddle.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
) -> paddle.Tensor:
    if batch is None:
        batch = paddle.zeros((x.shape[0],), dtype="int64")

    if x.shape[0] <= 1000:
        return radius_graph_simple(x, r, batch, loop, max_num_neighbors)
    else:
        return radius_graph_grid(x, r, batch, loop, max_num_neighbors)


def radius(
    x: paddle.Tensor,
    y: paddle.Tensor,
    r: float,
    batch_x: Optional[paddle.Tensor] = None,
    batch_y: Optional[paddle.Tensor] = None,
    max_num_neighbors: int = 32,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    if batch_x is None:
        batch_x = paddle.zeros((x.shape[0],), dtype="int64")
    if batch_y is None:
        batch_y = paddle.zeros((y.shape[0],), dtype="int64")

    atoms_grids_scenario = x.shape[0] < 1000 and y.shape[0] > 1000

    if atoms_grids_scenario:
        return radius_atoms_to_grids(x, y, r, batch_x, batch_y, max_num_neighbors)
    elif x.shape[0] > 1000 or y.shape[0] > 1000:
        return radius_grid(x, y, r, batch_x, batch_y, max_num_neighbors)
    else:
        return radius_simple(x, y, r, batch_x, batch_y, max_num_neighbors)


def radius_graph_simple(
    x: paddle.Tensor,
    r: float,
    batch: paddle.Tensor,
    loop: bool = False,
    max_num_neighbors: int = 32,
) -> paddle.Tensor:
    batch_size = int(batch.max().item()) + 1
    row_list, col_list = [], []

    for b in range(batch_size):
        mask = batch == b
        subset_x = x[mask]
        subset_idx = paddle.nonzero(mask, as_tuple=True)[0]

        n = subset_x.shape[0]
        if n == 0:
            continue

        dist = (
            paddle.sum(subset_x**2, axis=1, keepdim=True)
            + paddle.sum(subset_x**2, axis=1, keepdim=True).T
            - 2 * paddle.matmul(subset_x, subset_x.T)
        )
        dist = paddle.clip(dist, min=0.0)

        adj = dist <= r * r

        if not loop:
            diag_mask = paddle.eye(n, dtype="int32") == 0
            adj = adj & diag_mask

        row, col = paddle.nonzero(adj, as_tuple=True)

        if row.shape[0] > 0:
            if max_num_neighbors < 1000000:
                unique_rows, counts = paddle.unique(row, return_counts=True)
                keep_mask = paddle.ones_like(row, dtype="bool")

                for node, count in zip(unique_rows, counts):
                    if count > max_num_neighbors:
                        node_mask = row == node
                        edge_indices = paddle.nonzero(node_mask, as_tuple=True)[0]
                        perm = paddle.randperm(count.item())
                        drop_indices = edge_indices[perm[max_num_neighbors:]]
                        keep_mask[drop_indices] = False

                row = row[keep_mask]
                col = col[keep_mask]

            row_global = subset_idx[row]
            col_global = subset_idx[col]
            row_list.append(row_global)
            col_list.append(col_global)

    if len(row_list) == 0:
        return paddle.zeros([2, 0], dtype="int64")

    row = paddle.concat(row_list)
    col = paddle.concat(col_list)
    row = paddle.squeeze(row)
    col = paddle.squeeze(col)
    return paddle.stack([row, col], axis=0)


def radius_simple(
    x: paddle.Tensor,
    y: paddle.Tensor,
    r: float,
    batch_x: paddle.Tensor,
    batch_y: paddle.Tensor,
    max_num_neighbors: int = 32,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    batch_size = max(int(batch_x.max().item()), int(batch_y.max().item())) + 1
    x_idx_list, y_idx_list = [], []

    for b in range(batch_size):
        mask_x = batch_x == b
        mask_y = batch_y == b

        subset_x = x[mask_x]
        subset_y = y[mask_y]
        idx_x = paddle.nonzero(mask_x, as_tuple=True)[0]
        idx_y = paddle.nonzero(mask_y, as_tuple=True)[0]

        nx, ny = subset_x.shape[0], subset_y.shape[0]
        if nx == 0 or ny == 0:
            continue

        dist = (
            paddle.sum(subset_x**2, axis=1, keepdim=True)
            + paddle.sum(subset_y**2, axis=1, keepdim=True).T
            - 2 * paddle.matmul(subset_x, subset_y.T)
        )
        dist = paddle.clip(dist, min=0.0)

        adj = dist <= r * r
        row, col = paddle.nonzero(adj, as_tuple=True)

        if row.shape[0] > 0:
            if max_num_neighbors < 1000000:
                unique_rows, counts = paddle.unique(row, return_counts=True)
                keep_mask = paddle.ones_like(row, dtype="bool")

                for node, count in zip(unique_rows, counts):
                    if count > max_num_neighbors:
                        node_mask = row == node
                        edge_indices = paddle.nonzero(node_mask, as_tuple=True)[0]
                        perm = paddle.randperm(count.item())
                        drop_indices = edge_indices[perm[max_num_neighbors:]]
                        keep_mask[drop_indices] = False

                row = row[keep_mask]
                col = col[keep_mask]

            x_global = idx_x[row]
            y_global = idx_y[col]
            x_idx_list.append(x_global)
            y_idx_list.append(y_global)

    if len(x_idx_list) == 0:
        return paddle.zeros([0], dtype="int64"), paddle.zeros([0], dtype="int64")

    x_idx = paddle.concat(x_idx_list)
    y_idx = paddle.concat(y_idx_list)

    return y_idx, x_idx


def radius_atoms_to_grids(
    x: paddle.Tensor,
    y: paddle.Tensor,
    r: float,
    batch_x: paddle.Tensor,
    batch_y: paddle.Tensor,
    max_num_neighbors: int = 32,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    batch_size = max(int(batch_x.max().item()), int(batch_y.max().item())) + 1

    grid_idx_list, atom_idx_list = [], []
    r_squared = r * r

    for b in range(batch_size):
        atom_mask = batch_x == b
        grid_mask = batch_y == b

        if not atom_mask.any() or not grid_mask.any():
            continue

        atoms = x[atom_mask].numpy()
        atom_indices = paddle.nonzero(atom_mask, as_tuple=True)[0]
        grid_indices = paddle.nonzero(grid_mask, as_tuple=True)[0]

        min_coords = np.min(atoms, axis=0) - r
        max_coords = np.max(atoms, axis=0) + r
        cell_size = r

        atom_grid = {}
        for i, atom_pos in enumerate(atoms):
            cell_idx = tuple(np.floor((atom_pos - min_coords) / cell_size).astype(int))
            if cell_idx not in atom_grid:
                atom_grid[cell_idx] = []
            atom_grid[cell_idx].append(i)

        grids = y[grid_mask]
        grid_chunk_size = 10000

        for start_idx in range(0, grids.shape[0], grid_chunk_size):
            end_idx = min(start_idx + grid_chunk_size, grids.shape[0])
            grid_chunk = grids[start_idx:end_idx]
            grid_chunk_np = grid_chunk.numpy()

            batch_grid_idx, batch_atom_idx = [], []

            for i, grid_pos in enumerate(grid_chunk_np):
                cell_idx = tuple(
                    np.floor((grid_pos - min_coords) / cell_size).astype(int)
                )

                nearby_atoms = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            nei_cell = (
                                cell_idx[0] + dx,
                                cell_idx[1] + dy,
                                cell_idx[2] + dz,
                            )
                            if nei_cell in atom_grid:
                                nearby_atoms.extend(atom_grid[nei_cell])

                if not nearby_atoms:
                    continue

                atom_pos = atoms[nearby_atoms]
                dists = np.sum((atom_pos - grid_pos) ** 2, axis=1)
                valid_mask = dists <= r_squared

                valid_atoms = np.array(nearby_atoms)[valid_mask]
                n_valid = valid_atoms.size

                if n_valid > 0:
                    if n_valid > max_num_neighbors:
                        perm = np.random.permutation(n_valid)
                        valid_atoms = valid_atoms[perm[:max_num_neighbors]]

                    batch_grid_idx.extend([i + start_idx] * len(valid_atoms))
                    batch_atom_idx.extend(valid_atoms.tolist())

            if batch_grid_idx:
                global_grid_idx = grid_indices[batch_grid_idx].numpy()
                global_atom_idx = atom_indices[batch_atom_idx].numpy()

                grid_idx_list.append(paddle.to_tensor(global_grid_idx, dtype="int64"))
                atom_idx_list.append(paddle.to_tensor(global_atom_idx, dtype="int64"))

    if not grid_idx_list:
        return paddle.zeros([0], dtype="int64"), paddle.zeros([0], dtype="int64")

    grid_idx = paddle.concat(grid_idx_list)
    atom_idx = paddle.concat(atom_idx_list)

    if len(grid_idx.shape) > 1:
        grid_idx = paddle.squeeze(grid_idx, axis=-1)
    if len(atom_idx.shape) > 1:
        atom_idx = paddle.squeeze(atom_idx, axis=-1)
    return grid_idx, atom_idx


def radius_graph_grid(
    x: paddle.Tensor,
    r: float,
    batch: paddle.Tensor,
    loop: bool = False,
    max_num_neighbors: int = 32,
) -> paddle.Tensor:
    x_cpu = x.numpy()
    batch_cpu = batch.numpy()
    batch_size = int(batch.max().item()) + 1

    row_list, col_list = [], []

    for b in range(batch_size):
        mask = batch_cpu == b
        if not mask.any():
            continue

        subset_x = x_cpu[mask]
        mask_indices = np.where(mask)[0]
        n = subset_x.shape[0]

        min_coords = np.min(subset_x, axis=0) - r
        max_coords = np.max(subset_x, axis=0) + r
        cell_size = r

        grid_dict = {}
        for i in range(n):
            cell_idx = tuple(
                np.floor((subset_x[i] - min_coords) / cell_size).astype(int)
            )
            if cell_idx not in grid_dict:
                grid_dict[cell_idx] = []
            grid_dict[cell_idx].append(i)

        rows, cols = [], []
        for i in range(n):
            point = subset_x[i]
            cell_idx = tuple(np.floor((point - min_coords) / cell_size).astype(int))

            neighbor_indices = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        nei_idx = (cell_idx[0] + dx, cell_idx[1] + dy, cell_idx[2] + dz)
                        if nei_idx in grid_dict:
                            neighbor_indices.extend(grid_dict[nei_idx])

            if neighbor_indices:
                neighbors = subset_x[neighbor_indices]
                dists = np.sum((neighbors - point) ** 2, axis=1)
                valid = dists <= r * r

                if not loop:
                    valid = valid & (np.array(neighbor_indices) != i)

                valid_indices = np.array(neighbor_indices)[valid]

                if len(valid_indices) > max_num_neighbors:
                    perm = np.random.permutation(len(valid_indices))
                    valid_indices = valid_indices[perm[:max_num_neighbors]]

                if len(valid_indices) > 0:
                    rows.extend([i] * len(valid_indices))
                    cols.extend(valid_indices.tolist())

        if rows:
            global_rows = mask_indices[rows]
            global_cols = mask_indices[cols]

            row_tensor = paddle.to_tensor(global_rows, dtype="int64")
            col_tensor = paddle.to_tensor(global_cols, dtype="int64")

            row_list.append(row_tensor)
            col_list.append(col_tensor)

    if not row_list:
        return paddle.zeros([2, 0], dtype="int64")

    row = paddle.concat(row_list)
    col = paddle.concat(col_list)
    row = paddle.squeeze(row, axis=-1)
    col = paddle.squeeze(col, axis=-1)
    return paddle.stack([row, col], axis=0)


def radius_grid(
    x: paddle.Tensor,
    y: paddle.Tensor,
    r: float,
    batch_x: paddle.Tensor,
    batch_y: paddle.Tensor,
    max_num_neighbors: int = 32,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    x_cpu = x.numpy()
    y_cpu = y.numpy()
    batch_x_cpu = batch_x.numpy()
    batch_y_cpu = batch_y.numpy()

    batch_size = max(int(batch_x.max().item()), int(batch_y.max().item())) + 1

    x_idx_list, y_idx_list = [], []

    for b in range(batch_size):
        mask_x = batch_x_cpu == b
        mask_y = batch_y_cpu == b

        if not mask_x.any() or not mask_y.any():
            continue

        subset_x = x_cpu[mask_x]
        subset_y = y_cpu[mask_y]
        idx_x = np.where(mask_x)[0]
        idx_y = np.where(mask_y)[0]

        nx, ny = subset_x.shape[0], subset_y.shape[0]

        min_coords = (
            np.min(np.vstack([subset_x.min(axis=0), subset_y.min(axis=0)]), axis=0) - r
        )
        max_coords = (
            np.max(np.vstack([subset_x.max(axis=0), subset_y.max(axis=0)]), axis=0) + r
        )
        cell_size = r

        grid_dict = {}
        for i in range(ny):
            cell_idx = tuple(
                np.floor((subset_y[i] - min_coords) / cell_size).astype(int)
            )
            if cell_idx not in grid_dict:
                grid_dict[cell_idx] = []
            grid_dict[cell_idx].append(i)

        batch_x_idx, batch_y_idx = [], []

        for i in range(nx):
            point = subset_x[i]
            cell_idx = tuple(np.floor((point - min_coords) / cell_size).astype(int))

            neighbor_indices = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        nei_idx = (cell_idx[0] + dx, cell_idx[1] + dy, cell_idx[2] + dz)
                        if nei_idx in grid_dict:
                            neighbor_indices.extend(grid_dict[nei_idx])

            if neighbor_indices:
                neighbors = subset_y[neighbor_indices]
                dists = np.sum((neighbors - point) ** 2, axis=1)
                valid = dists <= r * r

                valid_indices = np.array(neighbor_indices)[valid]

                if len(valid_indices) > max_num_neighbors:
                    perm = np.random.permutation(len(valid_indices))
                    valid_indices = valid_indices[perm[:max_num_neighbors]]

                if len(valid_indices) > 0:
                    batch_x_idx.extend([i] * len(valid_indices))
                    batch_y_idx.extend(valid_indices.tolist())

        if batch_x_idx:
            global_x_idx = idx_x[batch_x_idx]
            global_y_idx = idx_y[batch_y_idx]

            x_tensor = paddle.to_tensor(global_x_idx, dtype="int64")
            y_tensor = paddle.to_tensor(global_y_idx, dtype="int64")

            x_idx_list.append(x_tensor)
            y_idx_list.append(y_tensor)

    if not x_idx_list:
        return paddle.zeros([0], dtype="int64"), paddle.zeros([0], dtype="int64")

    x_idx = paddle.concat(x_idx_list)
    y_idx = paddle.concat(y_idx_list)

    return y_idx, x_idx
