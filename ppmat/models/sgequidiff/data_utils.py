"""
晶体数据处理工具函数：坐标转换、图构建、空间群约束晶格参数。
"""
from typing import Tuple, Optional

import numpy as np
import paddle

import ppmat.models.sgequidiff.global_vars as global_vars
from ppmat.models.sgequidiff.constants import OFFSET_LIST
from ppmat.models.sgequidiff.scatter_utils import safe_scatter as scatter


def _scatter_min_with_argmin_gpu_safe(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    scatter_min + argmin 的 GPU 安全实现。
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    n = src.shape[0]
    sort_key = index.cast(paddle.float64) * (float(n) + 1.0) + src.cast(paddle.float64)
    sorted_order = paddle.argsort(sort_key)
    sorted_index = index[sorted_order]
    sorted_src = src[sorted_order]

    group_first = paddle.concat([
        paddle.to_tensor([True], dtype=paddle.bool),
        sorted_index[1:] != sorted_index[:-1],
    ])
    first_pos = paddle.nonzero(group_first).squeeze(1)
    group_ids = sorted_index[first_pos]
    argmin_orig_idx = sorted_order[first_pos]
    min_vals_group = sorted_src[first_pos]

    min_vals = paddle.full([dim_size], float('inf'), dtype=src.dtype)
    argmin = paddle.full([dim_size], dim_size, dtype=paddle.int64)
    min_vals = paddle.scatter(min_vals, group_ids, min_vals_group)
    argmin = paddle.scatter(argmin, group_ids, argmin_orig_idx)
    return min_vals, argmin

def _segment_coo_sum_gpu_safe(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    """
    使用 scatter(reduce='sum') 替代 segment_coo。GPU 安全。
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    return scatter(src=src, index=index, dim_size=dim_size, reduce="sum")

def _scatter_min_indices_gpu_safe(
    ov_row: paddle.Tensor,
    ov_col: paddle.Tensor,
    n_total: int,
) -> paddle.Tensor:
    """
    返回每个 ov_row 对应的最小 ov_col。GPU 安全。未出现的行返回自身索引。
    """
    if ov_row.shape[0] == 0:
        return paddle.arange(n_total, dtype=paddle.int64)

    sort_key = ov_row.cast(paddle.int64) * n_total + ov_col.cast(paddle.int64)
    sorted_order = paddle.argsort(sort_key)
    sorted_row = ov_row[sorted_order]
    sorted_col = ov_col[sorted_order]

    row_first = paddle.concat([
        paddle.to_tensor([True], dtype=paddle.bool),
        sorted_row[1:] != sorted_row[:-1],
    ])
    first_pos = paddle.nonzero(row_first).squeeze(1)
    unique_rows = sorted_row[first_pos]
    min_cols = sorted_col[first_pos]

    result = paddle.arange(n_total, dtype=paddle.float32)
    result = paddle.scatter(result, unique_rows, min_cols.cast(paddle.float32))
    return result.cast(paddle.int64)

def scatter_min_with_argmin(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    scatter_min + argmin。GPU 安全的纯 Paddle 实现。
    """
    return _scatter_min_with_argmin_gpu_safe(src, index, dim_size)


def lattice_params_to_matrix_paddle(
    lattice_lengths: paddle.Tensor,
    lattice_angles: paddle.Tensor,
) -> paddle.Tensor:
    """
    将晶格参数 (a,b,c, alpha,beta,gamma) 转换为 (N,3,3) 晶格矩阵。
    """
    angles_r = paddle.deg2rad(lattice_angles)
    coses = paddle.cos(angles_r)
    sins = paddle.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = paddle.clip(val, -1.0, 1.0)
    gamma_star = paddle.acos(val)

    zeros = paddle.zeros([lattice_lengths.shape[0]], dtype=lattice_lengths.dtype)
    vector_a = paddle.stack(
        [
            lattice_lengths[:, 0] * sins[:, 1],
            zeros,
            lattice_lengths[:, 0] * coses[:, 1],
        ],
        axis=1,
    )
    vector_b = paddle.stack(
        [
            -lattice_lengths[:, 1] * sins[:, 0] * paddle.cos(gamma_star),
            lattice_lengths[:, 1] * sins[:, 0] * paddle.sin(gamma_star),
            lattice_lengths[:, 1] * coses[:, 0],
        ],
        axis=1,
    )
    vector_c = paddle.stack(
        [zeros, zeros, lattice_lengths[:, 2]], axis=1,
    )
    return paddle.stack([vector_a, vector_b, vector_c], axis=1)

def lattice_matrix_to_params(matrix: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    将 (N,3,3) 晶格矩阵转换为 (N,3) lengths 和 (N,3) angles。
    """
    lengths = paddle.sqrt(paddle.sum(matrix ** 2, axis=-1))

    j = paddle.to_tensor([1, 2, 0], dtype=paddle.int64)
    k = paddle.to_tensor([2, 0, 1], dtype=paddle.int64)
    angles = paddle.clip(
        (matrix[:, j, :] * matrix[:, k, :]).sum(axis=-1) / (lengths[:, j] * lengths[:, k]),
        -1.0, 1.0,
    )
    angles = paddle.acos(angles) * 180.0 / paddle.to_tensor(float(np.pi))
    return lengths, angles


def _expand_lattice_to_nodes(
    lattice_tensor: paddle.Tensor,
    num_atoms_per_crystal: paddle.Tensor,
    num_nodes: int,
) -> paddle.Tensor:
    """根据每晶体原子数展开晶格张量到每原子节点。"""
    if num_nodes == 0:
        return lattice_tensor[:0]

    atom_counts = num_atoms_per_crystal.cast("int64").reshape([-1])
    cumulative = paddle.cumsum(atom_counts, axis=0)
    atom_ids = paddle.arange(num_nodes, dtype=cumulative.dtype).reshape([-1, 1])
    crystal_ids = (atom_ids >= cumulative.reshape([1, -1])).cast("int64").sum(axis=1)
    crystal_ids = paddle.clip(crystal_ids, min=0, max=lattice_tensor.shape[0] - 1)
    return paddle.gather(lattice_tensor, crystal_ids, axis=0)

def frac_to_cart_coords(
    frac_coords: paddle.Tensor,
    num_atoms_per_crystal: paddle.Tensor,
    lattice_lengths: Optional[paddle.Tensor] = None,
    lattice_angles: Optional[paddle.Tensor] = None,
    lattice_matrix: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """
    分数坐标 -> 笛卡尔坐标。
    """
    if lattice_matrix is None:
        assert lattice_lengths is not None and lattice_angles is not None
        lattice_matrix = lattice_params_to_matrix_paddle(lattice_lengths, lattice_angles)
    lattice_nodes = _expand_lattice_to_nodes(
        lattice_matrix.cast(paddle.float32),
        num_atoms_per_crystal,
        frac_coords.shape[0],
    )
    cart_coords = paddle.einsum("bi,bij->bj", frac_coords, lattice_nodes)
    return cart_coords

def cart_to_frac_coords(
    cart_coords: paddle.Tensor,
    num_atoms_per_crystal: paddle.Tensor,
    lattice_lengths: Optional[paddle.Tensor] = None,
    lattice_angles: Optional[paddle.Tensor] = None,
    lattice_matrix: Optional[paddle.Tensor] = None,
    mod_lattice_translations: bool = True,
) -> paddle.Tensor:
    """
    笛卡尔坐标 -> 分数坐标。
    """
    if lattice_matrix is None:
        assert lattice_lengths is not None and lattice_angles is not None
        lattice_matrix = lattice_params_to_matrix_paddle(lattice_lengths, lattice_angles)
    inv_lattice = paddle.linalg.pinv(lattice_matrix)
    inv_lattice_nodes = _expand_lattice_to_nodes(
        inv_lattice,
        num_atoms_per_crystal,
        cart_coords.shape[0],
    )
    frac_coords = paddle.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    if mod_lattice_translations:
        frac_coords = frac_coords % 1.0
    return frac_coords

def lattice_transform_and_log_prob_mask(
    spacegroup: int,
    device: str = "cpu",
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    返回用于空间群约束晶格参数的变换矩阵和掩码。
    """
    if 1 <= spacegroup <= 2:
        length_matrix = paddle.eye(3)
        angle_matrix = paddle.eye(3)
        angle_vector = paddle.zeros([3])
        log_prob_mask = paddle.ones([6])
    elif 3 <= spacegroup <= 15:
        length_matrix = paddle.eye(3)
        angle_matrix = paddle.to_tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        angle_vector = paddle.to_tensor([90.0, 0.0, 90.0])
        log_prob_mask = paddle.to_tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    elif 16 <= spacegroup <= 74:
        length_matrix = paddle.eye(3)
        angle_matrix = paddle.zeros([3, 3])
        angle_vector = paddle.to_tensor([90.0, 90.0, 90.0])
        log_prob_mask = paddle.to_tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    elif 75 <= spacegroup <= 142:
        length_matrix = paddle.to_tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        angle_matrix = paddle.zeros([3, 3])
        angle_vector = paddle.to_tensor([90.0, 90.0, 90.0])
        log_prob_mask = paddle.to_tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    elif 143 <= spacegroup <= 194:
        length_matrix = paddle.to_tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        angle_matrix = paddle.zeros([3, 3])
        angle_vector = paddle.to_tensor([90.0, 90.0, 120.0])
        log_prob_mask = paddle.to_tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    elif 195 <= spacegroup <= 230:
        length_matrix = paddle.to_tensor(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        angle_matrix = paddle.zeros([3, 3])
        angle_vector = paddle.to_tensor([90.0, 90.0, 90.0])
        log_prob_mask = paddle.to_tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        raise AttributeError(f"Invalid space group: {spacegroup}")
    return length_matrix, angle_matrix, angle_vector, log_prob_mask

def paddle_legal_lattice_parameters(
    space_groups: paddle.Tensor,
    lattice_parameters: paddle.Tensor,
    lattice_log_probs: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    对晶格参数施加空间群约束。
    """
    lattice_lengths, lattice_angles = paddle.chunk(lattice_parameters, 2, axis=1)

    length_projection_matrices = []
    angle_projection_matrices = []
    angle_translation_vectors = []
    log_prob_masks = []
    for spacegroup in space_groups.numpy().tolist():
        (
            length_matrix,
            angle_matrix,
            angle_vector,
            log_prob_mask,
        ) = lattice_transform_and_log_prob_mask(int(spacegroup))
        length_projection_matrices.append(length_matrix)
        angle_projection_matrices.append(angle_matrix)
        angle_translation_vectors.append(angle_vector)
        log_prob_masks.append(log_prob_mask)

    length_projection_matrices = paddle.stack(length_projection_matrices, axis=0)
    angle_projection_matrices = paddle.stack(angle_projection_matrices, axis=0)
    angle_translation_vectors = paddle.stack(angle_translation_vectors, axis=0)
    log_prob_masks = paddle.stack(log_prob_masks, axis=0)

    constrained_lengths = paddle.bmm(
        lattice_lengths.unsqueeze(1), length_projection_matrices
    ).squeeze(1)
    constrained_angles = (
        paddle.bmm(lattice_angles.unsqueeze(1), angle_projection_matrices).squeeze(1)
        + angle_translation_vectors
    )
    lattice_masked_log_probs = log_prob_masks * lattice_log_probs
    return constrained_lengths, constrained_angles, lattice_masked_log_probs

def primitive_lattice_matrix_from_conventional_lattice_params(
    space_group_indices: paddle.Tensor,
    conventional_lattice_lengths: Optional[paddle.Tensor] = None,
    conventional_lattice_angles: Optional[paddle.Tensor] = None,
    conventional_lattice_matrix: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """
    从 conventional 晶格参数计算 primitive 晶格矩阵。
    """
    if conventional_lattice_matrix is None:
        conventional_lattice_matrix = lattice_params_to_matrix_paddle(
            conventional_lattice_lengths, conventional_lattice_angles
        )
    P_matrices = global_vars.conventional_to_primitive_P_matrices[space_group_indices]
    return paddle.bmm(P_matrices, conventional_lattice_matrix)

def construct_fully_connected_graphs_with_periodic_boundaries(
    cart_coords: paddle.Tensor,
    lattice_matrix: paddle.Tensor,
    num_nodes_per_crystal: paddle.Tensor,
    break_minimum_edge_ties: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    构建周期边界条件下的全连接图（最小图像约定）。
    """
    batch_size = num_nodes_per_crystal.shape[0]
    atom_pos = cart_coords

    num_atoms_per_crystal_sqr = (num_nodes_per_crystal ** 2).cast(paddle.int64)

    first_node_index_per_crystal = (
        paddle.cumsum(num_nodes_per_crystal, axis=0) - num_nodes_per_crystal
    )
    first_node_index_per_crystal_expand = paddle.repeat_interleave(
        first_node_index_per_crystal, num_atoms_per_crystal_sqr
    )
    num_atoms_per_crystal_expand = paddle.repeat_interleave(
        num_nodes_per_crystal, num_atoms_per_crystal_sqr
    )

    num_atom_pairs = paddle.sum(num_atoms_per_crystal_sqr)
    index_sqr_offset = (
        paddle.cumsum(num_atoms_per_crystal_sqr, axis=0) - num_atoms_per_crystal_sqr
    )
    index_sqr_offset = paddle.repeat_interleave(index_sqr_offset, num_atoms_per_crystal_sqr)
    atom_count_sqr = paddle.arange(int(num_atom_pairs.item())) - index_sqr_offset

    destination_index = (
        paddle.floor_divide(atom_count_sqr, num_atoms_per_crystal_expand)
        + first_node_index_per_crystal_expand
    )
    source_index = (
        atom_count_sqr % num_atoms_per_crystal_expand
        + first_node_index_per_crystal_expand
    )
    map_edge_to_crystal = paddle.arange(batch_size).repeat_interleave(
        num_atoms_per_crystal_sqr, axis=0
    )

    n_edges_per_crystal_before_masking = num_atoms_per_crystal_sqr

    source_position = atom_pos[source_index]
    destination_position = atom_pos[destination_index]

    num_supercell_images = len(OFFSET_LIST)
    supercell_frac_offsets = paddle.to_tensor(OFFSET_LIST, dtype=paddle.float32)
    # (27, 3)
    batch_supercell_frac_offsets = supercell_frac_offsets.unsqueeze(0).expand(
        [batch_size, num_supercell_images, 3]
    )
    # (batch_size, 27, 3)
    pbc_frac_offsets_per_source_atom = paddle.repeat_interleave(
        batch_supercell_frac_offsets, n_edges_per_crystal_before_masking, axis=0
    )  # (num_atom_pairs, 27, 3)

    pbc_cart_offsets_per_source_atom = paddle.bmm(
        pbc_frac_offsets_per_source_atom,
        paddle.repeat_interleave(
            lattice_matrix, n_edges_per_crystal_before_masking, axis=0
        ),
    )  # (num_atom_pairs, 27, 3)

    destination_position = destination_position.unsqueeze(1).expand([-1, num_supercell_images, -1])
    source_position = (
        source_position.unsqueeze(1).expand([-1, num_supercell_images, -1])
        + pbc_cart_offsets_per_source_atom
    )

    source_index = source_index.unsqueeze(1).expand([-1, num_supercell_images])
    destination_index = destination_index.unsqueeze(1).expand([-1, num_supercell_images])
    map_edge_to_crystal = map_edge_to_crystal.unsqueeze(1).expand([-1, num_supercell_images])

    inter_atom_distances = (source_position - destination_position).norm(axis=-1)

    mask = paddle.logical_or(
        source_index != destination_index,
        inter_atom_distances > 1e-5,
    )

    destination_index = destination_index[mask]
    source_index = source_index[mask]
    map_edge_to_crystal = map_edge_to_crystal[mask]
    pbc_frac_offsets_per_source_atom = pbc_frac_offsets_per_source_atom.reshape(
        [-1, 3]
    )[mask.reshape([-1])]
    pbc_cart_offsets_per_source_atom = pbc_cart_offsets_per_source_atom.reshape(
        [-1, 3]
    )[mask.reshape([-1])]

    source_position_flat = atom_pos[source_index]
    destination_position_flat = atom_pos[destination_index]
    inter_atom_distances = (
        destination_position_flat
        - source_position_flat
        - pbc_cart_offsets_per_source_atom
    ).norm(axis=-1)

    (
        destination_index,
        source_index,
        pbc_frac_offsets_per_source_atom,
        num_edges_per_crystal,
    ) = get_smallest_edge_per_primal_node_pair(
        dst_idx=destination_index,
        src_idx=source_index,
        map_edge_to_crystal=map_edge_to_crystal,
        inter_atom_distances=inter_atom_distances,
        pbc_frac_offsets_per_source_atom=pbc_frac_offsets_per_source_atom,
        num_nodes_per_crystal=num_nodes_per_crystal,
        break_minimum_edge_ties=break_minimum_edge_ties,
    )
    return (
        destination_index,
        source_index,
        pbc_frac_offsets_per_source_atom,
        num_edges_per_crystal,
    )

def get_smallest_edge_per_primal_node_pair(
    dst_idx: paddle.Tensor,
    src_idx: paddle.Tensor,
    map_edge_to_crystal: paddle.Tensor,
    inter_atom_distances: paddle.Tensor,
    pbc_frac_offsets_per_source_atom: paddle.Tensor,
    num_nodes_per_crystal: paddle.Tensor,
    break_minimum_edge_ties: bool,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """
    保留每对节点之间的最小边（最小图像约定）。
    """
    edges = paddle.stack([dst_idx, src_idx], axis=-1)
    unique_edges, map_edge_to_unique = paddle.unique(edges, axis=0, return_inverse=True)

    min_inter_atom_distances, smallest_edge_idxs = scatter_min_with_argmin(
        src=inter_atom_distances,
        index=map_edge_to_unique,
        dim_size=unique_edges.shape[0],
    )

    batch_size = num_nodes_per_crystal.shape[0]

    if break_minimum_edge_ties:
        edges = edges[smallest_edge_idxs]
        pbc_frac_offsets_per_source_atom = pbc_frac_offsets_per_source_atom[smallest_edge_idxs]
        num_edges_per_crystal = (num_nodes_per_crystal ** 2).cast(paddle.int64)
    else:
        cutoff_distances = 1e-4 + min_inter_atom_distances[map_edge_to_unique]
        keep_edge_mask = inter_atom_distances < cutoff_distances
        indices_to_keep = paddle.nonzero(keep_edge_mask).reshape([-1])

        edges = edges[indices_to_keep]
        pbc_frac_offsets_per_source_atom = pbc_frac_offsets_per_source_atom[indices_to_keep]

        num_edges_per_crystal = scatter(
            src=paddle.ones([edges.shape[0]], dtype=paddle.float32),
            index=map_edge_to_crystal[indices_to_keep],
            dim_size=batch_size,
            reduce="sum",
        ).cast(paddle.int64)

    dst_idx = edges[:, 0]
    src_idx = edges[:, 1]
    return dst_idx, src_idx, pbc_frac_offsets_per_source_atom, num_edges_per_crystal

def ocp_get_pbc_distances(
    coords: paddle.Tensor,
    source_id: paddle.Tensor,
    destination_id: paddle.Tensor,
    lattice: paddle.Tensor,
    pbc_frac_offsets_per_source_node: paddle.Tensor,
    num_edges_per_crystal: paddle.Tensor,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
) -> dict:
    """计算 PBC 下原子间距离。"""
    neighbors = num_edges_per_crystal.cast(paddle.int64)
    lattice = paddle.repeat_interleave(lattice, neighbors, axis=0)
    offsets = (
        pbc_frac_offsets_per_source_node.cast(paddle.float32)
        .unsqueeze(1)
        .bmm(lattice.cast(paddle.float32))
        .reshape([-1, 3])
    )
    distance_vectors = coords[source_id] + offsets - coords[destination_id]
    distances = distance_vectors.norm(axis=-1)
    edge_index = paddle.stack([source_id, destination_id], axis=0)

    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors
    if return_offsets:
        out["offsets"] = offsets
    return out


def batched_convert_asu_frac_coords_to_primitive_cartesian_coords(
    asu_frac_coords: paddle.Tensor,
    asu_element_indices: paddle.Tensor,
    asu_wyckoff_indices: paddle.Tensor,
    n_coords_per_asu: paddle.Tensor,
    conventional_lattice_matrix: paddle.Tensor,
    space_group_indices: paddle.Tensor,
    return_cartesian_coords: bool = True,
    return_node_is_original: bool = False,
    map_frac_coords_to_0_1_unit_cell: bool = True,
    get_primitive_cell: bool = True,
) -> tuple:
    """
    将非对称单元分数坐标批量展开到原始晶胞笛卡尔坐标。
    """
    batch_size = n_coords_per_asu.shape[0]
    num_asu_nodes_per_crystal = n_coords_per_asu

    conventional_to_primitive_transformations = (
        global_vars.conventional_to_primitive_invP_matrices[space_group_indices]
    )

    padded_gc_mats = global_vars.padded_general_wyckoff_matrices[space_group_indices]
    padded_gc_trans = global_vars.padded_general_wyckoff_translations[space_group_indices]
    padded_gc_mask = global_vars.padded_general_wyckoff_ops_mask[space_group_indices]
    general_wyckoff_multiplicity_per_crystal = padded_gc_mask.cast(paddle.int64).sum(axis=1)

    # 展开操作到每个 ASU atom
    n_total_asu = asu_frac_coords.shape[0]

    def _repeat_interleave_along_asu(tensor_B_X):
        if tensor_B_X.dtype == paddle.bool:
            return paddle.repeat_interleave(
                tensor_B_X.cast(paddle.int32), num_asu_nodes_per_crystal, axis=0
            ).cast(paddle.bool)
        return paddle.repeat_interleave(
            tensor_B_X, num_asu_nodes_per_crystal, axis=0
        )

    padded_gc_mats = _repeat_interleave_along_asu(padded_gc_mats)
    # (n_asu_atoms, 192, 3, 3)
    padded_gc_trans = _repeat_interleave_along_asu(padded_gc_trans)
    # (n_asu_atoms, 192, 1, 3)
    padded_gc_mask = _repeat_interleave_along_asu(padded_gc_mask)
    # (n_asu_atoms, 192)

    stacked_gc_mats = padded_gc_mats[padded_gc_mask].reshape([-1, 3, 3])
    # (n_orbited_atoms, 3, 3)
    stacked_gc_trans = padded_gc_trans[
        padded_gc_mask.unsqueeze(-1).unsqueeze(-1).expand_as(padded_gc_trans)
    ].reshape([-1, 1, 3])
    # (n_orbited_atoms, 1, 3)

    # orbit ASU atoms 到 conventional cell
    asu_multiplicity_per_asu_atom = general_wyckoff_multiplicity_per_crystal.repeat_interleave(
        num_asu_nodes_per_crystal, axis=0
    )
    # (n_asu_atoms,)
    asu_frac_coords_repeated = asu_frac_coords.repeat_interleave(
        asu_multiplicity_per_asu_atom, axis=0
    ).unsqueeze(1)
    # (n_orbited_atoms, 1, 3)

    conventional_frac_coords = (
        paddle.bmm(asu_frac_coords_repeated, stacked_gc_mats) + stacked_gc_trans
    )  # (n_orbited_atoms, 1, 3)
    if map_frac_coords_to_0_1_unit_cell:
        conventional_frac_coords = conventional_frac_coords % 1.0

    if get_primitive_cell:
        stacked_c2p = paddle.repeat_interleave(
            conventional_to_primitive_transformations,
            num_asu_nodes_per_crystal * general_wyckoff_multiplicity_per_crystal,
            axis=0,
        )
        # (n_orbited_atoms, 3, 3)
        primitive_frac_coords = paddle.bmm(
            conventional_frac_coords, stacked_c2p
        ).squeeze(1)  # (n_orbited_atoms, 3)
    else:
        primitive_frac_coords = conventional_frac_coords.reshape([-1, 3])

    if map_frac_coords_to_0_1_unit_cell:
        primitive_frac_coords = primitive_frac_coords % 1.0

    map_node_to_crystal = paddle.arange(batch_size).repeat_interleave(
        num_asu_nodes_per_crystal * general_wyckoff_multiplicity_per_crystal, axis=0
    )  # (n_orbited_atoms,)

    # 去重重叠原子
    orbit_size_per_asu = asu_multiplicity_per_asu_atom
    orbit_size_sqr = (orbit_size_per_asu ** 2).cast(paddle.int64)

    first_asu_atom_index_per_orbit = (
        paddle.cumsum(orbit_size_per_asu, axis=0) - orbit_size_per_asu
    )
    first_idx_expand = paddle.repeat_interleave(first_asu_atom_index_per_orbit, orbit_size_sqr)
    orbit_size_expand = paddle.repeat_interleave(orbit_size_per_asu, orbit_size_sqr)

    if return_node_is_original:
        node_is_original = paddle.zeros(
            [primitive_frac_coords.shape[0]], dtype=paddle.bool
        )
        node_is_original[first_asu_atom_index_per_orbit] = True

    num_atom_pairs = orbit_size_sqr.sum().item()
    index_sqr_offset = (
        paddle.cumsum(orbit_size_sqr, axis=0) - orbit_size_sqr
    ).repeat_interleave(orbit_size_sqr)
    atom_pair_indices = paddle.arange(int(num_atom_pairs)) - index_sqr_offset

    row_index = paddle.floor_divide(atom_pair_indices, orbit_size_expand) + first_idx_expand
    col_index = atom_pair_indices % orbit_size_expand + first_idx_expand

    row_coords = primitive_frac_coords[row_index]
    col_coords = primitive_frac_coords[col_index]

    overlapping_mask = paddle.all(
        paddle.abs(
            (row_coords - col_coords + 0.5) % 1.0 - 0.5
        ) < 1e-6, axis=1
    )

    overlapping_col = col_index[overlapping_mask]
    overlapping_row = row_index[overlapping_mask]

    min_col_per_row = _scatter_min_indices_gpu_safe(
        overlapping_row, overlapping_col, primitive_frac_coords.shape[0]
    )
    unique_non_overlapping_atom_indices = paddle.unique(min_col_per_row)

    primitive_frac_coords = primitive_frac_coords[unique_non_overlapping_atom_indices]
    map_node_to_crystal = map_node_to_crystal[unique_non_overlapping_atom_indices]
    if return_node_is_original:
        node_is_original = node_is_original[unique_non_overlapping_atom_indices]

    num_prim_nodes_per_crystal = _segment_coo_sum_gpu_safe(
        src=paddle.ones([map_node_to_crystal.shape[0]], dtype=paddle.int64),
        index=map_node_to_crystal,
        dim_size=batch_size,
    )
    assert num_prim_nodes_per_crystal.shape[0] == batch_size

    # 从 ASU 获取 Wyckoff 和元素索引
    map_prim_to_asu = paddle.arange(asu_frac_coords.shape[0]).repeat_interleave(
        asu_multiplicity_per_asu_atom, axis=0
    )[unique_non_overlapping_atom_indices]
    primitive_wyckoff_indices = asu_wyckoff_indices[map_prim_to_asu]
    primitive_element_indices = asu_element_indices[map_prim_to_asu]
    asu_frac_coords_of_prim_atoms = asu_frac_coords[map_prim_to_asu]

    # 计算 primitive 晶格矩阵
    primitive_lattice_matrix = primitive_lattice_matrix_from_conventional_lattice_params(
        space_group_indices=space_group_indices,
        conventional_lattice_matrix=conventional_lattice_matrix,
    )

    if return_cartesian_coords:
        out_coords = frac_to_cart_coords(
            primitive_frac_coords,
            num_prim_nodes_per_crystal,
            lattice_matrix=primitive_lattice_matrix,
        )
    else:
        out_coords = primitive_frac_coords

    if return_node_is_original:
        return (
            out_coords,
            primitive_element_indices,
            primitive_wyckoff_indices,
            num_prim_nodes_per_crystal,
            primitive_lattice_matrix,
            map_prim_to_asu,
            node_is_original,
            asu_frac_coords_of_prim_atoms,
        )
    else:
        return (
            out_coords,
            primitive_element_indices,
            primitive_wyckoff_indices,
            num_prim_nodes_per_crystal,
            primitive_lattice_matrix,
            map_prim_to_asu,
            asu_frac_coords_of_prim_atoms,
        )
