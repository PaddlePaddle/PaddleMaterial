"""
扩散过程核心工具函数：凸包判定、ASU 坐标包装、分数坐标投影等。
"""
from typing import Union, Tuple

import numpy as np
import paddle
from scipy.spatial import ConvexHull

import ppmat.models.sgequidiff.global_vars as global_vars
from ppmat.models.sgequidiff.constants import MAX_WYCKOFF_SITES, OFFSET_LIST
from ppmat.models.sgequidiff.data_utils import batched_convert_asu_frac_coords_to_primitive_cartesian_coords
from ppmat.models.sgequidiff.scatter_utils import safe_scatter as paddle_scatter


def _scatter_argmax_gpu_safe(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim_size: int,
) -> paddle.Tensor:
    """
    对每个 group 返回使 src 最大的原始元素全局索引。
    纯 Paddle 实现，GPU 安全（等价于 torch_scatter.scatter_max 的 argmax 版本）。
    """
    n = src.shape[0]
    sort_key = index.cast(paddle.float64) * (float(n) + 1.0) - src.cast(paddle.float64)
    sorted_order = paddle.argsort(sort_key)
    sorted_index = index[sorted_order]

    group_first = paddle.concat([
        paddle.to_tensor([True], dtype=paddle.bool),
        sorted_index[1:] != sorted_index[:-1],
    ])
    first_pos = paddle.nonzero(group_first).squeeze(1)
    group_ids = sorted_index[first_pos]
    argmax_orig_idx = sorted_order[first_pos]

    result = paddle.zeros([dim_size], dtype=paddle.int64)
    result = paddle.scatter(result, group_ids, argmax_orig_idx)
    return result

def _scatter_min_indices_gpu_safe(
    ov_row: paddle.Tensor,
    ov_col: paddle.Tensor,
    n_total: int,
) -> paddle.Tensor:
    """
    返回每个 ov_row 对应的最小 ov_col 值。GPU 安全。
    不在 ov_row 中的行默认返回自身索引（自映射）。
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


def atoms_are_in_hull(
    supercell_frac_coords: paddle.Tensor,
    hull_equations: paddle.Tensor,
    epsilon: float = -1e-5,
) -> paddle.Tensor:
    """
    判断 supercell 中各点是否在 Wyckoff 形状 hull 内。
    """
    n_atoms, n_images, _ = supercell_frac_coords.shape
    n_shapes = hull_equations.shape[1]
    n_bounds = hull_equations.shape[2]

    coords_flat = supercell_frac_coords.reshape([-1, 3])
    hulls_expanded = hull_equations.unsqueeze(1).expand(
        [n_atoms, n_images, n_shapes, n_bounds, 4]
    ).reshape([n_atoms * n_images, n_shapes, n_bounds, 4])

    normals = hulls_expanded[..., :3]
    offsets = hulls_expanded[..., 3]

    # (n*img, 1, 1, 3) @ (n*img, n_shapes, n_bounds, 3) -> sum
    dots = (
        coords_flat[:, None, None, :] * normals
    ).sum(axis=-1)
    # (n_atoms * n_images, n_shapes, n_bounds)

    inside = (dots < -offsets - epsilon).all(axis=-1)
    # (n_atoms * n_images, n_shapes)
    return inside.reshape([n_atoms, n_images, n_shapes])

def get_infinite_wyckoff_shape_hull_equations() -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    计算每个 0/1/2D Wyckoff 位置（无边界）的 hull 方程。
    """
    asu_wyckoff_dict = global_vars.asu_wyckoff_dict
    max_num_shape_bounds = max(2, 5, global_vars.max_simplicial_hull_facets)

    padded_hull_equations = -1.0 * paddle.nn.functional.one_hot(
        paddle.to_tensor(3), num_classes=4
    ).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
        [230, MAX_WYCKOFF_SITES, global_vars.max_shapes_per_wyckoff, max_num_shape_bounds, 4]
    ).cast(paddle.float32).clone()

    mask_padded_hull_equations = paddle.zeros(
        [230, MAX_WYCKOFF_SITES, global_vars.max_shapes_per_wyckoff],
        dtype=paddle.bool,
    )

    for sg_num in range(1, 231):
        sg_dict = asu_wyckoff_dict[str(sg_num)]
        for wp_idx, wp_letter in enumerate(sg_dict["ordered_wyckoff_letters"]):
            wp_dict = sg_dict[wp_letter]
            dim = int(wp_dict["dim"])

            if dim == 0:
                shape_idx = 0
                mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                vertex = wp_dict["vertices"].astype("float64").reshape(-1)
                eps = 1e-4
                hull_equations = np.zeros((6, 4))
                hull_equations[:3, :3] = np.eye(3)
                hull_equations[3:, :3] = -np.eye(3)
                hull_equations[:3, -1] = -(vertex + eps)
                hull_equations[3:, -1] = -(-vertex + eps)
                padded_hull_equations[sg_num - 1, wp_idx, shape_idx, :6] = paddle.to_tensor(
                    hull_equations, dtype=paddle.float32
                )

            elif dim == 1:
                for shape_idx, line_segment in enumerate(wp_dict["vertices"].astype("float64")):
                    mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                    line_dir = line_segment[1] - line_segment[0]
                    line_dir /= np.linalg.norm(line_dir)

                    # 获取两个互相正交的法向量
                    normal1 = np.cross(line_dir, np.random.rand(3))[np.newaxis, :]
                    normal1 /= np.linalg.norm(normal1)
                    normal2 = np.cross(line_dir, normal1)
                    normal2 /= np.linalg.norm(normal2)

                    line_dir_col = line_dir.reshape(3, 1)
                    proj = line_dir_col @ line_dir_col.T
                    I_minus_P = np.eye(3) - proj

                    hull_equations = np.zeros((4, 4))
                    hull_equations[:, :3] = np.concatenate([
                        normal1 @ I_minus_P,
                        -normal1 @ I_minus_P,
                        normal2 @ I_minus_P,
                        -normal2 @ I_minus_P,
                    ], axis=0)
                    hull_equations[:, -1] = -1 * np.concatenate([
                        1e-4 + normal1 @ I_minus_P @ line_segment[0].reshape(3, 1),
                        1e-4 - normal1 @ I_minus_P @ line_segment[0].reshape(3, 1),
                        1e-4 + normal2 @ I_minus_P @ line_segment[0].reshape(3, 1),
                        1e-4 - normal2 @ I_minus_P @ line_segment[0].reshape(3, 1),
                    ]).reshape(4)
                    padded_hull_equations[sg_num - 1, wp_idx, shape_idx, :4] = paddle.to_tensor(
                        hull_equations, dtype=paddle.float32
                    )

            elif dim == 2:
                for shape_idx, polygon_vertices in enumerate(wp_dict["vertices"]):
                    mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                    polygon_vertices = polygon_vertices.astype("float64")
                    ab = polygon_vertices[1] - polygon_vertices[0]
                    ac = polygon_vertices[2] - polygon_vertices[0]
                    normal = np.cross(ab, ac)
                    normal /= np.linalg.norm(normal)

                    normal_col = normal.reshape(3, 1)
                    proj = global_vars._project_onto_2d_subspace(
                        paddle.to_tensor(polygon_vertices, dtype=paddle.float32)
                    ).numpy().T

                    I_minus_P = np.eye(3) - proj
                    hull_equations = np.zeros((2, 4))
                    hull_equations[:, :3] = np.concatenate([
                        normal_col.T @ I_minus_P,
                        -normal_col.T @ I_minus_P,
                    ], axis=0)
                    hull_equations[:, -1] = -1.0 * np.concatenate([
                        1e-4 + normal_col.T @ I_minus_P @ polygon_vertices[0].reshape(3, 1),
                        1e-4 - normal_col.T @ I_minus_P @ polygon_vertices[0].reshape(3, 1),
                    ], axis=0).reshape(2)
                    padded_hull_equations[sg_num - 1, wp_idx, shape_idx, :2] = paddle.to_tensor(
                        hull_equations, dtype=paddle.float32
                    )

    return padded_hull_equations, mask_padded_hull_equations

def get_wyckoff_shape_hull_equations() -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    计算每个 0/1/2/3D Wyckoff 形状的 hull 方程（包含有界形状）。
    """
    asu_wyckoff_dict = global_vars.asu_wyckoff_dict
    max_num_shape_bounds = max(2, 5, global_vars.max_simplicial_hull_facets)

    padded_hull_equations = -1.0 * paddle.nn.functional.one_hot(
        paddle.to_tensor(3), num_classes=4
    ).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
        [230, MAX_WYCKOFF_SITES, global_vars.max_shapes_per_wyckoff, max_num_shape_bounds, 4]
    ).cast(paddle.float32).clone()

    mask_padded_hull_equations = paddle.zeros(
        [230, MAX_WYCKOFF_SITES, global_vars.max_shapes_per_wyckoff],
        dtype=paddle.bool,
    )

    for sg_num in range(1, 231):
        sg_dict = asu_wyckoff_dict[str(sg_num)]
        for wp_idx, wp_letter in enumerate(sg_dict["ordered_wyckoff_letters"]):
            wp_dict = sg_dict[wp_letter]
            dim = int(wp_dict["dim"])

            if dim == 0:
                shape_idx = 0
                mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                vertex = wp_dict["vertices"].astype("float64").reshape(-1)
                eps = 1e-4
                hull_equations = np.zeros((6, 4))
                hull_equations[:3, :3] = np.eye(3)
                hull_equations[3:, :3] = -np.eye(3)
                hull_equations[:3, -1] = -(vertex + eps)
                hull_equations[3:, -1] = -(-vertex + eps)
                padded_hull_equations[sg_num - 1, wp_idx, shape_idx, :6] = paddle.to_tensor(
                    hull_equations, dtype=paddle.float32
                )

            elif dim == 1:
                for shape_idx, line_segment in enumerate(wp_dict["vertices"].astype("float64")):
                    mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                    eps = 1e-4
                    line_dir = line_segment[1] - line_segment[0]
                    line_dir /= np.linalg.norm(line_dir)
                    normal1 = np.cross(line_dir, np.random.rand(3))[np.newaxis, :]
                    normal1 /= np.linalg.norm(normal1)
                    normal2 = np.cross(line_dir, normal1)
                    normal2 /= np.linalg.norm(normal2)
                    p1 = eps * normal1
                    p2 = eps * normal2
                    bounding_polytope = np.concatenate([
                        line_segment + (p1 + p2),
                        line_segment + (p1 - p2),
                        line_segment + (-p1 + p2),
                        line_segment + (-p1 - p2),
                    ], axis=0)  # (8, 3)
                    hull = ConvexHull(bounding_polytope)
                    n_facets = hull.equations.shape[0]
                    padded_hull_equations[sg_num - 1, wp_idx, shape_idx, :n_facets] = paddle.to_tensor(
                        hull.equations, dtype=paddle.float32
                    )

            elif dim == 2:
                for shape_idx, polygon_vertices in enumerate(wp_dict["vertices"]):
                    mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                    polygon_vertices = polygon_vertices.astype("float64")
                    ab = polygon_vertices[1] - polygon_vertices[0]
                    ac = polygon_vertices[2] - polygon_vertices[0]
                    normal = np.cross(ab, ac)
                    normal /= np.linalg.norm(normal)
                    bounding = np.concatenate([
                        polygon_vertices + 1e-4 * normal,
                        polygon_vertices - 1e-4 * normal,
                    ], axis=0)
                    hull = ConvexHull(bounding)
                    n_facets = hull.equations.shape[0]
                    padded_hull_equations[sg_num - 1, wp_idx, shape_idx, :n_facets] = paddle.to_tensor(
                        hull.equations, dtype=paddle.float32
                    )

            elif dim == 3:
                shape_idx = 0
                mask_padded_hull_equations[sg_num - 1, wp_idx, shape_idx] = True
                hull = ConvexHull(wp_dict["vertices"].astype("float64"))
                n_facets = hull.equations.shape[0]
                padded_hull_equations[sg_num - 1, wp_idx, shape_idx, :n_facets] = paddle.to_tensor(
                    hull.equations, dtype=paddle.float32
                )

    return padded_hull_equations, mask_padded_hull_equations


@paddle.no_grad()
def wrap_frac_coords_into_asu(
    frac_coords: paddle.Tensor,
    wyckoff_indices: paddle.Tensor,
    space_group_indices: paddle.Tensor,
    num_atoms_per_asu: paddle.Tensor,
    hull_equations: paddle.Tensor,
    hull_equations_mask: paddle.Tensor,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    将加噪后的分数坐标通过群操作映射回规范 ASU。
    """
    n_asu_atoms = frac_coords.shape[0]
    frac_coords = frac_coords % 1.0

    (
        conventional_cell_frac_coords,
        _,
        _,
        _,
        _,
        map_conventional_to_asu_coord,
        _,
    ) = batched_convert_asu_frac_coords_to_primitive_cartesian_coords(
        asu_frac_coords=frac_coords,
        asu_element_indices=paddle.zeros_like(wyckoff_indices),
        asu_wyckoff_indices=wyckoff_indices,
        n_coords_per_asu=num_atoms_per_asu,
        conventional_lattice_matrix=paddle.zeros(
            [space_group_indices.shape[0], 3, 3], dtype=paddle.float32
        ),
        space_group_indices=space_group_indices,
        return_cartesian_coords=False,
        get_primitive_cell=False,
    )

    # 超胞展开 + inside/outside 测试
    supercell_frac_translations = paddle.to_tensor(OFFSET_LIST, dtype=paddle.float32)
    supercell_frac_coords = (
        conventional_cell_frac_coords.unsqueeze(1) + supercell_frac_translations.unsqueeze(0)
    )
    hull_eq_expanded = hull_equations[map_conventional_to_asu_coord]
    wyckoff_shape_exists = hull_equations_mask[map_conventional_to_asu_coord]

    supercell_in_wyckoff = (
        atoms_are_in_hull(supercell_frac_coords, hull_eq_expanded, epsilon=-1e-5)
        & wyckoff_shape_exists.unsqueeze(1)
    )

    supercell_in_any_wyckoff = supercell_in_wyckoff.any(axis=-1)
    # (n_conventional_atoms, 27)
    conv_atom_has_image_in_asu = supercell_in_any_wyckoff.any(axis=-1)

    indices_of_conv_atoms_in_asu = _scatter_argmax_gpu_safe(
        src=conv_atom_has_image_in_asu.cast(paddle.float32),
        index=map_conventional_to_asu_coord,
        dim_size=n_asu_atoms,
    )
    # (n_asu_atoms,)
    assert indices_of_conv_atoms_in_asu.shape[0] == n_asu_atoms

    # 选出每个 ASU 原子的代表性 conventional 原子的超胞坐标
    supercell_frac_coords_in_asu = supercell_frac_coords[indices_of_conv_atoms_in_asu]
    # (n_asu_atoms, 27, 3)
    asu_atom_is_inside = supercell_in_wyckoff[indices_of_conv_atoms_in_asu]
    # (n_asu_atoms, 27, max_shapes)
    supercell_in_any_in_asu = supercell_in_any_wyckoff[indices_of_conv_atoms_in_asu]
    # (n_asu_atoms, 27)

    lattice_translation_into_asu_idx = paddle.argmax(
        supercell_in_any_in_asu.cast(paddle.float32), axis=-1
    )
    # (n_asu_atoms,)

    atom_indices = paddle.arange(n_asu_atoms)
    wrapped_asu_frac_coords = supercell_frac_coords_in_asu[
        atom_indices, lattice_translation_into_asu_idx
    ]
    # (n_asu_atoms, 3)
    wrapped_asu_wyckoff_shape_indices = paddle.argmax(
        asu_atom_is_inside[atom_indices, lattice_translation_into_asu_idx].cast(paddle.float32),
        axis=-1,
    )
    # (n_asu_atoms,)

    return wrapped_asu_frac_coords, wrapped_asu_wyckoff_shape_indices

@paddle.no_grad()
def p_asu_wrapped_normal(
    noisy_frac_coord: paddle.Tensor,
    conventional_frac_coords: paddle.Tensor,
    map_conventional_to_asu_frac_coords: paddle.Tensor,
    n_lattice_translations: int = 5,
    sigma: Union[float, paddle.Tensor] = 1.0,
) -> paddle.Tensor:
    """
    在非对称单元内计算各等效点的各向同性高斯和。不含预因子 1/(2pi*sigma)。
    """
    t = paddle.arange(-n_lattice_translations, n_lattice_translations + 1, dtype=paddle.float32)
    translations = paddle.stack(
        paddle.meshgrid(t, t, t, indexing="ij"), axis=-1
    ).reshape([-1, 3])

    noisy_x_minus_gt = (
        noisy_frac_coord[map_conventional_to_asu_frac_coords].unsqueeze(1)
        - (conventional_frac_coords.unsqueeze(1) + translations.unsqueeze(0))
    )
    # (n_conventional_atoms, n_lattice_translations, 3)

    diff_sq = (noisy_x_minus_gt ** 2).sum(axis=-1)
    gaussian = paddle.exp(-diff_sq / (2 * sigma ** 2))
    p = gaussian.sum(axis=1)
    # (n_conventional_atoms,)

    # scatter_add: conventional → ASU
    p_asu = paddle_scatter(
        src=p,
        index=map_conventional_to_asu_frac_coords,
        dim=0,
        dim_size=noisy_frac_coord.shape[0],
        reduce="sum",
    )
    # (n_asu_atoms,)
    return p_asu

@paddle.no_grad()
def d_log_p_asu_wrapped_normal(
    noisy_frac_coord: paddle.Tensor,
    conventional_frac_coords: paddle.Tensor,
    map_conventional_to_asu_frac_coords: paddle.Tensor,
    n_lattice_translations: int = 5,
    sigma: Union[float, paddle.Tensor] = 1.0,
) -> paddle.Tensor:
    """
    计算 ASU 包裹正态分布对数概率的梯度（即 ground truth score）。
    """
    if isinstance(sigma, float):
        sigma_conv = paddle.full([conventional_frac_coords.shape[0], 1], sigma)
    else:
        sigma_conv = sigma[map_conventional_to_asu_frac_coords].unsqueeze(1)
        # (n_conventional_atoms, 1)

    t = paddle.arange(-n_lattice_translations, n_lattice_translations + 1, dtype=paddle.float32)
    translations = paddle.stack(
        paddle.meshgrid(t, t, t, indexing="ij"), axis=-1
    ).reshape([-1, 3])
    # (n_lattice_translations, 3)

    noisy_x_minus_gt = (
        noisy_frac_coord[map_conventional_to_asu_frac_coords].unsqueeze(1)
        - (conventional_frac_coords.unsqueeze(1) + translations.unsqueeze(0))
    )
    # (n_conventional_atoms, n_lattice_translations, 3)

    diff_sq = (noisy_x_minus_gt ** 2).sum(axis=-1, keepdim=True)
    # (n_conventional_atoms, n_lattice_translations, 1)
    gaussian = paddle.exp(-diff_sq / (2 * sigma_conv.unsqueeze(-1) ** 2))
    numerator = -(gaussian * noisy_x_minus_gt).sum(axis=1)
    # (n_conventional_atoms, 3)

    n_asu_atoms = noisy_frac_coord.shape[0]
    numerator_asu = paddle.stack(
        [
            paddle_scatter(
                src=numerator[:, d],
                index=map_conventional_to_asu_frac_coords,
                dim=0,
                dim_size=n_asu_atoms,
                reduce="sum",
            )
            for d in range(3)
        ],
        axis=1,
    )
    # (n_asu_atoms, 3)

    denominator = p_asu_wrapped_normal(
        noisy_frac_coord,
        conventional_frac_coords,
        map_conventional_to_asu_frac_coords,
        n_lattice_translations,
        sigma_conv,
    ) * (sigma if isinstance(sigma, float) else sigma ** 2)
    # (n_asu_atoms,)
    return numerator_asu / denominator.unsqueeze(-1)

def get_space_group_ops_and_conventional_atoms(
    frac_coords: paddle.Tensor,
    element_indices: paddle.Tensor,
    wyckoff_indices: paddle.Tensor,
    space_group_indices: paddle.Tensor,
    n_atoms_per_xtal: paddle.Tensor,
) -> tuple:
    """
    获取空间群操作（mod 晶格平移）和无重复 conventional cell 原子。
    """
    batch_size = n_atoms_per_xtal.shape[0]

    padded_gc_mats = global_vars.padded_general_wyckoff_matrices[space_group_indices]
    padded_gc_inv_mats = global_vars.padded_inverse_general_wyckoff_matrices[space_group_indices]
    padded_gc_trans = global_vars.padded_general_wyckoff_translations[space_group_indices]
    padded_gc_mask = global_vars.padded_general_wyckoff_ops_mask[space_group_indices]
    # all (B, 192, ...)

    general_wyckoff_multiplicity = padded_gc_mask.cast(paddle.int64).sum(axis=1)

    padded_gc_mats = paddle.repeat_interleave(padded_gc_mats, n_atoms_per_xtal, axis=0)
    padded_gc_inv_mats = paddle.repeat_interleave(padded_gc_inv_mats, n_atoms_per_xtal, axis=0)
    padded_gc_trans = paddle.repeat_interleave(padded_gc_trans, n_atoms_per_xtal, axis=0)
    padded_gc_mask = paddle.repeat_interleave(
        padded_gc_mask.cast(paddle.int32), n_atoms_per_xtal, axis=0
    ).cast(paddle.bool)

    stacked_mats = padded_gc_mats[padded_gc_mask].reshape([-1, 3, 3])
    stacked_inv_mats = padded_gc_inv_mats[padded_gc_mask].reshape([-1, 3, 3])
    stacked_trans = padded_gc_trans[
        padded_gc_mask.unsqueeze(-1).unsqueeze(-1).expand_as(padded_gc_trans)
    ].reshape([-1, 1, 3])

    mult_per_asu_atom = general_wyckoff_multiplicity.repeat_interleave(n_atoms_per_xtal, axis=0)

    frac_coords_repeated = frac_coords.repeat_interleave(
        mult_per_asu_atom, axis=0
    ).unsqueeze(1)

    conventional_frac_coords_with_dupes = (
        paddle.bmm(frac_coords_repeated, stacked_mats) + stacked_trans
    ) % 1.0
    conventional_frac_coords = conventional_frac_coords_with_dupes.reshape([-1, 3])

    # de-duplicate
    orbit_size_per_asu = mult_per_asu_atom  # (n_asu_atoms,)
    orbit_size_sqr = (orbit_size_per_asu ** 2).cast(paddle.int64)
    first_idx = paddle.cumsum(orbit_size_per_asu, axis=0) - orbit_size_per_asu
    first_idx_expand = paddle.repeat_interleave(first_idx, orbit_size_sqr)
    orbit_size_expand = paddle.repeat_interleave(orbit_size_per_asu, orbit_size_sqr)

    n_pairs = int(orbit_size_sqr.sum().item())
    offset = (paddle.cumsum(orbit_size_sqr, axis=0) - orbit_size_sqr).repeat_interleave(orbit_size_sqr)
    pair_ids = paddle.arange(n_pairs) - offset
    row_ids = paddle.floor_divide(pair_ids, orbit_size_expand) + first_idx_expand
    col_ids = pair_ids % orbit_size_expand + first_idx_expand

    row_coords = conventional_frac_coords[row_ids]
    col_coords = conventional_frac_coords[col_ids]
    overlapping = paddle.all(
        paddle.abs((row_coords - col_coords + 0.5) % 1.0 - 0.5) < 1e-6, axis=1
    )

    ov_col = col_ids[overlapping]
    ov_row = row_ids[overlapping]
    min_col_per_row = _scatter_min_indices_gpu_safe(ov_row, ov_col, conventional_frac_coords.shape[0])

    unique_non_overlapping_atom_indices, inverse_indices = paddle.unique(
        min_col_per_row, return_inverse=True
    )

    conventional_frac_coords = conventional_frac_coords[unique_non_overlapping_atom_indices]

    map_conv_to_asu_with_dupes = paddle.arange(frac_coords.shape[0]).repeat_interleave(
        orbit_size_per_asu, axis=0
    )
    map_conventional_to_asu_atom = map_conv_to_asu_with_dupes[unique_non_overlapping_atom_indices]
    conventional_wyckoff_indices = wyckoff_indices[map_conventional_to_asu_atom]
    conventional_element_indices = element_indices[map_conventional_to_asu_atom]

    return (
        stacked_inv_mats,          # A_ops: (n_general_wyckoff_ops, 3, 3)
        stacked_trans,             # t_ops: (n_general_wyckoff_ops, 1, 3)
        inverse_indices,           # (n_general_wyckoff_ops,)
        map_conv_to_asu_with_dupes,  # (n_general_wyckoff_ops,)
        conventional_wyckoff_indices,
        conventional_element_indices,
        conventional_frac_coords,
        unique_non_overlapping_atom_indices,
    )

def get_wyckoff_projected_gaussian_noise(
    space_group_indices: paddle.Tensor,
    wyckoff_indices: paddle.Tensor,
    wyckoff_shape_indices: paddle.Tensor,
    n_atoms_per_xtal: paddle.Tensor,
    sigma: Union[float, paddle.Tensor],
) -> paddle.Tensor:
    """
    生成投影到 Wyckoff 子空间的高斯噪声。
    """
    n_asu_atoms = wyckoff_indices.shape[0]
    unprojected_noise = sigma * paddle.randn([n_asu_atoms, 3])
    # (n_asu_atoms, 3)
    sg_per_atom = space_group_indices.repeat_interleave(n_atoms_per_xtal, axis=0)
    projection_matrices = global_vars.noise_projection_matrices[
        sg_per_atom, wyckoff_indices, wyckoff_shape_indices
    ]
    # (n_asu_atoms, 3, 3)
    projected_noise = paddle.bmm(
        unprojected_noise.unsqueeze(1), projection_matrices
    ).reshape([-1, 3])
    return projected_noise
