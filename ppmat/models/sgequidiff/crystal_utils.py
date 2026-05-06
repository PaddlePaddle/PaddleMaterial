"""
非对称单元的晶体工具函数。

"""
from typing import List, Tuple, Union

import paddle

import ppmat.models.sgequidiff.global_vars as global_vars


def vmappable_inside(
    x: paddle.Tensor,
    hull_equations: paddle.Tensor,
    epsilon: float = 1e-6,
) -> paddle.Tensor:
    """
    判断点 x 是否在由 hull_equations 定义的凸多面体内部。

    Args:
        x: shape (3,)
        hull_equations: shape (n_linear_shape_bounds, 4)

    Returns:
        shape (,) boolean Tensor
    """
    return paddle.all(hull_equations[:, :-1] @ x < -hull_equations[:, -1] - epsilon)

def is_inside(
    xs: paddle.Tensor,
    hull_equations: paddle.Tensor,
    epsilon: float = 1e-6,
) -> paddle.Tensor:
    """
    批量版本：判断多个点是否在凸多面体内部。
    替代原始的 torch.vmap(vmappable_inside, in_dims=(0, None), out_dims=0)

    Args:
        xs: shape (n_points, 3)
        hull_equations: shape (n_linear_shape_bounds, 4)

    Returns:
        shape (n_points,) boolean Tensor
    """
    # (n_points, n_bounds) = (n_points, 3) @ (3, n_bounds) < -(n_bounds,) - epsilon
    results = xs @ hull_equations[:, :3].T < -hull_equations[:, 3][None, :] - epsilon
    # (n_points, n_bounds)
    return results.all(axis=-1)  # (n_points,)

def uniformly_sample_point_in_asu_wyckoff_site(
    space_group_numbers: List[str],
    wyckoff_letters: List[str],
    dictionary_of_wyckoffs_in_asu: dict,
    dictionary_of_wyckoff_shape_decompositions: dict,
    device=None,
    finished_sampling_mask: paddle.Tensor = None,
    n_samples_per_wyckoff: int = 1,
    return_sampled_wyckoff_shape_indices: bool = False,
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
    """
    在非对称单元 Wyckoff 位置中均匀采样点。

    Returns:
        random_samples_in_wyckoffs: (n_crystals, n_samples_per_wyckoff, 3)
        sampled_wyckoff_shape_indices: (n_crystals, n_samples_per_wyckoff)
    """
    random_samples_in_wyckoffs = []
    sampled_wyckoff_shape_indices = []
    for i, (space_group_number, wyckoff_letter) in enumerate(
        zip(space_group_numbers, wyckoff_letters)
    ):
        wyckoff_position_dict = dictionary_of_wyckoffs_in_asu[space_group_number][wyckoff_letter]
        wyckoff_dof = int(wyckoff_position_dict["dim"])

        if wyckoff_dof == 1:
            lengths = paddle.to_tensor(
                dictionary_of_wyckoff_shape_decompositions[space_group_number][wyckoff_letter]["volumes"],
                dtype=paddle.float32,
            )  # (num_line_segments,)
            vertices = paddle.to_tensor(
                wyckoff_position_dict["vertices"].astype("float32")
            )
            # (num_line_segments, 2, 3)
            sampled_line_index = paddle.multinomial(
                lengths, num_samples=n_samples_per_wyckoff, replacement=True
            )
            vertices = vertices[sampled_line_index]
            sampled_wyckoff_shape_index = sampled_line_index

        elif wyckoff_dof == 2:
            wyckoff_shapes_decomp_dict = dictionary_of_wyckoff_shape_decompositions[
                space_group_number
            ][wyckoff_letter]

            facet_areas = paddle.to_tensor(
                wyckoff_shapes_decomp_dict["volumes"], dtype=paddle.float32
            )
            sampled_facet_idxs = paddle.multinomial(
                facet_areas, num_samples=n_samples_per_wyckoff, replacement=True
            )

            max_num_triangles_per_facet: int = wyckoff_shapes_decomp_dict["max_triangles_per_facet"]
            _sampled_facet_triangle_areas = [
                paddle.to_tensor(
                    wyckoff_shapes_decomp_dict["facet_triangle_areas"][int(facet_index)],
                    dtype=paddle.float32,
                )
                for facet_index in sampled_facet_idxs
            ]

            sampled_facet_triangle_areas = paddle.zeros(
                [n_samples_per_wyckoff, max_num_triangles_per_facet]
            )
            for j in range(n_samples_per_wyckoff):
                n = _sampled_facet_triangle_areas[j].shape[0]
                sampled_facet_triangle_areas[j, :n] = _sampled_facet_triangle_areas[j]

            sampled_triangle_idxs = paddle.multinomial(
                sampled_facet_triangle_areas, num_samples=1
            ).squeeze(axis=1)

            vertices = paddle.stack([
                paddle.to_tensor(
                    wyckoff_shapes_decomp_dict["facet_triangles"][int(facet_idx)][int(triangle_idx)],
                    dtype=paddle.float32,
                )
                for facet_idx, triangle_idx in zip(sampled_facet_idxs, sampled_triangle_idxs)
            ], axis=0)
            sampled_wyckoff_shape_index = sampled_facet_idxs

        else:
            vertices = wyckoff_position_dict["vertices_tensor"]
            sampled_wyckoff_shape_index = paddle.to_tensor(
                [0], dtype=paddle.int64
            ).expand([n_samples_per_wyckoff])

        if finished_sampling_mask is not None and finished_sampling_mask[i]:
            sample_in_wyckoff = paddle.full(
                [n_samples_per_wyckoff, 3], -1.0, dtype=paddle.float32
            )
            sampled_wyckoff_shape_index = paddle.full(
                [n_samples_per_wyckoff], -1, dtype=paddle.int64
            )
        else:
            sample_in_wyckoff = uniformly_sample_point_in_convex_shape(
                space_group_number=int(space_group_number),
                vertices=vertices,
                wyckoff_site_dimensionality=wyckoff_dof,
                n_samples=n_samples_per_wyckoff,
            )
            assert sample_in_wyckoff is not None

        random_samples_in_wyckoffs.append(sample_in_wyckoff)
        sampled_wyckoff_shape_indices.append(sampled_wyckoff_shape_index)

    random_samples_in_wyckoffs = paddle.stack(random_samples_in_wyckoffs, axis=0)
    sampled_wyckoff_shape_indices = paddle.stack(sampled_wyckoff_shape_indices, axis=0)
    if return_sampled_wyckoff_shape_indices:
        return random_samples_in_wyckoffs, sampled_wyckoff_shape_indices
    else:
        return random_samples_in_wyckoffs

def uniformly_sample_point_in_convex_shape(
    space_group_number: int,
    vertices: paddle.Tensor,
    wyckoff_site_dimensionality: int,
    n_samples: int = 1,
) -> paddle.Tensor:
    """
    在凸形状内均匀采样点。

    Args:
        space_group_number: [1, 230]
        vertices:
            0D: (1, 3)
            1D: (n_samples, 2, 3)  line segments
            2D: (n_samples, 3, 3)  triangles
            3D: (n_vertices, 3)    polytope
        wyckoff_site_dimensionality: [0, 1, 2, 3]
        n_samples: 采样数量

    Returns:
        shape (n_samples, 3)
    """
    if wyckoff_site_dimensionality == 0:
        assert vertices.shape == [1, 3]
        return vertices.expand([n_samples, 3])

    elif wyckoff_site_dimensionality == 1:
        assert list(vertices.shape) == [n_samples, 2, 3]
        samples = paddle.rand([n_samples, 1])
        end_point1 = vertices[:, 0]
        end_point2 = vertices[:, 1]
        return samples * (end_point2 - end_point1) + end_point1

    elif wyckoff_site_dimensionality == 2:
        assert list(vertices.shape) == [n_samples, 3, 3]
        # 三角形内均匀采样: P = (1-sqrt(r1))*A + sqrt(r1)*(1-r2)*B + sqrt(r1)*r2*C
        r1_sqrt = paddle.rand([n_samples, 1]).sqrt()
        r2 = paddle.rand([n_samples, 1])
        return (
            (1.0 - r1_sqrt) * vertices[:, 0, :]
            + r1_sqrt * (1.0 - r2) * vertices[:, 1, :]
            + r1_sqrt * r2 * vertices[:, 2, :]
        )

    elif wyckoff_site_dimensionality == 3:
        # paddle.Tensor.min(axis) 在新版 Paddle 中返回 (values, indices) 元组，
        # 需使用 paddle.min / paddle.max 直接获取值
        box_lower_left = paddle.min(vertices, axis=0)  # [xmin, ymin, zmin]
        box_top_right = paddle.max(vertices, axis=0)   # [xmax, ymax, zmax]
        asu_hull_equations = global_vars.asu_hull_equations[space_group_number - 1]

        accepted_samples = []
        total_n_accepted = 0
        while True:
            candidate = paddle.rand([3 * n_samples, 3])
            candidate = candidate * (box_top_right - box_lower_left) + box_lower_left
            mask = is_inside(candidate, asu_hull_equations)
            accepted_samples.append(candidate[mask])
            total_n_accepted += mask.cast(paddle.int64).sum().item()
            if total_n_accepted >= n_samples:
                return paddle.concat(accepted_samples, axis=0)[:n_samples]
    else:
        raise AttributeError(f"Invalid dimensionality: {wyckoff_site_dimensionality}")
