"""Wyckoff 形状分解字典构建器：使用 scipy 替代原始的 meshpy。"""
import json
import os
import pickle

import numpy as np
from scipy.spatial import ConvexHull

def _to_array(region) -> np.ndarray:
    """将顶点列表转换为 numpy 数组，支持分数字符串如 "1/2"。"""
    if isinstance(region, np.ndarray):
        return region.astype(np.float64)
    result = []
    for pt in region:
        if isinstance(pt, (list, tuple)):
            coords = []
            for c in pt:
                if isinstance(c, str):
                    # 支持分数字符串如 "1/2"
                    coords.append(float(eval(c)))
                else:
                    coords.append(float(c))
            result.append(coords)
        else:
            result.append([float(pt)])
    return np.array(result, dtype=np.float64)

def _to_affine_transform(simplicial_complex: np.ndarray):
    """从单纯复形顶点提取仿射变换参数。"""
    assert simplicial_complex.shape[0] > 1
    offset = simplicial_complex[0]
    basis_maps = []
    for vec in simplicial_complex[1:]:
        basis_maps.append(vec - offset)
    return offset, np.stack(basis_maps)

def _fan_triangulate_convex_polygon_3d(vertices_3d: np.ndarray):
    """对三维空间中的凸多边形做扇形三角剖分（从质心出发）。"""
    n = len(vertices_3d)
    if n < 3:
        return np.zeros((0, 3, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

    centroid = vertices_3d.mean(axis=0)
    triangles = []
    areas = []
    for i in range(n):
        v0 = centroid
        v1 = vertices_3d[i]
        v2 = vertices_3d[(i + 1) % n]
        tri = np.stack([v0, v1, v2], axis=0)  # (3, 3)
        # 三角形面积 = 0.5 * |cross(v1-v0, v2-v0)|
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        triangles.append(tri)
        areas.append(area)
    return np.array(triangles, dtype=np.float64), np.array(areas, dtype=np.float64)

def _compute_3d_convex_hull_volume(vertices: np.ndarray) -> float:
    """计算三维凸多面体的体积。"""
    if len(vertices) < 4:
        return 0.0
    try:
        hull = ConvexHull(vertices)
        return float(hull.volume)
    except Exception:
        # 退化情况（所有点共面等）
        return 0.0

def build_wyckoff_shape_decomposition_dict(
    output_path: str,
    asu_dict_path: str,
) -> None:
    """
    构建 Wyckoff 形状分解字典并持久化到 pickle 文件。
    """
    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[wyckoff_shape_decomp_builder] 开始构建 {output_path} ...")

    with open(asu_dict_path, "r") as f:
        asu_dict = json.load(f)

    shape_decomposition_dict = {}

    for sg, sg_dict in asu_dict.items():
        sg_shape_info = {}

        for wyckoff_letter in sg_dict["ordered_wyckoff_letters"]:
            geometry = sg_dict[wyckoff_letter]
            site_dim = int(geometry["dim"])

            if site_dim == 0:
                wyckoff_shapes_info = {"dim": 0, "volumes": None}

            elif site_dim == 1:
                wyckoff_shapes_info = {"dim": 1, "volumes": []}
                for interval in geometry["vertices"]:
                    arr = _to_array(interval)  # (2, 3)
                    if arr.shape[0] >= 2:
                        diff = arr[1] - arr[0]  # (3,)
                        length = float(np.linalg.norm(diff))
                    else:
                        length = 0.0
                    wyckoff_shapes_info["volumes"].append(length)

            elif site_dim == 2:
                wyckoff_shapes_info = {
                    "dim": 2,
                    "volumes": [],
                    "facet_triangles": [],
                    "facet_triangle_areas": [],
                    "max_triangles_per_facet": 0,
                }
                max_tri = 0
                for facet in geometry["vertices"]:
                    arr = _to_array(facet)  # (n_v, 3)
                    triangles, areas = _fan_triangulate_convex_polygon_3d(arr)
                    total_area = float(areas.sum()) if len(areas) > 0 else 0.0
                    wyckoff_shapes_info["volumes"].append(total_area)
                    wyckoff_shapes_info["facet_triangles"].append(triangles)
                    wyckoff_shapes_info["facet_triangle_areas"].append(areas)
                    max_tri = max(max_tri, len(triangles))
                wyckoff_shapes_info["max_triangles_per_facet"] = max_tri

            elif site_dim == 3:
                wyckoff_shapes_info = {"dim": 3, "volumes": []}
                arr = _to_array(geometry["vertices"])  # (n_v, 3)
                vol = _compute_3d_convex_hull_volume(arr)
                wyckoff_shapes_info["volumes"].append(vol)

            else:
                raise ValueError(f"Wyckoff 位维度必须在 [0, 1, 2, 3] 内，得到: {site_dim}")

            sg_shape_info[wyckoff_letter] = wyckoff_shapes_info

        shape_decomposition_dict[sg] = sg_shape_info

    with open(output_path, "wb") as f:
        pickle.dump(shape_decomposition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[wyckoff_shape_decomp_builder] 构建完成 -> {output_path}")
