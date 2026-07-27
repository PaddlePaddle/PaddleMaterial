# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepH Hamiltonian graph construction for PaddleMaterials.

This module implements the DeepH graphene training and inference graph paths.
Training can use radius graphs or the sparsity pattern of DFT matrices. Inference
uses the overlap-derived local-coordinate keys produced by the DFT workflow.
"""

import itertools
import json
import os

import numpy as np


class Data:
    """Small array container used before conversion to PaddleMaterials Data."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _semifactorial(x):
    y = 1.0
    for n in range(x, 1, -2):
        y *= n
    return y


def _pochhammer(x, k):
    xf = float(x)
    for n in range(x + 1, x + k):
        xf *= n
    return xf


class _SphericalHarmonics:
    """Real spherical harmonics used by the LCMP subgraph angular features."""

    def __init__(self):
        self.leg = {}

    def clear(self):
        self.leg = {}

    def _negative_lpmv(self, degree, order, value):
        if order < 0:
            value *= (-1) ** order / _pochhammer(degree + order + 1, -2 * order)
        return value

    def lpmv(self, degree, order, x):
        order_abs = abs(order)
        if (degree, order) in self.leg:
            return self.leg[(degree, order)]
        if order_abs > degree:
            return None
        if degree == 0:
            self.leg[(degree, order)] = np.ones_like(x)
            return self.leg[(degree, order)]
        if order_abs == degree:
            value = (-1) ** order_abs * _semifactorial(2 * order_abs - 1)
            value *= np.power(1 - x * x, order_abs / 2)
            self.leg[(degree, order)] = self._negative_lpmv(degree, order, value)
            return self.leg[(degree, order)]

        self.lpmv(degree - 1, order, x)
        value = (
            ((2 * degree - 1) / (degree - order_abs))
            * x
            * self.lpmv(degree - 1, order_abs, x)
        )
        if degree - order_abs > 1:
            value -= ((degree + order_abs - 1) / (degree - order_abs)) * self.leg[
                (degree - 2, order_abs)
            ]
        if order < 0:
            value = self._negative_lpmv(degree, order, value)
        self.leg[(degree, order)] = value
        return self.leg[(degree, order)]

    def get_element(self, degree, order, theta, phi):
        norm = np.sqrt((2 * degree + 1) / (4 * np.pi))
        leg = self.lpmv(degree, abs(order), np.cos(theta))
        if order == 0:
            return norm * leg
        if order > 0:
            value = np.cos(order * phi) * leg
        else:
            value = np.sin(abs(order) * phi) * leg
        norm *= np.sqrt(
            2.0
            / _pochhammer(
                degree - abs(order) + 1,
                2 * abs(order),
            )
        )
        return value * norm

    def get(self, degree, theta, phi, refresh=True):
        if refresh:
            self.clear()
        return np.stack(
            [
                self.get_element(degree, order, theta, phi)
                for order in range(-degree, degree + 1)
            ],
            axis=-1,
        )


def _get_spherical_from_cartesian(cartesian):
    spherical = np.zeros(cartesian.shape[:-1] + (2,), dtype=cartesian.dtype)
    r_xy = cartesian[..., 1] ** 2 + cartesian[..., 2] ** 2
    spherical[..., 0] = np.arctan2(np.sqrt(r_xy), cartesian[..., 0])
    spherical[..., 1] = np.arctan2(cartesian[..., 2], cartesian[..., 1])
    return spherical


def _load_orbital_types(path):
    orbital_types = []
    with open(path) as f:
        line = f.readline()
        while line:
            orbital_types.append(list(map(int, line.split())))
            line = f.readline()
    return [
        sum(2 * orbital + 1 for orbital in atom_orbitals)
        for atom_orbitals in orbital_types
    ]


def _validate_supported_path(
    interface,
    target,
    create_from_DFT,
    if_lcmp_graph,
    separate_onsite,
    if_new_sp,
    huge_structure,
):
    if interface not in {"npz", "npz_rc_only"}:
        raise NotImplementedError(
            "DeepH PaddleMaterials integration supports interface='npz' and "
            "'npz_rc_only'."
        )
    if target != "hamiltonian":
        raise NotImplementedError(
            "DeepH PaddleMaterials integration currently supports target='hamiltonian'."
        )
    if interface == "npz_rc_only" and not create_from_DFT:
        raise ValueError("interface='npz_rc_only' requires create_from_DFT=True.")
    if not if_lcmp_graph:
        raise NotImplementedError(
            "DeepH PaddleMaterials integration expects if_lcmp_graph=True."
        )
    if separate_onsite:
        raise NotImplementedError(
            "DeepH separate_onsite=True is not enabled in this integration."
        )
    if if_new_sp:
        raise NotImplementedError(
            "DeepH new_sp=True is not enabled in this integration."
        )
    if huge_structure:
        raise NotImplementedError(
            "DeepH huge_structure=True is not enabled in this integration."
        )


def _periodic_image_ranges(frac_coords, lattice, radius):
    reciprocal_lattice = np.linalg.inv(lattice).T * 2 * np.pi
    recp_len = np.sqrt(np.sum(reciprocal_lattice**2, axis=1))
    maxr = np.ceil((radius + 0.15) * recp_len / (2 * np.pi))
    nmin = np.floor(np.min(frac_coords, axis=0)) - maxr
    nmax = np.ceil(np.max(frac_coords, axis=0)) + maxr
    return [np.arange(x, y, dtype="int64") for x, y in zip(nmin, nmax)]


def _build_edges_from_converter_graph(converter_graph, default_dtype):
    edge_idx = np.asarray(converter_graph.edges, dtype=np.int64).T
    cart_coords = np.asarray(
        converter_graph.node_feat["cart_coords"],
        dtype=default_dtype,
    )
    lattice = np.asarray(
        converter_graph.node_feat["lattice"],
        dtype=default_dtype,
    ).reshape(3, 3)
    pbc_offset = np.asarray(
        converter_graph.edge_feat["pbc_offset"],
        dtype=default_dtype,
    )
    edge_dist = np.asarray(
        converter_graph.edge_feat["bond_dist"],
        dtype=default_dtype,
    ).reshape(-1, 1)
    src, dst = edge_idx
    dst_cart_periodic = cart_coords[dst] + pbc_offset @ lattice
    edge_fea = np.concatenate(
        [
            edge_dist,
            cart_coords[src],
            dst_cart_periodic,
            cart_coords[dst],
        ],
        axis=-1,
    ).astype(default_dtype)
    onsite_src = np.arange(cart_coords.shape[0], dtype=np.int64)
    onsite_edge_idx = np.stack([onsite_src, onsite_src])
    onsite_edge_fea = np.concatenate(
        [
            np.zeros((cart_coords.shape[0], 1), dtype=default_dtype),
            cart_coords,
            cart_coords,
            cart_coords,
        ],
        axis=-1,
    )
    edge_idx = np.concatenate([edge_idx, onsite_edge_idx], axis=1)
    edge_fea = np.concatenate([edge_fea, onsite_edge_fea], axis=0)
    src, dst = edge_idx

    num_atom = cart_coords.shape[0]
    atom_idx_connect, edge_idx_connect = [], []
    for atom_idx in range(num_atom):
        outgoing_edge_idx = np.where(src == atom_idx)[0]
        edge_idx_connect.append(outgoing_edge_idx)
        atom_idx_connect.append(dst[outgoing_edge_idx])
    return edge_idx, edge_fea, atom_idx_connect, edge_idx_connect


def _load_npz_terms(tb_folder, default_dtype):
    atom_num_orbital = _load_orbital_types(os.path.join(tb_folder, "orbital_types.dat"))

    read_terms = {}
    hopping_dict_read = np.load(os.path.join(tb_folder, "rh.npz"))
    for k, v in hopping_dict_read.items():
        key = json.loads(k)
        key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
        read_terms[key] = np.asarray(v, dtype=default_dtype)

    local_rotation_dict = {}
    local_rotation_dict_read = np.load(os.path.join(tb_folder, "rc.npz"))
    for k, v in local_rotation_dict_read.items():
        key = json.loads(k)
        key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
        local_rotation_dict[key] = np.asarray(v, dtype=default_dtype)

    return atom_num_orbital, read_terms, local_rotation_dict


def _load_npz_rotations(tb_folder, default_dtype):
    atom_num_orbital = _load_orbital_types(os.path.join(tb_folder, "orbital_types.dat"))
    rotations = {}
    with np.load(os.path.join(tb_folder, "rc.npz")) as rotation_file:
        for key_str, value in rotation_file.items():
            key = json.loads(key_str)
            key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
            rotations[key] = np.asarray(value, dtype=default_dtype)
    return atom_num_orbital, rotations


def _build_edges_from_rotation_keys(
    cart_coords,
    lattice,
    local_rotation_dict,
    default_dtype,
):
    edge_keys = list(local_rotation_dict)
    if not edge_keys:
        raise ValueError("DeepH local-coordinate file does not contain any edges.")

    edge_idx = np.asarray([[key[3], key[4]] for key in edge_keys], dtype=np.int64).T
    lattice_shifts = np.asarray([key[:3] for key in edge_keys], dtype=default_dtype)
    src, dst = edge_idx
    dst_cart_periodic = cart_coords[dst] + lattice_shifts @ lattice
    edge_dist = np.linalg.norm(
        dst_cart_periodic - cart_coords[src], axis=1, keepdims=True
    )
    edge_fea = np.concatenate(
        [
            edge_dist,
            cart_coords[src],
            dst_cart_periodic,
            cart_coords[dst],
        ],
        axis=-1,
    ).astype(default_dtype)

    atom_idx_connect = []
    edge_idx_connect = []
    for atom_idx in range(cart_coords.shape[0]):
        outgoing_edge_idx = np.where(src == atom_idx)[0]
        if outgoing_edge_idx.size == 0:
            raise ValueError(f"Atom {atom_idx} has no overlap-derived DeepH edges.")
        edge_idx_connect.append(outgoing_edge_idx)
        atom_idx_connect.append(dst[outgoing_edge_idx])

    local_rotation = np.stack(
        [local_rotation_dict[key] for key in edge_keys], axis=0
    ).astype(default_dtype)
    return (
        edge_idx,
        edge_fea,
        atom_idx_connect,
        edge_idx_connect,
        local_rotation,
    )


def _attach_terms(
    edge_idx,
    edge_fea,
    lattice,
    atom_num_orbital,
    read_terms,
    local_rotation_dict,
    default_dtype,
):
    max_num_orbital = max(atom_num_orbital)
    term_mask = np.zeros(edge_fea.shape[0], dtype=bool)
    term_real = np.full(
        [edge_fea.shape[0], max_num_orbital, max_num_orbital],
        np.nan,
        dtype=default_dtype,
    )
    local_rotation = []
    inv_lattice = np.linalg.inv(lattice).astype(default_dtype)

    for index_edge in range(edge_fea.shape[0]):
        lattice_shift = (
            np.rint(
                edge_fea[index_edge, 4:7] @ inv_lattice
                - edge_fea[index_edge, 7:10] @ inv_lattice
            )
            .astype(int)
            .tolist()
        )
        i, j = edge_idx[:, index_edge]
        key_term = (*lattice_shift, int(i), int(j))
        if key_term not in read_terms:
            raise NotImplementedError(
                "Graph radius including hopping without calculation is not supported."
            )
        term_mask[index_edge] = True
        term_real[
            index_edge,
            : atom_num_orbital[i],
            : atom_num_orbital[j],
        ] = read_terms[key_term]
        local_rotation.append(local_rotation_dict[key_term])

    return term_mask, term_real, np.stack(local_rotation, axis=0)


def _build_lcmp_subgraph(
    edge_idx, edge_fea, local_rotation, atom_idx_connect, edge_idx_connect, num_l
):
    r_vec = edge_fea[:, 1:4] - edge_fea[:, 4:7]
    r_vec = np.matmul(
        r_vec[:, None, None, :],
        local_rotation[None, :, :, :],
    ).reshape(-1, 3)

    r_vec_sp = _get_spherical_from_cartesian(r_vec)
    sph_harm_func = _SphericalHarmonics()
    angular_expansion = []
    for l_value in range(num_l):
        angular_expansion.append(
            sph_harm_func.get(l_value, r_vec_sp[:, 0], r_vec_sp[:, 1])
        )
    angular_expansion = np.concatenate(angular_expansion, axis=-1).reshape(
        edge_fea.shape[0],
        edge_fea.shape[0],
        -1,
    )

    subgraph_atom_idx_list = []
    subgraph_edge_idx_list = []
    subgraph_edge_ang_list = []
    subgraph_index = []
    index_cursor = 0

    for index in range(edge_fea.shape[0]):
        i, j = edge_idx[:, index]

        subgraph_edge_idx = np.asarray(list(edge_idx_connect[i]), dtype=np.int64)
        subgraph_atom_idx = np.stack(
            [np.repeat(i, len(atom_idx_connect[i])), atom_idx_connect[i]],
            axis=1,
        )
        subgraph_atom_idx_list.append(subgraph_atom_idx)
        subgraph_edge_idx_list.append(subgraph_edge_idx)
        subgraph_edge_ang_list.append(angular_expansion[subgraph_edge_idx, index, :])
        subgraph_index += [index_cursor] * len(atom_idx_connect[i])
        index_cursor += 1

        subgraph_edge_idx = np.asarray(list(edge_idx_connect[j]), dtype=np.int64)
        subgraph_atom_idx = np.stack(
            [np.repeat(j, len(atom_idx_connect[j])), atom_idx_connect[j]],
            axis=1,
        )
        subgraph_atom_idx_list.append(subgraph_atom_idx)
        subgraph_edge_idx_list.append(subgraph_edge_idx)
        subgraph_edge_ang_list.append(angular_expansion[subgraph_edge_idx, index, :])
        subgraph_index += [index_cursor] * len(atom_idx_connect[j])
        index_cursor += 1

    return {
        "subgraph_atom_idx": np.concatenate(subgraph_atom_idx_list, axis=0).astype(
            np.int64
        ),
        "subgraph_edge_idx": np.concatenate(subgraph_edge_idx_list, axis=0).astype(
            np.int64
        ),
        "subgraph_edge_ang": np.concatenate(subgraph_edge_ang_list, axis=0).astype(
            edge_fea.dtype
        ),
        "subgraph_index": np.asarray(subgraph_index, dtype=np.int64),
    }


def get_graph(
    cart_coords,
    frac_coords,
    numbers,
    stru_id,
    r,
    max_num_nbr,
    numerical_tol,
    lattice,
    default_dtype,
    tb_folder,
    interface,
    num_l,
    create_from_DFT,
    if_lcmp_graph,
    separate_onsite,
    target="hamiltonian",
    huge_structure=False,
    only_get_R_list=False,
    if_new_sp=False,
    **kwargs,
):
    _validate_supported_path(
        interface,
        target,
        create_from_DFT,
        if_lcmp_graph,
        separate_onsite,
        if_new_sp,
        huge_structure,
    )
    if tb_folder is None:
        raise ValueError("DeepH graph construction requires tb_folder.")
    converter_graph = kwargs.get("converter_graph")
    if max_num_nbr > 0:
        raise NotImplementedError(
            "DeepH PaddleMaterials integration currently relies on radius-based "
            "FindPointsInSpheres graph conversion and expects max_num_nbr=0."
        )

    cart_coords = np.asarray(cart_coords, dtype=default_dtype)
    frac_coords = np.asarray(frac_coords, dtype=default_dtype)
    numbers = np.asarray(numbers, dtype=np.int64)
    lattice = np.asarray(lattice, dtype=default_dtype)

    if only_get_R_list:
        return np.array(
            list(itertools.product(*_periodic_image_ranges(frac_coords, lattice, r))),
            dtype=default_dtype,
        )

    term_mask = term_real = None
    if create_from_DFT:
        atom_num_orbital, local_rotation_dict = _load_npz_rotations(
            tb_folder, default_dtype
        )
        (
            edge_idx,
            edge_fea,
            atom_idx_connect,
            edge_idx_connect,
            local_rotation,
        ) = _build_edges_from_rotation_keys(
            cart_coords,
            lattice,
            local_rotation_dict,
            default_dtype,
        )
        if interface == "npz":
            _, read_terms, _ = _load_npz_terms(tb_folder, default_dtype)
            term_mask, term_real, _ = _attach_terms(
                edge_idx,
                edge_fea,
                lattice,
                atom_num_orbital,
                read_terms,
                local_rotation_dict,
                default_dtype,
            )
    else:
        if converter_graph is None:
            raise ValueError(
                "Radius-based DeepH graph construction expects a "
                "PaddleMaterials graph_converter graph."
            )
        (
            edge_idx,
            edge_fea,
            atom_idx_connect,
            edge_idx_connect,
        ) = _build_edges_from_converter_graph(converter_graph, default_dtype)
        atom_num_orbital, read_terms, local_rotation_dict = _load_npz_terms(
            tb_folder, default_dtype
        )
        term_mask, term_real, local_rotation = _attach_terms(
            edge_idx,
            edge_fea,
            lattice,
            atom_num_orbital,
            read_terms,
            local_rotation_dict,
            default_dtype,
        )
    subgraph = _build_lcmp_subgraph(
        edge_idx,
        edge_fea,
        local_rotation,
        atom_idx_connect,
        edge_idx_connect,
        num_l,
    )

    data = Data(
        x=numbers,
        edge_index=edge_idx,
        edge_attr=edge_fea,
        stru_id=stru_id,
        onsite_term_real=None,
        atom_num_orbital=np.asarray(atom_num_orbital, dtype=np.int64),
        subgraph_dict=subgraph,
        spinful=False,
    )
    if term_mask is not None:
        data.term_mask = term_mask
        data.term_real = term_real
    return data
