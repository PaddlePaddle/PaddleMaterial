# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import math
import paddle
from ppmat.utils.crystal import lattice_params_to_matrix_paddle
from ppmat.utils.scatter import scatter_sum as _base_scatter_sum
from ppmat.utils.scatter import scatter_mean as _base_scatter_mean

__all__ = [
    "get_index_embedding",
    "to_dense_batch",
    "scatter_sum",
    "scatter_mean",
    "lattice_params_to_matrix",
    "frac_to_cart_coords",
    "cart_to_frac_coords",
    "get_pbc_distances",
    "lattice_vector_to_volume",
    "apply_augmentation",
    "apply_noise",
    "set_gelu_approx",
    "make_attn_mask",
]


def get_index_embedding(indices, emb_dim, max_len=2048):
    K = paddle.arange(emb_dim // 2)
    pos_embedding_sin = paddle.sin(
        indices.unsqueeze(-1) * math.pi / (max_len ** (2 * K / emb_dim))
    )
    pos_embedding_cos = paddle.cos(
        indices.unsqueeze(-1) * math.pi / (max_len ** (2 * K / emb_dim))
    )
    pos_embedding = paddle.concat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def to_dense_batch(x, batch_idx, max_num_nodes=None):
    batch_size = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1
    batch_size_t = paddle.to_tensor(batch_size, dtype='int64')
    num_nodes = paddle.zeros([batch_size], dtype='int64')
    expanded_idx = paddle.arange(batch_size, dtype='int64').unsqueeze(1).expand([batch_size, batch_idx.shape[0]])
    num_nodes = (expanded_idx == batch_idx.unsqueeze(0)).astype('int64').sum(axis=1)
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max().item())
    feat_dim = x.shape[-1]
    x_dense = paddle.zeros([batch_size, max_num_nodes, feat_dim], dtype=x.dtype)
    mask = paddle.zeros([batch_size, max_num_nodes], dtype='bool')
    cumsum = paddle.concat([paddle.zeros([1], dtype='int64'), paddle.cumsum(num_nodes, axis=0)[:-1]])
    for i in range(batch_size):
        start = int(cumsum[i].item())
        end = start + int(num_nodes[i].item())
        n = int(num_nodes[i].item())
        x_dense[i, :n] = x[start:end]
        mask[i, :n] = True
    return x_dense, mask


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    return _base_scatter_sum(src, index, dim, out, dim_size)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    return _base_scatter_mean(src, index, dim, out, dim_size)


def lattice_params_to_matrix(lengths, angles):
    return lattice_params_to_matrix_paddle(lengths, angles)


def frac_to_cart_coords(frac_coords, lattice):
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    if lattice.ndim == 3 and lattice.shape[0] == 1:
        lattice = lattice.squeeze(0)
    return paddle.einsum('ij,jk->ik', frac_coords, lattice)


def cart_to_frac_coords(cart_coords, lattice):
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    if lattice.ndim == 3 and lattice.shape[0] == 1:
        lattice = lattice.squeeze(0)
    inv_lattice = paddle.linalg.pinv(lattice)
    return paddle.einsum('ij,jk->ik', cart_coords, inv_lattice)


def get_pbc_distances(coords1, coords2, lattice, num_atoms=None, return_offsets=False):
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    if coords1.shape != coords2.shape:
        raise ValueError("coords1 and coords2 must have the same shape")
    diff = coords2 - coords1
    diff_frac = cart_to_frac_coords(diff, lattice)
    diff_frac = diff_frac - paddle.round(diff_frac)
    diff_cart = frac_to_cart_coords(diff_frac, lattice)
    distances = paddle.norm(diff_cart, axis=-1)
    if return_offsets:
        return distances, paddle.round(cart_to_frac_coords(coords2 - coords1, lattice))
    return distances


def lattice_vector_to_volume(lattice):
    if lattice.ndim == 2:
        lattice = lattice.unsqueeze(0)
    a, b, c = lattice[:, 0, :], lattice[:, 1, :], lattice[:, 2, :]
    return paddle.abs(paddle.sum(a * paddle.cross(b, c), axis=-1))


def apply_augmentation(batch, translate=False, rotate=False):
    if not translate and not rotate:
        return batch
    batch_aug = batch.clone()
    if translate:
        batch_aug = _augmentation_translate(batch_aug)
    if rotate:
        batch_aug = _augmentation_rotate(batch_aug)
    return batch_aug


def _augmentation_translate(batch):
    lengths_mean = batch.lengths.mean(axis=0)
    lengths_std = batch.lengths.std(axis=0, unbiased=False)
    random_translate = paddle.normal(
        mean=paddle.abs(lengths_mean),
        std=paddle.maximum(paddle.abs(lengths_std), paddle.to_tensor([1e-8]))
    ) / 2
    cart_coords_aug = batch.cart_coords + random_translate
    cell_per_node_inv = paddle.inverse(batch.lattices[batch.batch])
    frac_coords_aug = paddle.einsum('bi,bij->bj', cart_coords_aug, cell_per_node_inv) % 1.0
    batch.cart_coords = cart_coords_aug
    batch.frac_coords = frac_coords_aug
    return batch


def _augmentation_rotate(batch):
    rot_mat = _random_rotation_matrix()
    batch.cart_coords = paddle.matmul(batch.cart_coords, rot_mat.T)
    batch.lattices = paddle.matmul(batch.lattices, rot_mat.T)
    return batch


def apply_noise(batch, ratio=0.1, corruption_scale=0.1):
    if ratio <= 0:
        return batch
    batch_noise = batch.clone()
    total_num_atoms = batch_noise.num_nodes
    noise_num_atoms = int(total_num_atoms * ratio)
    noise_atom_types = batch_noise.atom_types.clone()
    type_noise_idx = paddle.randperm(total_num_atoms)[:noise_num_atoms]
    noise_atom_types[type_noise_idx] = 0
    noise_cart_coords = batch_noise.cart_coords.clone()
    coord_noise_idx = paddle.randperm(total_num_atoms)[:noise_num_atoms]
    noise_cart_coords[coord_noise_idx] += paddle.randn([noise_num_atoms, 3]) * corruption_scale
    cell_per_node_inv = paddle.inverse(batch.lattices[batch.batch])
    noise_frac_coords = paddle.einsum('bi,bij->bj', noise_cart_coords, cell_per_node_inv) % 1.0
    batch_noise.atom_types = noise_atom_types
    batch_noise.cart_coords = noise_cart_coords
    batch_noise.frac_coords = noise_frac_coords
    return batch_noise


def set_gelu_approx(transformer):
    if hasattr(transformer, 'layers'):
        for layer in transformer.layers:
            if hasattr(layer, 'activation'):
                layer.activation = paddle.nn.GELU(approximate='tanh')


def make_attn_mask(token_mask):
    if not token_mask.cast('bool').all():
        bsize, seq_len = token_mask.shape
        mask = paddle.zeros([bsize, 1, 1, seq_len], dtype='float32')
        return mask - 1e9 * (~token_mask).unsqueeze(1).unsqueeze(2).astype('float32')
    return None


def _random_rotation_matrix():
    q = paddle.rand([4])
    q = q / paddle.norm(q)
    rot_mat = paddle.to_tensor([
        [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3], 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2],
    ], dtype='float32')
    return rot_mat
