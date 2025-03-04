import sys

import paddle

from paddle_utils import *

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/gemnet.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""
from dataclasses import dataclass
from typing import Optional
from typing import Tuple

from paddle.sparse import sparse_coo_tensor
from paddle_scatter import scatter

from mattergen.common.gemnet.layers.atom_update_block import OutputBlock
from mattergen.common.gemnet.layers.base_layers import Dense
from mattergen.common.gemnet.layers.efficient import EfficientInteractionDownProjection
from mattergen.common.gemnet.layers.embedding_block import EdgeEmbedding
from mattergen.common.gemnet.layers.interaction_block import InteractionBlockTripletsOnly
from mattergen.common.gemnet.layers.radial_basis import RadialBasis
from mattergen.common.gemnet.layers.scaling import AutomaticFit
from mattergen.common.gemnet.layers.spherical_basis import CircularBasisLayer
from mattergen.common.gemnet.utils import inner_product_normalized
from mattergen.common.gemnet.utils import mask_neighbors
from mattergen.common.gemnet.utils import ragged_range
from mattergen.common.gemnet.utils import repeat_blocks
from mattergen.common.utils.data_utils import frac_to_cart_coords_with_lattice
from mattergen.common.utils.data_utils import get_pbc_distances
from mattergen.common.utils.data_utils import lattice_params_to_matrix_paddle
from mattergen.common.utils.data_utils import radius_graph_pbc
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.common.utils.lattice_score import edge_score_to_lattice_score_frac_symmetric


@dataclass(frozen=True)
class ModelOutput:
    energy: paddle.Tensor
    node_embeddings: paddle.Tensor
    forces: Optional[paddle.Tensor] = None
    stress: Optional[paddle.Tensor] = None


class RBFBasedLatticeUpdateBlock(paddle.nn.Layer):
    def __init__(
        self,
        emb_size: int,
        activation: str,
        emb_size_rbf: int,
        emb_size_edge: int,
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_out = num_heads
        self.mlp = paddle.nn.Sequential(
            Dense(emb_size, emb_size, activation=activation), Dense(emb_size, emb_size)
        )
        self.dense_rbf_F = Dense(emb_size_rbf, emb_size_edge, activation=None, bias=False)
        self.out_forces = Dense(emb_size_edge, num_heads, bias=False, activation=None)

    def compute_score_per_edge(self, edge_emb: paddle.Tensor, rbf: paddle.Tensor) -> paddle.Tensor:
        x_F = self.mlp(edge_emb)
        rbf_emb_F = self.dense_rbf_F(rbf)
        x_F_rbf = x_F * rbf_emb_F
        x_F = self.out_forces(x_F_rbf)
        return x_F


class RBFBasedLatticeUpdateBlockFrac(RBFBasedLatticeUpdateBlock):
    def __init__(
        self,
        emb_size: int,
        activation: str,
        emb_size_rbf: int,
        emb_size_edge: int,
        num_heads: int = 1,
    ):
        super().__init__(
            emb_size=emb_size,
            activation=activation,
            emb_size_rbf=emb_size_rbf,
            emb_size_edge=emb_size_edge,
            num_heads=num_heads,
        )

    def forward(
        self,
        edge_emb: paddle.Tensor,
        edge_index: paddle.Tensor,
        distance_vec: paddle.Tensor,
        lattice: paddle.Tensor,
        batch: paddle.Tensor,
        rbf: paddle.Tensor,
        normalize_score: bool = True,
    ) -> paddle.Tensor:
        edge_scores = self.compute_score_per_edge(edge_emb=edge_emb, rbf=rbf)
        if normalize_score:
            num_edges = scatter(paddle.ones_like(x=distance_vec[:, 0]), batch[edge_index[0]])
            edge_scores /= num_edges[batch[edge_index[0]], None]
        outs = []
        for i in range(self.num_out):
            lattice_update = edge_score_to_lattice_score_frac_symmetric(
                score_d=edge_scores[:, i],
                edge_index=edge_index,
                edge_vectors=distance_vec,
                batch=batch,
            )
            outs.append(lattice_update)
        outs = paddle.stack(x=outs, axis=-1).sum(axis=-1)
        return outs


class GemNetT(paddle.nn.Layer):
    """
    GemNet-T, triplets-only variant of GemNet

    Parameters
    ----------
        num_targets: int
            Number of prediction targets.

        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        atom_embedding: paddle.nn.Layer
            a module that embeds atomic numbers into vectors of size emb_dim_atomic_number.
        emb_size_atom: int
            Embedding size of the atoms. This can be different from emb_dim_atomic_number.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.
        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
        encoder_mode: bool
            if <True>, use the encoder mode of the model, i.e. only get the atom/edge embedddings.
    """

    def __init__(
        self,
        num_targets: int,
        latent_dim: int,
        atom_embedding: paddle.nn.Layer,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 512,
        emb_size_edge: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        regress_stress: bool = False,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        otf_graph: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        max_cell_images_per_dim: int = 5,
        encoder_mode: bool = False,
        **kwargs,
    ):
        super().__init__()
        scale_file = f"{MODELS_PROJECT_ROOT}/common/gemnet/gemnet-dT.json"
        assert scale_file is not None, "`scale_file` is required."
        self.encoder_mode = encoder_mode
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        emb_dim_atomic_number = getattr(atom_embedding, "emb_size")
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.max_cell_images_per_dim = max_cell_images_per_dim
        self.otf_graph = otf_graph
        self.regress_stress = regress_stress
        self.angle_edge_emb = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=emb_size_edge + 3, out_features=emb_size_edge),
            paddle.nn.ReLU(),
            paddle.nn.Linear(in_features=emb_size_edge, out_features=emb_size_edge),
        )
        AutomaticFit.reset()
        self.radial_basis = RadialBasis(
            num_radial=num_radial, cutoff=cutoff, rbf=rbf, envelope=envelope
        )
        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial, cutoff=cutoff, rbf=rbf, envelope=envelope
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical, radial_basis=radial_basis_cbf3, cbf=cbf, efficient=True
        )
        self.regress_stress = regress_stress
        self.lattice_out_blocks = paddle.nn.LayerList(
            sublayers=[
                RBFBasedLatticeUpdateBlockFrac(
                    emb_size_edge, activation, emb_size_rbf, emb_size_edge
                )
                for _ in range(num_blocks + 1)
            ]
        )
        self.mlp_rbf_lattice = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_rbf3 = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_cbf3 = EfficientInteractionDownProjection(num_spherical, num_radial, emb_size_cbf)
        self.mlp_rbf_h = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_rbf_out = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.atom_emb = atom_embedding
        self.atom_latent_emb = paddle.nn.Linear(
            in_features=emb_dim_atomic_number + latent_dim, out_features=emb_size_atom
        )
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )
        out_blocks = []
        int_blocks = []
        interaction_block = InteractionBlockTripletsOnly
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    scale_file=scale_file,
                    name=f"IntBlock_{i + 1}",
                )
            )
        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=True,
                    scale_file=scale_file,
                    name=f"OutBlock_{i}",
                )
            )
        self.out_blocks = paddle.nn.LayerList(sublayers=out_blocks)
        self.int_blocks = paddle.nn.LayerList(sublayers=int_blocks)
        self.shared_parameters = [
            (self.mlp_rbf3, self.num_blocks),
            (self.mlp_cbf3, self.num_blocks),
            (self.mlp_rbf_h, self.num_blocks),
            (self.mlp_rbf_out, self.num_blocks + 1),
        ]

    def get_triplets(
        self, edge_index: paddle.Tensor, num_atoms: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: paddle.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: paddle.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: paddle.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index

        # import paddle_sparse
        # from paddle_sparse.tensor import SparseTensor
        # value = paddle.arange(dtype=idx_s.dtype, end=idx_s.shape[0])
        # adj = SparseTensor(
        #     row=idx_t, col=idx_s, value=value, sparse_sizes=(num_atoms, num_atoms)
        # )
        # adj_edges = adj[idx_t]
        # id3_ba = adj_edges.storage.value()
        # id3_ca = adj_edges.storage.row()

        value = paddle.arange(start=1, end=idx_s.shape[0] + 1, dtype=idx_s.dtype)
        # indices = paddle.to_tensor([idx_t, idx_s])
        # adj = sparse_coo_tensor(indices, value, (num_atoms, num_atoms))
        # adj_edges = adj.to_dense()[idx_s].to_sparse_coo(2)
        # id3_ba = adj_edges.values() - 1
        # id3_ca = adj_edges.indices()[0]

        def custom_bincount(x, minlength=0):
            unique, counts = paddle.unique(x, return_counts=True)
            max_val = paddle.max(unique).numpy().item() if len(unique) > 0 else -1
            length = (max_val + 1) if (max_val+1) > minlength else minlength
            result = paddle.zeros([length], dtype='int64')
            if len(unique) > 0:
                result = paddle.scatter_nd(unique.unsqueeze(1), counts, result.shape)
            return result

        n = idx_t.shape[0]
        rows = paddle.arange(n).unsqueeze(1)  # [0,1,2,...,n-1]^T
        cols = paddle.arange(n).unsqueeze(0)  # [0,1,2,...,n-1]
        mask = (idx_t.unsqueeze(1) == idx_t.unsqueeze(0)) & (cols <= rows)
        col = mask.sum(axis=1).astype('int64')-1
        rows = idx_t
        indices = paddle.stack([rows, col], axis=1)

        shape = [num_atoms.item(), col.max().item()+1]
        result = paddle.scatter_nd(indices, value, shape)
        mat = result

        # data_list = []
        # max_data_size = 0
        # for i in range(num_atoms):
        #     data = value[idx_t == i]
        #     data_list.append(data)
        #     if data.shape[0] > max_data_size:
        #         max_data_size = data.shape[0]

        # mat = paddle.zeros((num_atoms, max_data_size), dtype="int64")
        # # mat = paddle.zeros((num_atoms, num_atoms), dtype='int64')
        # for i in range(num_atoms):
        #     data = data_list[i]
        #     mat[i, : data.shape[0]] = data
        
        # if (mat-result).abs().max() > 0:
        #     import pdb;pdb.set_trace()
        
        id3_ba = mat[idx_t][mat[idx_t] > 0] - 1
        tmp_r = paddle.nonzero(mat[idx_t], as_tuple=False)
        id3_ca = tmp_r[:, 0]

        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        num_triplets = custom_bincount(id3_ca, minlength=idx_s.shape[0])
        # num_triplets_api = paddle.bincount(x=id3_ca, minlength=idx_s.shape[0])
        # assert (num_triplets == num_triplets_api).all()
        
        id3_ragged_idx = ragged_range(num_triplets)
        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = paddle.concat(x=[tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self,
        edge_index: paddle.Tensor,
        cell_offsets: paddle.Tensor,
        neighbors: paddle.Tensor,
        edge_dist: paddle.Tensor,
        edge_vector: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running paddle.unique.
        But this does not seem worth it.
        """
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | (cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0)
            | (cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0)
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms
        edge_index_new = edge_index[mask[None, :].expand(shape=[2, -1])].view(2, -1)
        edge_index_cat = paddle.concat(
            x=[
                edge_index_new,
                paddle.stack(x=[edge_index_new[1], edge_index_new[0]], axis=0),
            ],
            axis=1,
        )
        batch_edge = paddle.repeat_interleave(
            x=paddle.arange(end=neighbors.shape[0]), repeats=neighbors
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * paddle.bincount(x=batch_edge, minlength=neighbors.shape[0])
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.shape[1],
        )
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
        edge_dist_new = self.select_symmetric_edges(edge_dist, mask, edge_reorder_idx, False)
        edge_vector_new = self.select_symmetric_edges(edge_vector, mask, edge_reorder_idx, True)
        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def select_edges(
        self,
        edge_index: paddle.Tensor,
        cell_offsets: paddle.Tensor,
        neighbors: paddle.Tensor,
        edge_dist: paddle.Tensor,
        edge_vector: paddle.Tensor,
        cutoff: Optional[float] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff
            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_interaction_graph(
        self,
        cart_coords: paddle.Tensor,
        lattice: paddle.Tensor,
        num_atoms: paddle.Tensor,
        edge_index: paddle.Tensor,
        to_jimages: paddle.Tensor,
        num_bonds: paddle.Tensor,
    ) -> Tuple[
        Tuple[paddle.Tensor, paddle.Tensor],
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
    ]:
        if self.otf_graph:
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords=cart_coords,
                lattice=lattice,
                num_atoms=num_atoms,
                radius=self.cutoff,
                max_num_neighbors_threshold=self.max_neighbors,
                max_cell_images_per_dim=self.max_cell_images_per_dim,
            )
        # import pdb;pdb.set_trace()

        out = get_pbc_distances(
            cart_coords,
            edge_index,
            lattice,
            to_jimages,
            num_atoms,
            num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        edge_index = out["edge_index"]
        D_st = out["distances"]
        V_st = -out["distance_vec"] / D_st[:, None]
        edge_index, cell_offsets, neighbors, D_st, V_st = self.reorder_symmetric_edges(
            edge_index, to_jimages, num_bonds, D_st, V_st
        )
        block_sizes = neighbors // 2
        block_sizes = paddle.masked_select(x=block_sizes, mask=block_sizes > 0)

        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(edge_index, num_atoms=num_atoms.sum())
        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            cell_offsets,
        )

    def forward(
        self,
        z: paddle.Tensor,
        frac_coords: paddle.Tensor,
        atom_types: paddle.Tensor,
        num_atoms: paddle.Tensor,
        batch: paddle.Tensor,
        lengths: Optional[paddle.Tensor] = None,
        angles: Optional[paddle.Tensor] = None,
        edge_index: Optional[paddle.Tensor] = None,
        to_jimages: Optional[paddle.Tensor] = None,
        num_bonds: Optional[paddle.Tensor] = None,
        lattice: Optional[paddle.Tensor] = None,
    ) -> ModelOutput:
        """
        args:
            z: (N_cryst, num_latent)
            frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, ) with D3PM need to use atomic number
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3) (optional, either lengths and angles or lattice must be passed)
            angles: (N_cryst, 3) (optional, either lengths and angles or lattice must be passed)
            edge_index: (2, N_edge) (optional, only needed if self.otf_graph is False)
            to_jimages: (N_edge, 3) (optional, only needed if self.otf_graph is False)
            num_bonds: (N_cryst,) (optional, only needed if self.otf_graph is False)
            lattice: (N_cryst, 3, 3) (optional, either lengths and angles or lattice must be passed)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        if self.otf_graph:
            assert all(
                [edge_index is None, to_jimages is None, num_bonds is None]
            ), "OTF graph construction is active but received input graph information."
        else:
            assert not any(
                [edge_index is None, to_jimages is None, num_bonds is None]
            ), "OTF graph construction is off but received no input graph information."
        assert (angles is None and lengths is None) != (
            lattice is None
        ), "Either lattice or lengths and angles must be provided, not both or none."
        if angles is not None and lengths is not None:
            lattice = lattice_params_to_matrix_paddle(lengths, angles)
        assert lattice is not None
        distorted_lattice = lattice
        pos = frac_to_cart_coords_with_lattice(frac_coords, num_atoms, lattice=distorted_lattice)
        atomic_numbers = atom_types.cast(dtype="int64")
        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            to_jimages,
        ) = self.generate_interaction_graph(
            pos, distorted_lattice, num_atoms, edge_index, to_jimages, num_bonds
        )

        idx_s, idx_t = edge_index
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)
        rbf = self.radial_basis(D_st)
        h = self.atom_emb(atomic_numbers)
        if z is not None:
            z_per_atom = z[batch]
            h = paddle.concat(x=[h, z_per_atom], axis=1)
            h = self.atom_latent_emb(h)
        m = self.edge_emb(h, rbf, idx_s, idx_t)
        batch_edge = batch[edge_index[0]]
        cosines = paddle.nn.functional.cosine_similarity(
            x1=V_st[:, None], x2=distorted_lattice[batch_edge], axis=-1
        )
        m = paddle.concat(x=[m, cosines], axis=-1)
        m = self.angle_edge_emb(m)
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        distance_vec = V_st * D_st[:, None]
        lattice_update = None
        rbf_lattice = self.mlp_rbf_lattice(rbf)
        lattice_update = self.lattice_out_blocks[0](
            edge_emb=m,
            edge_index=edge_index,
            distance_vec=distance_vec,
            lattice=distorted_lattice,
            batch=batch,
            rbf=rbf_lattice,
            normalize_score=True,
        )
        F_fully_connected = paddle.to_tensor(data=0.0, place=distorted_lattice.place)
        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )
            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            F_st += F
            E_t += E
            rbf_lattice = self.mlp_rbf_lattice(rbf)
            lattice_update += self.lattice_out_blocks[i + 1](
                edge_emb=m,
                edge_index=edge_index,
                distance_vec=distance_vec,
                lattice=distorted_lattice,
                batch=batch,
                rbf=rbf_lattice,
                normalize_score=True,
            )
        nMolecules = paddle.max(x=batch) + 1
        if self.encoder_mode:
            return E_t
        E_t = scatter(E_t, batch, dim=0, dim_size=nMolecules, reduce="sum")
        output = dict(energy=E_t, node_embeddings=h)
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        F_t = scatter(F_st_vec, idx_t, dim=0, dim_size=num_atoms.sum(), reduce="add")
        F_t = F_t.squeeze(axis=1)
        output["forces"] = F_t + F_fully_connected
        if self.regress_stress:
            output["stress"] = lattice_update
        return ModelOutput(**output)

    @property
    def num_params(self):
        return sum(p.size for p in self.parameters())
