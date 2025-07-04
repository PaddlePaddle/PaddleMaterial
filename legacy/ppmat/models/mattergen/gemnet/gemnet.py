import math
from typing import Optional

import paddle

from ppmat.utils import paddle_aux  # noqa: F401
from ppmat.utils.crystal import frac_to_cart_coords  # noqa: F401
from ppmat.utils.crystal import get_pbc_distances
from ppmat.utils.crystal import radius_graph_pbc

from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense
from .layers.efficient import EfficientInteractionDownProjection
from .layers.embedding_block import AtomEmbedding
from .layers.embedding_block import EdgeEmbedding
from .layers.interaction_block import InteractionBlockTripletsOnly
from .layers.radial_basis import RadialBasis
from .layers.scaling import AutomaticFit
from .layers.spherical_basis import CircularBasisLayer
from .utils import inner_product_normalized
from .utils import ragged_range
from .utils import repeat_blocks
from .utils import scatter


class SinusoidsEmbedding(paddle.nn.Layer):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(end=self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(axis=-1) * self.frequencies[None, None, :]  # .to(x.devices)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


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

        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing
            block after the bilinear layer.

        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.

        regress_forces: bool
            Whether to predict forces. Default: True
        direct_forces: bool
            If True predict forces based on aggregation of interatomic directions.
            If False predict forces based on negative gradient of energy potential.

        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        aggregate: bool
            Whether to aggregated node outputs
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.
        scale_file: str
            Path to the json file containing the scaling factors.
    """

    def __init__(
        self,
        num_targets: int,
        latent_dim: int,
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
        regress_forces: bool = True,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        otf_graph: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        scale_file: Optional[str] = None,
        index_start: int = 1,
        num_classes: int = 100,
    ):
        super().__init__()
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.num_classes = num_classes
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
        self.mlp_rbf3 = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )
        self.mlp_rbf_h = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.mlp_rbf_out = Dense(num_radial, emb_size_rbf, activation=None, bias=False)
        self.atom_emb = AtomEmbedding(emb_size_atom, index_start=index_start)
        # self.atom_latent_emb = paddle.nn.Linear(
        #     in_features=emb_size_atom + latent_dim,
        #     out_features=emb_size_atom,
        #     bias_attr=False
        # )
        self.atom_latent_emb = Dense(
            emb_size_atom + latent_dim,
            emb_size_atom,
        )
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        self.edge_latent_emb = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=512 + 9, out_features=512),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=512, out_features=512),
            paddle.nn.Silu(),
        )
        self.dist_emb = SinusoidsEmbedding()
        self.rbf_emb = paddle.nn.Sequential(Dense(128 + 9 + self.dist_emb.dim, 128))

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
        lattice_out_blocks = []
        for i in range(num_blocks):
            lattice_out_blocks.append(
                paddle.nn.Linear(in_features=512 + 3, out_features=1, bias_attr=False)
                # Dense(in_features=512+3, out_features=1)
            )

        self.lattice_out_blocks = paddle.nn.LayerList(sublayers=lattice_out_blocks)

        property_blocks = []
        for i in range(num_blocks):
            property_blocks.append(
                paddle.nn.Sequential(
                    paddle.nn.Linear(
                        in_features=512, out_features=512, bias_attr=False
                    ),
                    paddle.nn.Silu(),
                    paddle.nn.Linear(
                        in_features=512, out_features=512, bias_attr=False
                    ),
                    paddle.nn.Silu(),
                    paddle.nn.Linear(
                        in_features=512,
                        out_features=512,
                        weight_attr=paddle.ParamAttr(
                            initializer=paddle.nn.initializer.Constant(value=0.0)
                        ),
                        bias_attr=False,
                    ),
                )
            )
        self.property_blocks = paddle.nn.LayerList(sublayers=property_blocks)

        self.out_blocks = paddle.nn.LayerList(sublayers=out_blocks)
        self.int_blocks = paddle.nn.LayerList(sublayers=int_blocks)
        self.type_out = paddle.nn.Linear(
            in_features=512, out_features=self.num_classes, bias_attr=False
        )
        self.lattice_out = paddle.nn.Linear(
            in_features=512, out_features=9, bias_attr=False
        )

        self.shared_parameters = [
            (self.mlp_rbf3, self.num_blocks),
            (self.mlp_cbf3, self.num_blocks),
            (self.mlp_rbf_h, self.num_blocks),
            (self.mlp_rbf_out, self.num_blocks + 1),
        ]

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index
        # value = paddle.arange(dtype=idx_s.dtype, end=idx_s.shape[0])

        # from utils.paddle_sparse import SparseTensor

        # adj = SparseTensor(
        #     row=idx_t, col=idx_s, value=value, sparse_sizes=(num_atoms, num_atoms)
        # )
        # adj_edges_cus = adj[idx_t]
        # id3_ba_cus = adj_edges_cus.storage.value()
        # id3_ca_cus= adj_edges_cus.storage.row()

        value = paddle.arange(start=1, end=idx_s.shape[0] + 1, dtype=idx_s.dtype)
        from paddle.sparse import sparse_coo_tensor

        indices = paddle.to_tensor([idx_t, idx_s])
        adj = sparse_coo_tensor(indices, value, (num_atoms, num_atoms))
        adj_edges = adj.to_dense()[idx_s].to_sparse_coo(2)
        id3_ba = adj_edges.values() - 1
        id3_ca = adj_edges.indices()[0]

        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]
        num_triplets = paddle.bincount(x=id3_ca, minlength=idx_s.shape[0])
        id3_ragged_idx = ragged_range(num_triplets)
        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = paddle.concat(x=[tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | (cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0)
            | (cell_offsets[:, 0] == 0)
            & (cell_offsets[:, 1] == 0)
            & (cell_offsets[:, 2] < 0)
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
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )
        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def generate_interaction_graph(
        self, cart_coords, lattices, num_atoms, edge_index, to_jimages, num_bonds
    ):
        if self.otf_graph:
            edge_index, to_jimages, num_bonds = radius_graph_pbc(
                cart_coords,
                lattices,
                num_atoms,
                self.cutoff,
                self.max_neighbors,
                device=num_atoms.place,
            )
        out = get_pbc_distances(
            cart_coords,
            edge_index,
            lattices,
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
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )
        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms.sum()
        )
        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    def forward(
        self,
        z,
        frac_coords,
        lattices,
        atom_types,
        num_atoms,
        edge_index=None,
        to_jimages=None,
        num_bonds=None,
        property_emb=None,
        property_mask=None,
    ):
        """
        args:
            z: (N_cryst, num_latent)
            frac_coords: (N_atoms, 3)
            lattices: (N_cryst, 3, 3)
            atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        assert (property_emb is None and property_mask is None) or (
            property_emb is not None and property_mask is not None
        )
        pos = frac_to_cart_coords(frac_coords, num_atoms, lattices=lattices)
        batch = paddle.arange(end=num_atoms.shape[0]).repeat_interleave(
            repeats=num_atoms, axis=0
        )
        atomic_numbers = atom_types
        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(
            pos, lattices, num_atoms, edge_index, to_jimages, num_bonds
        )
        idx_s, idx_t = edge_index
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)
        rbf = self.radial_basis(D_st)
        h = self.atom_emb(atomic_numbers)
        if z is not None:
            z_per_atom = z.repeat_interleave(repeats=num_atoms, axis=0)
            h = paddle.concat(x=[h, z_per_atom], axis=1)
            h = self.atom_latent_emb(h)

        lattices_edges = lattices.repeat_interleave(repeats=neighbors, axis=0)
        d_st = V_st * D_st[:, None]
        d_st = self.dist_emb(d_st)
        rbf = self.rbf_emb(
            paddle.concat([rbf, d_st, lattices_edges.reshape([-1, 9])], axis=1)
        )

        m = self.edge_emb(h, rbf, idx_s, idx_t)
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)

        # angle1 = paddle.nn.functional.cosine_similarity(
        #     lattices_edges[:, :, 0], V_st, axis=1
        # )
        # angle2 = paddle.nn.functional.cosine_similarity(
        #     lattices_edges[:, :, 1], V_st, axis=1
        # )
        # angle3 = paddle.nn.functional.cosine_similarity(
        #     lattices_edges[:, :, 2], V_st, axis=1
        # )
        # angles = paddle.stack(x=[angle1, angle2, angle3], axis=1)

        # pred_l = None
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

            # lattice_score = self.lattice_out_blocks[i](
            #     paddle.concat([m, angles], axis=1)
            # )
            if property_emb is not None:
                property_features = self.property_blocks[i](property_emb)
                property_features = property_features * property_mask
                property_features = paddle.repeat_interleave(
                    property_features, num_atoms, axis=0
                )
                h = h + property_features

            # lattice_scores = lattice_score.squeeze().split(neighbors.numpy().tolist())
            # v_sts = V_st.split(neighbors.numpy().tolist())
            # lattice_sub_layers = []
            # for j in range(len(lattice_scores)):
            #     lattice_score = paddle.diag(lattice_scores[j])
            #     ll = v_sts[j].transpose([1, 0]) @ lattice_score @ v_sts[j]
            #     ll = ll / neighbors[j]
            #     lattice_sub_layers.append(ll)
            # lattice_sub_layers = paddle.stack(x=lattice_sub_layers, axis=0)
            # if pred_l is None:
            #     pred_l = lattice_sub_layers
            # else:
            #     pred_l += lattice_sub_layers

        pred_a = self.type_out(h)

        nMolecules = paddle.max(x=batch) + 1
        E_t = scatter(E_t, batch, dim=0, dim_size=nMolecules, reduce="mean")
        E_t = E_t.reshape([-1, 3, 3])
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        F_t = scatter(F_st_vec, idx_t, dim=0, dim_size=num_atoms.sum(), reduce="add")
        F_t = F_t.squeeze(axis=1)
        return E_t, F_t, pred_a

    @property
    def num_params(self):
        return sum(p.size for p in self.parameters())
