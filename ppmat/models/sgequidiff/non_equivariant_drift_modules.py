"""非等变漂移模块：GNN（基于 PBC 消息传递）和 CSPNet（来自 DiffCSP）。"""
import dataclasses
import math
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import ppmat.models.sgequidiff.global_vars as global_vars
from ppmat.models.sgequidiff.constants import lattice_parameter_ranges, NUM_ELEMENTS
from ppmat.models.sgequidiff.data_utils import (
    construct_fully_connected_graphs_with_periodic_boundaries,
    frac_to_cart_coords,
    ocp_get_pbc_distances,
)
from ppmat.models.sgequidiff.submodules import (
    VariancePreservingAggregation,
    Swish,
    GraphNorm,
)
from ppmat.models.sgequidiff.scatter_utils import safe_scatter as paddle_scatter


class FourierTimeEmbeddings(nn.Layer):
    """Fourier 时间嵌入。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: paddle.Tensor) -> paddle.Tensor:
        """(n,) -> (n, dim)"""
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = paddle.exp(
            paddle.arange(half_dim, dtype=paddle.float32) * -embeddings
        )
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = paddle.concat([embeddings.sin(), embeddings.cos()], axis=-1)
        return embeddings  # (n, dim)

class NonEquivariantDriftModule(nn.Layer):
    """基类，包含平面波和 Fourier 特征提取工具。"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_plane_wave_frequencies(
        num_freqs: int,
        max_freq: int = 512,
        fourier_scale: float = 1.0,
        isotropic_plane_waves: bool = False,
    ) -> paddle.Tensor:
        """返回 (3, num_freqs) 平面波频率张量。"""
        if isotropic_plane_waves:
            plane_wave_freqs = paddle.linspace(
                1, num_freqs, num_freqs
            ).unsqueeze(0).expand([3, num_freqs])
        else:
            freqs_1d_grid = paddle.linspace(-max_freq, max_freq, 1 + 2 * max_freq)
            freqs_1d_grid = freqs_1d_grid[freqs_1d_grid != 0.0]
            normal = paddle.distribution.Normal(
                paddle.to_tensor([0.0]), paddle.to_tensor([fourier_scale])
            )
            probs = normal.log_prob(freqs_1d_grid).exp()

            plane_wave_freqs = paddle.empty([3, 0])
            samples_per_iter = 2 * num_freqs
            max_iters = 100
            iteration = 0
            while plane_wave_freqs.shape[-1] < num_freqs and iteration <= max_iters:
                iteration += 1
                kx = freqs_1d_grid[paddle.multinomial(probs, num_samples=samples_per_iter, replacement=True)]
                ky = freqs_1d_grid[paddle.multinomial(probs, num_samples=samples_per_iter, replacement=True)]
                kz = freqs_1d_grid[paddle.multinomial(probs, num_samples=samples_per_iter, replacement=True)]
                plane_wave_freqs = paddle.concat(
                    [plane_wave_freqs, paddle.stack([kx, ky, kz])], axis=-1
                )
                # unique argsort: keep unique columns
                _, unique_idx = paddle.unique(plane_wave_freqs, axis=-1, return_index=True)
                plane_wave_freqs = plane_wave_freqs[:, unique_idx]
        return plane_wave_freqs[:, :num_freqs]

    @staticmethod
    def plane_wave_fourier_features(
        x: paddle.Tensor, plane_wave_freqs: paddle.Tensor
    ) -> paddle.Tensor:
        """(n, 3) x (3, num_freqs) -> (n, 2*num_freqs)"""
        v = 2 * math.pi * x @ plane_wave_freqs
        return paddle.concat([v.sin(), v.cos()], axis=-1)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class TorusMLP(NonEquivariantDriftModule):
    """基于 Fourier 特征的简单 MLP 漂移模型。"""

    def __init__(
        self,
        time_embedder: FourierTimeEmbeddings,
        num_plane_wave_freqs: int,
    ):
        super().__init__()
        self.time_embedder = time_embedder
        self.num_plane_wave_freqs = num_plane_wave_freqs
        plane_wave_freqs = self.get_plane_wave_frequencies(num_freqs=num_plane_wave_freqs)
        self.register_buffer("plane_wave_freqs", plane_wave_freqs)
        # (3, num_plane_wave_freqs)
        self.layers = nn.Sequential(
            nn.Linear(2 * self.num_plane_wave_freqs + time_embedder.dim, 128),
            nn.Silu(),
            nn.Linear(128, 128),
            nn.Silu(),
            nn.Linear(128, 128),
            nn.Silu(),
            nn.Linear(128, 128),
            nn.Silu(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        return self.layers(
            paddle.concat(
                [
                    self.plane_wave_fourier_features(frac_coords, self.plane_wave_freqs),
                    time_embeddings,
                ],
                axis=-1,
            )
        )

class GaussianSmearing(nn.Layer):
    """高斯距离扩展。"""

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = paddle.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: paddle.Tensor) -> paddle.Tensor:
        dist = dist.reshape([-1, 1]) - self.offset.reshape([1, -1])
        return paddle.exp(self.coeff * dist ** 2)

def custom_he_orthogonal_(weight: paddle.Tensor, gain: float = 1.0) -> paddle.Tensor:
    """He 初始化 + 正交化。"""
    with paddle.no_grad():
        fan_in = weight.shape[1]
        assert fan_in > 1

        nn.initializer.Orthogonal()(weight)
        eps = 1e-6
        mean = weight.mean(axis=1, keepdim=True)
        var = weight.std(axis=1, keepdim=True)
        result = gain * math.sqrt(1 / fan_in) * ((weight - mean) / (var + eps).sqrt())
        paddle.assign(result, weight)
    return weight

class NodeAndEdgeEmbedder(nn.Layer):
    """节点和边的初始嵌入模块。"""

    def __init__(
        self,
        num_cartesian_distance_gaussians: int,
        edge_hidden_dim: int,
        atom_hidden_dim: int,
        fourier_frac_edge_dim: int,
        gaussian_cart_edge_dim: int,
        time_emb_dim: int,
        activation: nn.Layer,
        use_frac_coords_in_node_emb: bool,
    ):
        super().__init__()
        self.act = activation
        self.use_frac_coords_in_node_emb = use_frac_coords_in_node_emb

        if use_frac_coords_in_node_emb:
            self.frac_pos_emb = nn.Linear(fourier_frac_edge_dim, atom_hidden_dim)
            self.ele_emb = nn.Linear(
                global_vars.embedding_tools.element_embedding_length, atom_hidden_dim
            )
            self.atom_emb1 = nn.Linear(2 * atom_hidden_dim, atom_hidden_dim)
        else:
            self.atom_emb1 = nn.Linear(
                global_vars.embedding_tools.element_embedding_length, atom_hidden_dim
            )
        self.atom_emb2 = nn.Linear(atom_hidden_dim + time_emb_dim, atom_hidden_dim)

        self.edge_emb1 = nn.Linear(
            fourier_frac_edge_dim + gaussian_cart_edge_dim + 6, edge_hidden_dim, bias_attr=False
        )
        self.edge_emb2 = nn.Linear(edge_hidden_dim, edge_hidden_dim, bias_attr=False)
        self._reset_parameters()

    def _reset_parameters(self):
        custom_he_orthogonal_(self.atom_emb1.weight, gain=2.0)
        paddle.assign(paddle.zeros_like(self.atom_emb1.bias), self.atom_emb1.bias)
        custom_he_orthogonal_(self.atom_emb2.weight, gain=2.0)
        paddle.assign(paddle.zeros_like(self.atom_emb2.bias), self.atom_emb2.bias)
        custom_he_orthogonal_(self.edge_emb1.weight, gain=2.0)
        custom_he_orthogonal_(self.edge_emb2.weight, gain=2.0)

    def forward(
        self,
        element_indices: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        fourier_relative_frac_pos: paddle.Tensor,
        gaussian_cart_dists: paddle.Tensor,
        normed_lattice_params: paddle.Tensor,
        fourier_atom_frac_pos: Optional[paddle.Tensor] = None,
    ) -> dict:
        e = self.edge_emb1(
            paddle.concat([fourier_relative_frac_pos, gaussian_cart_dists, normed_lattice_params], axis=-1)
        )
        e = self.act(e)
        e = self.act(self.edge_emb2(e))

        if self.use_frac_coords_in_node_emb:
            frac_pos_emb = self.act(self.frac_pos_emb(fourier_atom_frac_pos))
            ele_emb_out = self.act(
                self.ele_emb(
                    global_vars.embedding_tools.get_element_embedding(1 + element_indices)
                )
            )
            h = self.atom_emb1(paddle.concat([frac_pos_emb, ele_emb_out], axis=-1))
        else:
            h = self.atom_emb1(
                global_vars.embedding_tools.get_element_embedding(1 + element_indices)
            )
        h = self.act(h)
        h = self.act(self.atom_emb2(paddle.concat([h, time_embeddings], axis=-1)))
        return {"h": h, "e": e}

class InteractionBlock(nn.Layer):
    """自定义消息传递 GNN 层。"""

    def __init__(
        self,
        hidden_channels: int,
        edge_hidden_dim: int,
        activation: nn.Layer,
        graph_norm: bool = True,
        use_vpa: bool = True,
    ):
        super().__init__()
        self.act = activation
        self.hidden_channels = hidden_channels
        self.use_graph_norm = graph_norm
        self.use_vpa = use_vpa

        if use_vpa:
            self.aggregator = VariancePreservingAggregation()
        if graph_norm:
            self.graph_norm = GraphNorm(hidden_channels)

        self.lin_geom = nn.Linear(
            edge_hidden_dim + 2 * hidden_channels, hidden_channels, bias_attr=False
        )
        self.lin_h = nn.Linear(hidden_channels, hidden_channels)
        self.out_layer = nn.Linear(hidden_channels, hidden_channels)
        self.skipinit_gain = self.create_parameter(
            [], default_initializer=nn.initializer.Constant(0.0)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        custom_he_orthogonal_(self.lin_geom.weight, gain=4.0)
        custom_he_orthogonal_(self.out_layer.weight, gain=3.0)
        paddle.assign(paddle.zeros_like(self.out_layer.bias), self.out_layer.bias)
        custom_he_orthogonal_(self.lin_h.weight, gain=3.0)
        paddle.assign(paddle.zeros_like(self.lin_h.bias), self.lin_h.bias)

    def forward(
        self,
        h: paddle.Tensor,
        edge_index: paddle.Tensor,
        e: paddle.Tensor,
        map_node_to_graph: Optional[paddle.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> paddle.Tensor:
        """消息传递 + 聚合。"""
        src_ids = edge_index[0]  # source
        dst_ids = edge_index[1]  # destination

        # 拼接边特征
        e_full = paddle.concat([e, h[src_ids], h[dst_ids]], axis=1)
        e_full = self.act(self.lin_geom(e_full))
        # (n_edges, hidden_channels)

        # 消息: m_ij = h_j * e_ij（source * edge）
        messages = h[src_ids] * e_full  # (n_edges, hidden_channels)

        # 聚合
        n_nodes = h.shape[0]
        if self.use_vpa:
            h_agg = self.aggregator(messages, dst_ids, dim_size=n_nodes)
        else:
            h_agg = paddle_scatter(messages, dst_ids, dim=0, dim_size=n_nodes, reduce="sum")

        if self.use_graph_norm:
            h_agg = self.graph_norm(h_agg, map_node_to_graph, num_graphs)
            h_agg = self.act(h_agg)
        h_agg = self.act(self.lin_h(h_agg))
        h_agg = self.act(self.out_layer(h_agg))

        return self.skipinit_gain * h_agg

@dataclasses.dataclass
@dataclasses.dataclass
class GNNConfig:
    num_plane_wave_freqs: int = 64
    num_cartesian_distance_gaussians: int = 64
    edge_hidden_dim: int = 256
    atom_hidden_dim: int = 256
    use_vpa: bool = True
    use_graph_norm: bool = True
    num_msg_pass_steps: int = 5
    cutoff: float = 7.0
    use_frac_coords_in_node_emb: bool = False
    dataset_name: str = "mp_20"

class GNN(NonEquivariantDriftModule):
    """GNN 非等变漂移模块。"""

    def __init__(self, config: GNNConfig, time_embedder: FourierTimeEmbeddings):
        super().__init__()
        self.config = config
        self.time_embedder = time_embedder
        self.num_plane_wave_freqs = config.num_plane_wave_freqs
        self.num_cartesian_distance_gaussians = config.num_cartesian_distance_gaussians
        self.edge_hidden_dim = config.edge_hidden_dim
        self.atom_hidden_dim = config.atom_hidden_dim
        self.use_graph_norm = config.use_graph_norm
        self.use_vpa = config.use_vpa
        self.num_msg_pass_steps = config.num_msg_pass_steps
        self.cutoff = config.cutoff

        plane_wave_freqs = self.get_plane_wave_frequencies(
            num_freqs=self.num_plane_wave_freqs
        )
        self.register_buffer("plane_wave_freqs", plane_wave_freqs)

        self.gaussian_smearing = GaussianSmearing(
            0.0, self.cutoff, self.num_cartesian_distance_gaussians
        )
        self.activation = Swish()
        self.embed_block = NodeAndEdgeEmbedder(
            self.num_cartesian_distance_gaussians,
            self.edge_hidden_dim,
            self.atom_hidden_dim,
            2 * self.num_plane_wave_freqs,
            self.num_cartesian_distance_gaussians,
            self.time_embedder.dim,
            self.activation,
            self.config.use_frac_coords_in_node_emb,
        )
        self.interaction_blocks = nn.LayerList(
            [
                InteractionBlock(
                    hidden_channels=self.atom_hidden_dim,
                    edge_hidden_dim=self.edge_hidden_dim,
                    activation=self.activation,
                    graph_norm=self.use_graph_norm,
                    use_vpa=self.use_vpa,
                )
                for _ in range(self.num_msg_pass_steps)
            ]
        )
        self.mlp_skip_co = nn.Linear(
            (self.num_msg_pass_steps + 1) * self.atom_hidden_dim,
            self.atom_hidden_dim,
        )
        self.mlp_out = nn.Linear(self.atom_hidden_dim, 3)

    def forward(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        differentiate_graph_construction: bool = False,
    ) -> paddle.Tensor:
        """GNN 前向传播，返回 (n_atoms, 3)。"""
        if not differentiate_graph_construction:
            with paddle.no_grad():
                (
                    map_atom_to_xtal,
                    edge_index,
                    relative_fractional_positions,
                    cartesian_distances,
                    num_edges_per_crystal,
                ) = self.construct_graphs(frac_coords, n_atoms_per_xtal, lattice_matrices)
                fourier_relative_frac_pos = self.plane_wave_fourier_features(
                    relative_fractional_positions, self.plane_wave_freqs
                )
                gaussian_smeared_cart_dists = self.gaussian_smearing(cartesian_distances)
                normed_lattice_params = self.norm_lattice_params(
                    lattice_lengths, lattice_angles
                ).repeat_interleave(num_edges_per_crystal, axis=0)
        else:
            (
                map_atom_to_xtal,
                edge_index,
                relative_fractional_positions,
                cartesian_distances,
                num_edges_per_crystal,
            ) = self.construct_graphs(frac_coords, n_atoms_per_xtal, lattice_matrices)
            fourier_relative_frac_pos = self.plane_wave_fourier_features(
                relative_fractional_positions, self.plane_wave_freqs
            )
            gaussian_smeared_cart_dists = self.gaussian_smearing(cartesian_distances)
            normed_lattice_params = self.norm_lattice_params(
                lattice_lengths, lattice_angles
            ).repeat_interleave(num_edges_per_crystal, axis=0)

        if self.config.use_frac_coords_in_node_emb:
            fourier_atom_frac_pos = self.plane_wave_fourier_features(
                frac_coords, self.plane_wave_freqs
            )
        else:
            fourier_atom_frac_pos = None

        embed_out = self.embed_block(
            element_indices,
            time_embeddings,
            fourier_relative_frac_pos,
            gaussian_smeared_cart_dists,
            normed_lattice_params,
            fourier_atom_frac_pos,
        )
        node_latents = embed_out["h"]
        edge_latents = embed_out["e"]

        skip_connection_elements = []
        for interaction in self.interaction_blocks:
            skip_connection_elements.append(node_latents)
            node_latents = node_latents + interaction(
                node_latents,
                edge_index,
                edge_latents,
                map_atom_to_xtal,
                lattice_matrices.shape[0],
            )

        skip_connection_elements.append(node_latents)
        node_latents = self.mlp_skip_co(paddle.concat(skip_connection_elements, axis=1))
        out_vectors = self.mlp_out(node_latents)
        return out_vectors

    @staticmethod
    def construct_graphs(
        frac_coords: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
    ) -> tuple:
        """构建 PBC 图结构。"""
        n_crystals = lattice_matrices.shape[0]
        n_nodes = frac_coords.shape[0]
        atom_counts = n_atoms_per_xtal.cast("int64").reshape([-1])
        cumulative = paddle.cumsum(atom_counts, axis=0)
        node_ids = paddle.arange(n_nodes, dtype=cumulative.dtype).reshape([-1, 1])
        map_atom_to_xtal = (node_ids >= cumulative.reshape([1, -1])).cast("int64").sum(axis=1)
        map_atom_to_xtal = paddle.clip(map_atom_to_xtal, min=0, max=n_crystals - 1)

        cart_coords = frac_to_cart_coords(
            frac_coords, n_atoms_per_xtal, lattice_matrix=lattice_matrices
        )
        (
            destination_ids,
            source_ids,
            source_node_image_offsets,
            num_edges_per_crystal,
        ) = construct_fully_connected_graphs_with_periodic_boundaries(
            cart_coords=cart_coords,
            lattice_matrix=lattice_matrices,
            num_nodes_per_crystal=n_atoms_per_xtal,
        )
        out = ocp_get_pbc_distances(
            coords=cart_coords,
            source_id=source_ids,
            destination_id=destination_ids,
            lattice=lattice_matrices,
            pbc_frac_offsets_per_source_node=source_node_image_offsets,
            num_edges_per_crystal=num_edges_per_crystal,
        )
        cartesian_distances = out["distances"]
        edge_index = out["edge_index"]
        relative_fractional_positions = frac_coords[source_ids] - frac_coords[destination_ids]
        return (
            map_atom_to_xtal,
            edge_index,
            relative_fractional_positions,
            cartesian_distances,
            num_edges_per_crystal,
        )

    @paddle.no_grad()
    def norm_lattice_params(
        self, lattice_lengths: paddle.Tensor, lattice_angles: paddle.Tensor
    ) -> paddle.Tensor:
        """将晶格参数规范化到 [-1, 1]。"""
        param_ranges = lattice_parameter_ranges[self.config.dataset_name]
        min_len = param_ranges["min_lattice_length"]
        max_len = param_ranges["max_lattice_length"]
        min_ang = param_ranges["min_lattice_angle"]
        max_ang = param_ranges["max_lattice_angle"]

        normed_lengths = 2.0 * (lattice_lengths - min_len) / (max_len - min_len) - 1.0
        normed_angles = 2.0 * (lattice_angles - min_ang) / (max_ang - min_ang) - 1.0
        return paddle.concat([normed_lengths, normed_angles], axis=-1)

class SinusoidsEmbedding(nn.Layer):
    """正弦嵌入。"""

    def __init__(self, n_frequencies: int = 10, n_space: int = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(self.n_frequencies, dtype=paddle.float32)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        emb = x.unsqueeze(-1) * self.frequencies.reshape([1, 1, -1])
        emb = emb.reshape([-1, self.n_frequencies * self.n_space])
        emb = paddle.concat([emb.sin(), emb.cos()], axis=-1)
        return emb.detach()

class CSPLayer(nn.Layer):
    """CSPNet 消息传递层。"""

    def __init__(
        self,
        hidden_dim: int = 128,
        act_fn: nn.Layer = None,
        dis_emb=None,
        ln: bool = False,
    ):
        super().__init__()
        if act_fn is None:
            act_fn = nn.Silu()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattice_rep: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge2graph: paddle.Tensor,
        frac_diff: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        hi = node_features[edge_index[0]]
        hj = node_features[edge_index[1]]
        if frac_diff is None:
            xi = frac_coords[edge_index[0]]
            xj = frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        lattice_rep_edges = lattice_rep[edge2graph]
        edges_input = paddle.concat([hi, hj, lattice_rep_edges, frac_diff], axis=1)
        return self.edge_mlp(edges_input)

    def node_model(
        self,
        node_features: paddle.Tensor,
        edge_features: paddle.Tensor,
        edge_index: paddle.Tensor,
    ) -> paddle.Tensor:
        agg = paddle_scatter(
            edge_features,
            edge_index[0],
            dim=0,
            dim_size=node_features.shape[0],
            reduce="mean",
        )
        agg = paddle.concat([node_features, agg], axis=1)
        return self.node_mlp(agg)

    def forward(
        self,
        node_features: paddle.Tensor,
        frac_coords: paddle.Tensor,
        lattices: paddle.Tensor,
        edge_index: paddle.Tensor,
        edge2graph: paddle.Tensor,
        frac_diff: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output

@dataclasses.dataclass
@dataclasses.dataclass
class CSPNetConfig:
    hidden_dim: int = 256
    num_msg_pass_steps: int = 6
    ln: bool = False
    act_fn: str = "silu"
    dis_emb: str = "sin"
    num_freqs: int = 128
    dense: bool = False

class CSPNet(nn.Layer):
    """DiffCSP 架构的 CSPNet。"""

    def __init__(self, config: CSPNetConfig, time_embedder: nn.Layer):
        super().__init__()
        latent_dim = time_embedder.dim
        num_layers = config.num_msg_pass_steps
        max_atoms = NUM_ELEMENTS
        hidden_dim = config.hidden_dim
        num_freqs = config.num_freqs
        act_fn_str = config.act_fn
        dis_emb_str = config.dis_emb
        dense = config.dense
        ln = config.ln

        self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        if act_fn_str == "silu":
            self.act_fn = nn.Silu()
        if dis_emb_str == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb_str == "none":
            self.dis_emb = None

        for i in range(num_layers):
            self.add_sublayer(
                f"csp_layer_{i}",
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln),
            )
        self.num_layers = num_layers
        self.dense = dense

        hidden_dim_before_out = hidden_dim
        if self.dense:
            hidden_dim_before_out = hidden_dim_before_out * (num_layers + 1)

        self.coord_out = nn.Linear(hidden_dim_before_out, 3, bias_attr=False)
        self.ln = ln
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

    def gen_edges(
        self, num_atoms: paddle.Tensor, frac_coords: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """生成全连接边。"""
        lis = [
            paddle.ones([n, n], dtype=paddle.float32)
            for n in num_atoms.numpy().tolist()
        ]
        fc_graph = paddle.block_diag(lis)
        fc_edges = paddle.nonzero(fc_graph).T
        # (2, n_edges)
        frac_diff = (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        return fc_edges, frac_diff

    def forward(
        self,
        frac_coords: paddle.Tensor,
        element_indices: paddle.Tensor,
        n_atoms_per_xtal: paddle.Tensor,
        lattice_matrices: paddle.Tensor,
        lattice_lengths: paddle.Tensor,
        lattice_angles: paddle.Tensor,
        time_embeddings: paddle.Tensor,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        """CSPNet 前向传播。"""
        n_crystals = n_atoms_per_xtal.shape[0]
        node2graph = paddle.arange(n_crystals).repeat_interleave(n_atoms_per_xtal, axis=0)
        atom_types = element_indices
        lattices = paddle.concat([lattice_lengths, lattice_angles], axis=-1)

        edges, frac_diff = self.gen_edges(n_atoms_per_xtal, frac_coords)
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)
        node_features = paddle.concat([node_features, time_embeddings], axis=-1)
        node_features = self.atom_latent_emb(node_features)

        h_list = [node_features]
        for i in range(self.num_layers):
            # self.sublayers(name) 在 Paddle 中返回所有子层列表，不能按名称索引；改用 getattr
            node_features = getattr(self, f"csp_layer_{i}")(
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff
            )
            if i != self.num_layers - 1:
                h_list.append(node_features)

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        h_list.append(node_features)

        if self.dense:
            node_features = paddle.concat(h_list, axis=-1)

        return self.coord_out(node_features)
