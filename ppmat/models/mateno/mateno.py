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

import paddle
from paddle_scatter import scatter

from ppmat.datasets.graph_utils.infgcn_graph_utils import radius, radius_graph

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.math import soft_one_hot_linspace
from ppmat.models.common.orbital import BroadcastGTOTensor
from ppmat.models.common.activation import ScalarActivation
from ppmat.models.common.activation import NormActivation


class LogGaussianOrbital(paddle.nn.Layer):
    """
    Gaussian orbital with log-spaced bases for better multi-scale coverage.
    """

    def __init__(self, gauss_start: float, gauss_end: float, num_gauss: int, lmax: int = 7):
        super().__init__()
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.num_gauss = num_gauss
        self.lmax = lmax

        self.lc2lcm = BroadcastGTOTensor(lmax, num_gauss, src="lc", dst="lcm")
        self.m2lcm = BroadcastGTOTensor(lmax, num_gauss, src="m", dst="lcm")

        # log-spaced radial bases (was linear in the reference Paddle implementation)
        self.register_buffer(
            name="gauss",
            tensor=paddle.logspace(
                start=paddle.log(paddle.to_tensor(gauss_start)),
                stop=paddle.log(paddle.to_tensor(gauss_end)),
                num=num_gauss,
                base=paddle.exp(paddle.to_tensor(1.0)),
            ),
        )
        self.register_buffer(name="lognorm", tensor=self._generate_lognorm())

    def _generate_lognorm(self):
        power = (paddle.arange(end=self.lmax + 1) + 1.5).unsqueeze(axis=-1)
        numerator = power * paddle.log(x=2 * self.gauss).unsqueeze(axis=0) + paddle.log(
            paddle.to_tensor(2.0)
        )
        denominator = paddle.lgamma(x=power)
        lognorm = (numerator - denominator) / 2
        return lognorm.view(-1)

    def forward(self, vec: paddle.Tensor) -> paddle.Tensor:
        r = vec.norm(axis=-1) + 1e-8
        spherical = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            vec / r[..., None],
            normalize=False,
            normalization="integral",
        )
        r = r.unsqueeze(axis=-1)
        lognorm = self.lognorm * paddle.ones_like(x=r)
        exponent = -self.gauss * (r * r)
        poly = paddle.arange(dtype="float32", end=self.lmax + 1) * paddle.log(x=r)
        log = exponent.unsqueeze(axis=-2) + poly.unsqueeze(axis=-1)
        radial = paddle.exp(x=log.view(*tuple(log.shape)[:-2], -1) + lognorm)
        return self.lc2lcm(radial) * self.m2lcm(spherical)


class EnhancedGCNLayer(paddle.nn.Layer):
    """
    GCN layer with optional atom-type-aware modulation of tensor-product weights.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_edge,
        radial_embed_size,
        num_radial_layer,
        radial_hidden_size,
        n_atom_type,
        is_fc=True,
        use_sc=False,
        irrep_normalization="component",
        path_normalization="element",
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.radial_embed_size = radial_embed_size
        self.is_fc = is_fc
        self.use_sc = use_sc

        if self.is_fc:
            self.tp = o3.FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        else:
            instr = [
                (i_1, i_2, i_out, "uvu", True)
                for i_1, (_, ir_1) in enumerate(self.irreps_in)
                for i_2, (_, ir_edge) in enumerate(self.irreps_edge)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_out in ir_1 * ir_edge
            ]
            self.tp = o3.TensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                instr,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )

        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(radial_embed_size, radial_hidden_size),
            paddle.nn.Silu(),
            *sum(
                [
                    [paddle.nn.Linear(radial_hidden_size, radial_hidden_size), paddle.nn.Silu()]
                    for _ in range(num_radial_layer - 1)
                ],
                [],
            ),
            paddle.nn.Linear(radial_hidden_size, self.tp.weight_numel),
        )

        self.sc = None
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)

        # atom-type conditioning branch
        self.atom_attr_embedding = paddle.nn.Sequential(
            paddle.nn.Linear(n_atom_type, radial_hidden_size),
            paddle.nn.Silu(),
            paddle.nn.Linear(radial_hidden_size, radial_hidden_size),
        )
        self.atom_edge_influence = paddle.nn.Sequential(
            paddle.nn.Linear(radial_hidden_size + radial_embed_size, radial_hidden_size),
            paddle.nn.Silu(),
            paddle.nn.Linear(radial_hidden_size, radial_hidden_size),
        )
        self.weight_modulator = paddle.nn.Sequential(
            paddle.nn.Linear(radial_hidden_size, radial_hidden_size // 2),
            paddle.nn.Silu(),
            paddle.nn.Linear(radial_hidden_size // 2, 1),
            paddle.nn.Sigmoid(),
        )

    def forward(
        self,
        edge_index,
        node_feat,
        edge_feat,
        edge_embed,
        node_attrs=None,
        dim_size=None,
    ):
        src, dst = edge_index
        base_weights = self.fc(edge_embed)

        if node_attrs is not None:
            atom_features = self.atom_attr_embedding(node_attrs)
            atom_edge_features = paddle.concat(x=[atom_features[src], edge_embed], axis=-1)
            edge_modulation = self.atom_edge_influence(atom_edge_features)
            weight_scaling = self.weight_modulator(edge_modulation)
            adjusted_weights = base_weights * (1.0 + weight_scaling)
        else:
            adjusted_weights = base_weights

        out = self.tp(node_feat[src], edge_feat, weight=adjusted_weights)
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce="sum")
        if self.use_sc and self.sc is not None:
            out = out + self.sc(node_feat)
        return out


class MatENO(paddle.nn.Layer):
    """
    Paddle version of the optimized InfGCN: wider embedding, log-spaced orbitals,
    atom-type-aware message passing, and grid residue.
    """

    def __init__(
        self,
        n_atom_type: int,
        num_radial: int,
        num_spherical: int,
        radial_embed_size: int,
        radial_hidden_size: int,
        num_radial_layer: int = 2,
        num_gcn_layer: int = 3,
        cutoff: float = 3.0,
        grid_cutoff: float = 3.0,
        gauss_start: float = 0.5,
        gauss_end: float = 5.0,
        activation: str = "norm",
        residual: bool = True,
        pbc: bool = False,
        is_fc: bool = True,
        embedding_dim: int = 512,
        max_num_neighbors: int = 32,
        target_name: str = "density",
        label_key: str = "density",
        mask_key: str = "density_mask",
        loss_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        assert activation in ["scalar", "norm"]

        self.n_atom_type = n_atom_type
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_radial_layer = num_radial_layer
        self.num_gcn_layer = num_gcn_layer
        self.cutoff = cutoff
        self.grid_cutoff = grid_cutoff
        self.activation = activation
        self.residual = residual
        self.pbc = pbc
        self.max_num_neighbors = max_num_neighbors
        self.target_name = target_name
        self.label_key = label_key
        self.mask_key = mask_key
        self.loss_eps = loss_eps
        self._criterion = paddle.nn.MSELoss(reduction="mean")

        self.embedding = paddle.nn.Embedding(num_embeddings=n_atom_type, embedding_dim=embedding_dim)
        self._init_embeddings()

        self.irreps_sh = o3.Irreps.spherical_harmonics(num_spherical, p=1)
        self.irreps_feat = (self.irreps_sh * num_radial).sort().irreps.simplify()

        self.gcns = paddle.nn.LayerList(
            [
                EnhancedGCNLayer(
                    irreps_in=(f"{embedding_dim}x0e" if i == 0 else self.irreps_feat),
                    irreps_out=self.irreps_feat,
                    irreps_edge=self.irreps_sh,
                    radial_embed_size=radial_embed_size,
                    num_radial_layer=num_radial_layer,
                    radial_hidden_size=radial_hidden_size,
                    n_atom_type=n_atom_type,
                    is_fc=(True if i == 0 else is_fc),
                    use_sc=False,
                )
                for i in range(num_gcn_layer)
            ]
        )

        self.act = (
            ScalarActivation(self.irreps_feat, paddle.nn.functional.silu, paddle.nn.functional.sigmoid)
            if activation == "scalar"
            else NormActivation(self.irreps_feat)
        )

        self.residue = None
        if self.residual:
            self.residue = EnhancedGCNLayer(
                irreps_in=self.irreps_feat,
                irreps_out=o3.Irreps("0e"),
                irreps_edge=self.irreps_sh,
                radial_embed_size=radial_embed_size,
                num_radial_layer=num_radial_layer,
                radial_hidden_size=radial_hidden_size,
                n_atom_type=n_atom_type,
                is_fc=True,
                use_sc=False,
            )

        self.orbital = LogGaussianOrbital(gauss_start, gauss_end, num_radial, num_spherical)

    def _init_embeddings(self):
        paddle.nn.initializer.XavierUniform()(self.embedding.weight)

    def forward(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            return self._forward_with_batch(args[0])
        return self._forward_density(*args, **kwargs)

    def _forward_with_batch(self, batch):
        graph = batch["graph"]
        density = batch.get(self.label_key, None)
        grid = batch["grid_coord"]
        infos = batch.get("infos", None)
        mask = batch.get(self.mask_key, None)

        device = paddle.get_device()
        graph = graph.to(device)
        grid = grid.astype("float32").to(device)
        if density is not None:
            density = density.astype("float32").to(device)
        if mask is not None:
            mask = mask.astype("float32").to(device)
        prepared_infos = self._prepare_infos(infos, device)

        pred = self._forward_density(graph.x, graph.pos, grid, graph.batch, prepared_infos)

        loss_dict = {}
        masked_pred = pred
        if mask is not None:
            mask = mask.astype(pred.dtype)
            masked_pred = pred * mask

        if density is not None:
            if mask is not None:
                label_masked = density * mask
                denom = paddle.sum(mask) + self.loss_eps
                loss = paddle.sum((masked_pred - label_masked) ** 2) / denom
                mae = paddle.sum(paddle.abs(masked_pred - label_masked)) / denom
            else:
                label_masked = density
                loss = self._criterion(pred, label_masked)
                mae = paddle.mean(paddle.abs(pred - label_masked))
            loss_dict["loss"] = loss
            loss_dict["mae"] = mae

        pred_dict = {self.target_name: masked_pred}
        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def _prepare_infos(self, infos, device):
        if infos is None:
            return None
        prepared_infos = []
        for info in infos:
            cur = dict(info) if isinstance(info, dict) else info
            if isinstance(cur, dict) and "cell" in cur and hasattr(cur["cell"], "to"):
                cur["cell"] = cur["cell"].to(device)
            prepared_infos.append(cur)
        return prepared_infos

    def _forward_density(self, atom_types, atom_coord, grid, batch, infos):
        cell = None
        if infos is not None and len(infos) > 0:
            first_info = infos[0]
            if isinstance(first_info, dict) and "cell" in first_info:
                cell = paddle.stack(x=[info["cell"] for info in infos], axis=0).astype(atom_coord.dtype)

        feat = self.embedding(atom_types)
        node_attrs = paddle.nn.functional.one_hot(atom_types, self.n_atom_type).astype("float32")

        edge_index = radius_graph(
            atom_coord,
            self.cutoff,
            batch,
            loop=False,
            max_num_neighbors=self.max_num_neighbors,
        )
        src, dst = edge_index
        edge_vec = atom_coord[src] - atom_coord[dst]
        edge_len = paddle.norm(edge_vec, axis=-1) + 1e-8
        edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical + 1)),
            edge_vec / edge_len[..., None],
            normalize=False,
            normalization="integral",
        )
        edge_embed = soft_one_hot_linspace(
            edge_len,
            start=0.0,
            end=self.cutoff,
            number=self.radial_embed_size,
            basis="gaussian",
            cutoff=False,
        ) * (self.radial_embed_size**0.5)

        for i, gcn in enumerate(self.gcns):
            feat = gcn(edge_index, feat, edge_feat, edge_embed, node_attrs, dim_size=atom_types.shape[0])
            if i != self.num_gcn_layer - 1:
                feat = self.act(feat)

        n_graph, n_sample = grid.shape[0], grid.shape[1]
        if self.residual:
            grid_flat = grid.reshape([-1, 3])
            grid_batch = paddle.arange(n_graph, dtype="int64").repeat_interleave(repeats=n_sample)
            grid_dst, node_src = radius(
                atom_coord,
                grid_flat,
                self.grid_cutoff,
                batch,
                grid_batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            grid_edge = grid_flat[grid_dst] - atom_coord[node_src]
            if grid_edge.shape[0] != 0:
                grid_len = paddle.norm(grid_edge, axis=-1) + 1e-8
                grid_edge_feat = o3.spherical_harmonics(
                    list(range(self.num_spherical + 1)),
                    grid_edge / (grid_len[..., None] + 1e-8),
                    normalize=False,
                    normalization="integral",
                )
                grid_edge_embed = soft_one_hot_linspace(
                    grid_len,
                    start=0.0,
                    end=self.grid_cutoff,
                    number=self.radial_embed_size,
                    basis="gaussian",
                    cutoff=False,
                ) * (self.radial_embed_size**0.5)

                residue = self.residue(
                    edge_index=(node_src, grid_dst),
                    node_feat=feat,
                    edge_feat=grid_edge_feat,
                    edge_embed=grid_edge_embed,
                    node_attrs=node_attrs,
                    dim_size=grid_flat.shape[0],
                )
            else:
                residue = paddle.zeros([grid_flat.shape[0], 1], dtype=feat.dtype)
        else:
            residue = 0.0

        sample_vec = grid[batch] - atom_coord.unsqueeze(axis=-2)
        if self.pbc and cell is not None:
            sample_vec = self._pbc_vec(sample_vec, cell[batch])
        orbital = self.orbital(sample_vec)
        density = (orbital * feat.unsqueeze(axis=1)).sum(axis=-1)
        density = scatter(density, batch, dim=0, reduce="sum")

        if self.residual:
            density = density + residue.reshape(density.shape)

        return density

    @staticmethod
    def _pbc_vec(vec, cell):
        coord = vec @ paddle.linalg.inv(cell)
        coord = coord - paddle.round(coord)
        pbc_vec = coord @ cell
        return pbc_vec.detach()
