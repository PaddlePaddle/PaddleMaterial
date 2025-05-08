# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import paddle
import paddle.nn as nn
from ppmat.schedulers import build_scheduler
from tqdm import tqdm

from ppmat.models.common import initializer
from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings
from ppmat.models.common.time_embedding import uniform_sample_t
from ppmat.utils import paddle_aux  # noqa
from ppmat.utils.crystal import lattice_params_to_matrix_paddle


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += paddle.exp(x=-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        exp1 = paddle.exp(x=-((x + T * i) ** 2) / 2 / sigma**2)
        p_ += (x + T * i) / sigma**2 * exp1
    return p_ / p_wrapped_normal(x, sigma, N, T)


class SinusoidsEmbedding(paddle.nn.Layer):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * paddle.arange(end=self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(axis=-1) * self.frequencies[None, None, :]
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


class CSPLayer(paddle.nn.Layer):
    """Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        prop_dim=512,
        act_fn=paddle.nn.Silu(),
        dis_emb=None,
        ln=False,
        ip=True,
    ):
        super(CSPLayer, self).__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(
                in_features=hidden_dim * 2 + 9 + self.dis_dim, out_features=hidden_dim
            ),
            act_fn,
            paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )
        self.node_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            act_fn,
            paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )

        self.prop_mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=prop_dim, out_features=hidden_dim),
            act_fn,
            paddle.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            act_fn,
        )

        self.ln = ln
        if self.ln:
            self.layer_norm = paddle.nn.LayerNorm(normalized_shape=hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            x = lattices
            perm_0 = list(range(x.ndim))
            perm_0[-1] = -2
            perm_0[-2] = -1
            lattice_ips = lattices @ x.transpose(perm=perm_0)
        else:
            lattice_ips = lattices

        lattice_ips_flatten = lattice_ips.reshape([-1, 9])
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = paddle.concat(
            x=[hi, hj, lattice_ips_flatten_edges, frac_diff], axis=1
        )
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = paddle.geometric.segment_mean(edge_features, edge_index[0])
        agg = paddle.concat(x=[node_features, agg], axis=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
        num_atoms=None,
        property_emb=None,
        property_mask=None,
    ):
        if property_emb is not None:
            property_features = self.prop_mlp(property_emb)
            if property_mask is not None:
                property_features = property_features * property_mask
            property_features = paddle.repeat_interleave(
                property_features, num_atoms, axis=0
            )
            node_features = node_features + property_features

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(paddle.nn.Layer):
    """CSPNet model, based on https://arxiv.org/abs/2309.04475

    Args:
        hidden_dim (int, optional): Hidden dimension. Defaults to 128.
        latent_dim (int, optional): Latent space dimension for time embedding.
            Defaults to 256.
        num_layers (int, optional): Number of CSPLayer. Defaults to 4.
        act_fn (str, optional): Activation function type. Defaults to "silu".
        dis_emb (str, optional): Distance embedding method, can be 'sin' or 'none'.
            Defaults to "sin".
        num_freqs (int, optional): Number of frequency components (effective only when
            dis_emb='sin'). Defaults to 10.
        edge_style (str, optional): Edge feature encoding method. Must be set to 'fc'
            (fully connected atomic interactions). Default: "fc".
        ln (bool, optional): Enable LayerNorm after CSPLayer. Defaults to False.
        ip (bool, optional): Apply lattice inner product for O(3)-invariance. Defaults
            to True.
        smooth (bool, optional): Atomic number encoding method. True: Linear layer,
            False: Embedding layer. Defaults to False.
        pred_type (bool, optional): Enable atom type prediction. Defaults to False.
        prop_dim (int, optional): Property feature dimension for scalar property
            guidance. Defaults to 512.
        pred_scalar (bool, optional): Enable scalar property prediction. Defaults to
            False.
        num_classes (Optional[int], optional): Number of atom type classes. Defaults
            to None.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        num_layers: int = 4,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 10,
        edge_style: str = "fc",
        ln: bool = False,
        ip: bool = True,
        smooth: bool = False,
        pred_type: bool = False,
        prop_dim: int = 512,
        pred_scalar: bool = False,
        num_classes: int = 100,
    ):
        super(CSPNet, self).__init__()
        self.ip = ip
        self.smooth = smooth
        self.num_classes = num_classes

        if self.smooth:
            self.node_embedding = paddle.nn.Linear(
                in_features=self.num_classes, out_features=hidden_dim
            )
        else:
            self.node_embedding = paddle.nn.Embedding(
                num_embeddings=self.num_classes, embedding_dim=hidden_dim
            )
        self.atom_latent_emb = paddle.nn.Linear(
            in_features=hidden_dim + latent_dim, out_features=hidden_dim
        )
        if act_fn == "silu":
            self.act_fn = paddle.nn.Silu()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        self.prop_dim = prop_dim
        for i in range(0, num_layers):
            self.add_sublayer(
                name="csp_layer_%d" % i,
                sublayer=CSPLayer(
                    hidden_dim,
                    prop_dim=self.prop_dim,
                    act_fn=self.act_fn,
                    dis_emb=self.dis_emb,
                    ln=ln,
                    ip=ip,
                ),
            )
        self.num_layers = num_layers
        self.coord_out = paddle.nn.Linear(
            in_features=hidden_dim, out_features=3, bias_attr=False
        )
        self.lattice_out = paddle.nn.Linear(
            in_features=hidden_dim, out_features=9, bias_attr=False
        )
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = paddle.nn.LayerNorm(normalized_shape=hidden_dim)
        if self.pred_type:
            self.type_out = paddle.nn.Linear(
                in_features=hidden_dim, out_features=self.num_classes
            )
        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = paddle.nn.Linear(in_features=hidden_dim, out_features=1)

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = paddle.concat(x=[tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def gen_edges(self, num_atoms, frac_coords):
        if self.edge_style == "fc":
            cum_num_atoms = paddle.cumsum(x=num_atoms)
            indices_pp = []
            rows = paddle.arange(num_atoms.max())
            ind1, ind2 = paddle.meshgrid(rows, rows)
            index = paddle.stack(x=[ind1, ind2], axis=0)
            for n, cum_n in zip(num_atoms, cum_num_atoms):
                offset = cum_n - n
                indices_pp.append(index[:, :n, :n].reshape((2, -1)) + offset)
            indices_pp = paddle.concat(x=indices_pp, axis=1)
            fc_edges = indices_pp
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        else:
            raise NotImplementedError("Edge style '%s'" % self.edge_style)

    def forward(
        self,
        t,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        property_emb=None,
        property_mask=None,
    ):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edges[0]]
        node_features = self.node_embedding(atom_types)

        t_per_atom = t.repeat_interleave(repeats=num_atoms, axis=0)
        node_features = paddle.concat(x=[node_features, t_per_atom], axis=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            node_features = eval("self.csp_layer_%d" % i)(
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
                num_atoms=num_atoms,
                property_emb=property_emb,
                property_mask=property_mask,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)
        graph_features = paddle.geometric.segment_mean(node_features, node2graph)
        if self.pred_scalar:
            return self.scalar_out(graph_features)
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.einsum("bij,bjk->bik", lattice_out, lattices)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out
        return lattice_out, coord_out


class DiffCSP(paddle.nn.Layer):
    """Crystal Structure Prediction by Joint Equivariant Diffusion

    https://arxiv.org/abs/2309.04475

    Args:
        decoder_cfg (dict): Decoder layer configuration. See `CSPNet` for more details.
        lattice_noise_scheduler_cfg (dict): Noise scheduler configuration for lattice.
        coord_noise_scheduler_cfg (dict): Noise scheduler configuration for coordinate.
        num_train_timesteps (int): Number of diffusion steps. Defaults to 1000.
        time_dim (int): Time embedding dimension. Defaults to 256.
        lattice_loss_weight (float, optional): Lattice loss weight. Defaults to 1.0.
        coord_loss_weight (float, optional): Coordinate loss weight. Defaults to 1.0.
    """

    def __init__(
        self,
        decoder_cfg: dict,
        lattice_noise_scheduler_cfg: dict,
        coord_noise_scheduler_cfg: dict,
        num_train_timesteps: int = 1000,
        time_dim: int = 256,
        lattice_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
    ) -> None:

        super().__init__()

        self.decoder = CSPNet(**decoder_cfg)

        self.lattice_scheduler = build_scheduler(lattice_noise_scheduler_cfg)
        self.coord_scheduler = build_scheduler(coord_noise_scheduler_cfg)

        self.num_train_timesteps = num_train_timesteps
        self.time_dim = time_dim
        self.lattice_loss_weight = lattice_loss_weight
        self.coord_loss_weight = coord_loss_weight

        self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            initializer.linear_init_(m)
        elif isinstance(m, nn.Embedding):
            initializer.normal_(m.weight)
        elif isinstance(m, nn.LSTM):
            initializer.lstm_init_(m)

    def forward(self, batch, **kwargs):

        structure_array = batch["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )

        times = uniform_sample_t(batch_size, self.num_train_timesteps)
        times_per_atom = times.repeat_interleave(repeats=structure_array["num_atoms"])

        time_emb = self.time_embedding(times)

        if hasattr(structure_array, "lattice"):
            lattices = structure_array["lattice"]
        else:
            lattices = lattice_params_to_matrix_paddle(
                structure_array["lengths"], structure_array["angles"]
            )

        frac_coords = structure_array["frac_coords"]
        rand_l, rand_x = paddle.randn(
            shape=lattices.shape, dtype=lattices.dtype
        ), paddle.randn(shape=frac_coords.shape, dtype=frac_coords.dtype)

        input_lattice = self.lattice_scheduler.add_noise(
            lattices, rand_l, timesteps=times
        )

        input_frac_coords = self.coord_scheduler.add_noise(
            frac_coords, rand_x, timesteps=times_per_atom
        )
        input_frac_coords = input_frac_coords % 1.0

        pred_l, pred_x = self.decoder(
            time_emb,
            structure_array["atom_types"] - 1,
            input_frac_coords,
            input_lattice,
            structure_array["num_atoms"],
            batch_idx,
        )

        sigmas_per_atom = self.coord_scheduler.discrete_sigmas[times_per_atom][:, None]
        sigmas_norm_per_atom = self.coord_scheduler.discrete_sigmas_norm[
            times_per_atom
        ][:, None]

        tar_x = d_log_p_wrapped_normal(
            sigmas_per_atom * rand_x, sigmas_per_atom
        ) / paddle.sqrt(x=sigmas_norm_per_atom)

        loss_lattice = paddle.nn.functional.mse_loss(input=pred_l, label=rand_l)
        loss_coord = paddle.nn.functional.mse_loss(input=pred_x, label=tar_x)
        loss = (
            self.lattice_loss_weight * loss_lattice
            + self.coord_loss_weight * loss_coord
        )
        loss_dict = {
            "loss": loss,
            "loss_lattice": loss_lattice,
            "loss_coord": loss_coord,
        }

        return {
            "loss_dict": loss_dict,
        }

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=1000, **kwargs):
        structure_array = batch_data["structure_array"]
        batch_size = structure_array["num_atoms"].shape[0]
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), repeats=structure_array["num_atoms"]
        )
        l_T, x_T = paddle.randn(shape=[batch_size, 3, 3]), paddle.rand(
            shape=[structure_array["num_atoms"].sum(), 3]
        )
        l_t, x_t = l_T, x_T

        self.lattice_scheduler.set_timesteps(num_inference_steps)
        self.coord_scheduler.set_timesteps(num_inference_steps)

        for lattice_t, coord_t in tqdm(
            zip(self.lattice_scheduler.timesteps, self.coord_scheduler.timesteps),
            total=num_inference_steps,
            desc="Sampling...",
        ):
            time_emb = self.time_embedding(
                paddle.ones([batch_size], dtype="int64") * lattice_t
            )
            pred_l, pred_x = self.decoder(
                time_emb,
                structure_array["atom_types"] - 1,
                x_t,
                l_t,
                structure_array["num_atoms"],
                batch_idx,
            )
            x_t = self.coord_scheduler.step_correct(pred_x, coord_t, x_t).prev_sample

            pred_l, pred_x = self.decoder(
                time_emb,
                structure_array["atom_types"] - 1,
                x_t,
                l_t,
                structure_array["num_atoms"],
                batch_idx,
            )
            output = self.coord_scheduler.step_pred(
                pred_x,
                coord_t,
                x_t,
            )
            x_t, x_t_mean = output.prev_sample, output.prev_sample_mean

            l_t = self.lattice_scheduler.step(
                pred_l,
                lattice_t,
                l_t,
            ).prev_sample

            x_t = x_t % 1.0

        x_t = x_t_mean % 1.0

        start_idx = 0
        result = []
        for i in range(batch_size):
            end_idx = start_idx + structure_array["num_atoms"][i]
            result.append(
                {
                    "num_atoms": structure_array["num_atoms"][i].tolist(),
                    "atom_types": structure_array["atom_types"][
                        start_idx:end_idx
                    ].tolist(),
                    "frac_coords": x_t[start_idx:end_idx].tolist(),
                    "lattice": l_t[i].tolist(),
                }
            )
            start_idx += structure_array["num_atoms"][i]

        return {"result": result}
