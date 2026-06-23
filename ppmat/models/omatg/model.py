# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OMATG-specific CSPNet with knn graph, internal time embedding, dual outputs.

Extends diffcsp.CSPNet to add:
  - knn graph construction via radius_graph_pbc
  - internal time embedding (SinusoidalTimeEmbeddings)
  - dual output heads (coord_out_2, lattice_out_2, type_out_2)
  - species_shift and enable_masked_species() for DNG mode
"""

import paddle.nn as nn

from ppmat.losses import MSELoss
from ppmat.models.diffcsp.diffcsp import CSPNet
from ppmat.utils.crystal import radius_graph_pbc, frac_to_cart_coords_with_lattice
from ppmat.utils.misc import repeat_blocks


class OMATGCSPNet(CSPNet):
    """OMATG-specific CSPNet extending diffcsp backbone."""

    def __init__(
        self,
        hidden_dim=128,
        num_layers=4,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
        time_embed_dim=None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            latent_dim=time_embed_dim if time_embed_dim is not None else 1,
            num_layers=num_layers,
            act_fn=act_fn,
            dis_emb=dis_emb,
            num_freqs=num_freqs,
            edge_style=edge_style,
            ln=ln,
            ip=ip,
            smooth=smooth,
            pred_type=pred_type,
            prop_dim=hidden_dim,
            pred_scalar=pred_scalar,
            num_classes=max_atoms,
        )
        # OMATG does not use property embedding; clear prop_mlp created by CSPNet
        for i in range(num_layers):
            getattr(self, "csp_layer_%d" % i).prop_mlp = None

        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.species_shift = 1

        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            from ppmat.models.common.time_embedding import SinusoidalTimeEmbeddings

            self.time_embedder = SinusoidalTimeEmbeddings(time_embed_dim)
            self.atom_latent_emb = nn.Linear(
                hidden_dim + time_embed_dim, hidden_dim
            )
        else:
            self.time_embedder = None
            self.atom_latent_emb = nn.Linear(hidden_dim + 1, hidden_dim)

        self.dis_dim = (num_freqs * 2 * 3) if dis_emb == "sin" else 0

        self.coord_out_2 = nn.Linear(hidden_dim, 3, bias_attr=False)
        self.lattice_out_2 = nn.Linear(hidden_dim, 9, bias_attr=False)
        if self.pred_type:
            self.type_out_2 = nn.Linear(hidden_dim, max_atoms)

    def enable_masked_species(self):
        self.node_embedding = nn.Embedding(self.num_classes + 1, self.hidden_dim)
        self.species_shift = 0

    def reorder_symmetric_edges(self, edge_index, cell_offsets, neighbors, edge_vector):
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms = mask_same_atoms & cell_earlier
        mask = mask_sep_atoms | mask_same_atoms
        edge_index_new = edge_index[mask[None, :].expand([2, -1])].reshape([2, -1])
        edge_index_cat = paddle.concat(
            [
                edge_index_new,
                paddle.stack([edge_index_new[1], edge_index_new[0]], axis=0),
            ],
            axis=1,
        )
        batch_edge = paddle.repeat_interleave(
            paddle.arange(neighbors.shape[0]), neighbors
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * paddle.bincount(batch_edge, minlength=neighbors.shape[0])
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.shape[1],
        )
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = CSPNet.select_symmetric_edges(
            self, cell_offsets, mask, edge_reorder_idx, True
        )
        edge_vector_new = CSPNet.select_symmetric_edges(
            self, edge_vector, mask, edge_reorder_idx, True
        )
        return edge_index_new, cell_offsets_new, neighbors_new, edge_vector_new

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
            return CSPNet.gen_edges(self, num_atoms, frac_coords)

        cart_coords = frac_to_cart_coords_with_lattice(frac_coords, num_atoms, lattices)
        edge_index, to_jimages, num_bonds = radius_graph_pbc(
            cart_coords,
            lattices,
            num_atoms,
            self.cutoff,
            self.max_neighbors,
            num_atoms.place,
        )
        j_index, i_index = edge_index[0], edge_index[1]
        distance_vectors = frac_coords[j_index] - frac_coords[i_index]
        distance_vectors = distance_vectors + to_jimages
        edge_index_new, _, _, edge_vector_new = self.reorder_symmetric_edges(
            edge_index, to_jimages, num_bonds, distance_vectors
        )
        return edge_index_new, -edge_vector_new

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]

        if self.smooth:
            node_features = self.node_embedding(atom_types.cast("float32"))
        else:
            node_features = self.node_embedding(atom_types - self.species_shift)

        if t.ndim == 0:
            t = t.unsqueeze(0)

        if self.time_embed_dim is not None:
            t_embed = self.time_embedder(t)
        else:
            t_embed = t

        if t_embed.ndim == 1:
            t_embed = t_embed.unsqueeze(0)
        t_per_atom = paddle.repeat_interleave(t_embed, num_atoms, axis=0)
        node_features = paddle.concat([node_features, t_per_atom], axis=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(self.num_layers):
            layer = getattr(self, "csp_layer_%d" % i)
            node_features = layer(
                node_features,
                frac_coords,
                lattices,
                edges,
                edge2graph,
                frac_diff=frac_diff,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)
        coord_out_2 = self.coord_out_2(node_features)

        graph_features = paddle.geometric.segment_mean(node_features, node2graph)

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.reshape([-1, 3, 3])
        if self.ip:
            lattice_out = paddle.matmul(lattice_out, lattices)

        lattice_out_2 = self.lattice_out_2(graph_features)
        lattice_out_2 = lattice_out_2.reshape([-1, 3, 3])
        if self.ip:
            lattice_out_2 = paddle.matmul(lattice_out_2, lattices)

        if self.pred_type:
            type_out = self.type_out(node_features)
            type_out_2 = self.type_out_2(node_features)
            return (
                lattice_out,
                coord_out,
                type_out,
                lattice_out_2,
                coord_out_2,
                type_out_2,
            )

        return lattice_out, coord_out, lattice_out_2, coord_out_2

    def forward_dict(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        preds = OMATGCSPNet.forward(
            self, t, atom_types, frac_coords, lattices, num_atoms, node2graph
        )
        if self.pred_scalar:
            return {"scalar": preds}
        if self.pred_type:
            return {
                "pos_b": preds[1],
                "pos_eta": preds[4],
                "cell_b": preds[0],
                "cell_eta": preds[3],
                "species_b": preds[2],
                "species_eta": preds[5],
            }
        return {
            "pos_b": preds[1],
            "pos_eta": preds[3],
            "cell_b": preds[0],
            "cell_eta": preds[2],
        }

class OMATGCSPNetFull(OMATGCSPNet):
    """Full CSPNet with time embedding integrated for OMatG models.

    This is a convenience wrapper that creates CSPNet with time embedding.
    Renamed from CSPNetFull to avoid naming conflicts with other CSPNet implementations.

    Args:
        hidden_dim: Hidden dimension for embeddings
        num_layers: Number of message passing layers
        max_atoms: Maximum number of atoms in crystal
        act_fn: Activation function
        dis_emb: Distance embedding type
        num_freqs: Number of frequency bands for sinusoidal embedding
        edge_style: Edge construction style
        cutoff: Distance cutoff for graph construction
        max_neighbors: Maximum number of neighbors
        ln: Whether to use layer normalization
        ip: Whether to use inner product
        smooth: Whether to use smooth embedding
        pred_type: Whether to predict atom types
        pred_scalar: Whether to predict scalar properties
        time_embed_dim: Time embedding dimension
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        max_atoms: int = 100,
        act_fn: str = "silu",
        dis_emb: str = "sin",
        num_freqs: int = 10,
        edge_style: str = "fc",
        cutoff: float = 6.0,
        max_neighbors: int = 20,
        ln: bool = False,
        ip: bool = True,
        smooth: bool = False,
        pred_type: bool = False,
        pred_scalar: bool = False,
        time_embed_dim: int = 256,
        use_si: bool = False,
        si_cfg: dict = None,
        sampler_cfg: dict = None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_atoms=max_atoms,
            act_fn=act_fn,
            dis_emb=dis_emb,
            num_freqs=num_freqs,
            edge_style=edge_style,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            ln=ln,
            ip=ip,
            smooth=smooth,
            pred_type=pred_type,
            pred_scalar=pred_scalar,
            time_embed_dim=time_embed_dim,
        )
        self.use_si = use_si
        self._si = None
        self._sampler = None
        self._relative_si_costs = None
        self.mse_loss = MSELoss()
        if use_si:
            self._build_si(si_cfg or {}, sampler_cfg or {})

    def forward(self, data, t=None):
        """Forward pass accepting dict or OMATGData/Batch object.

        Args:
            data: dict (for standalone use) or OMATGData/Batch (from collator pipeline).
            t: Time tensor (optional, will be sampled if None)

        Returns:
            Dictionary containing loss_dict with loss components
        """
        if isinstance(data, dict):
            atom_types = data["atom_types"]
            frac_coords = data["frac_coords"]
            lattices = data["lattices"]
            num_atoms = data["num_atoms"]
            node2graph = data["node2graph"]
        else:
            atom_types = data.species
            frac_coords = data.pos
            lattices = data.cell
            num_atoms = data.n_atoms
            node2graph = data.batch

        # SI-based training path: velocity matching loss via StochasticInterpolants
        if self.use_si:
            return self._si_forward(data)

        # Sample time if not provided
        if t is None:
            batch_size = lattices.shape[0]
            t = paddle.rand([batch_size])

        # Call forward_dict for b/eta outputs
        predictions = self.forward_dict(
            t=t,
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices=lattices,
            num_atoms=num_atoms,
            node2graph=node2graph,
        )

        lattices_gt = lattices
        frac_coords_gt = frac_coords

        if self.pred_type:
            loss_lattice = self.mse_loss(predictions["cell_b"], lattices_gt)
            loss_coord = self.mse_loss(predictions["pos_b"], frac_coords_gt)
            loss_type = paddle.nn.functional.cross_entropy(
                input=predictions["species_b"], label=atom_types - self.species_shift
            )
            loss = loss_lattice + loss_coord + loss_type
            return {
                "loss_dict": {
                    "loss": loss,
                    "loss_lattice": loss_lattice,
                    "loss_coord": loss_coord,
                    "loss_type": loss_type,
                }
            }

        # CSP mode: use b components from forward_dict
        loss_lattice = self.mse_loss(predictions["cell_b"], lattices_gt)
        loss_coord = self.mse_loss(predictions["pos_b"], frac_coords_gt)
        loss = loss_lattice + loss_coord
        return {
            "loss_dict": {
                "loss": loss,
                "loss_lattice": loss_lattice,
                "loss_coord": loss_coord,
            }
        }

    def sample(self, batch_data, num_inference_steps=100, **kwargs):
        """Sample crystal structures.

        Based on the original OMatG implementation in omg_trainer.py and generate_csp.py.
        Uses reverse-time ODE integration with Euler method.

        Args:
            batch_data: Dictionary (for standalone sampling) or OMATGData/Batch object.
                Dictionary supports:
                - structure_array with num_atoms and optionally atom_types
                - or direct fields: atom_types, frac_coords (ignored), lattices (ignored),
                  num_atoms, node2graph (ignored)
            num_inference_steps: Number of integration steps (default 100)
            **kwargs: Additional sampling parameters
                - step_lr: step learning rate for Euler integration (default 1e-5)

        Returns:
            Dictionary containing generated structures
        """
        if self.use_si and self._si is not None:
            return self._si_sample(batch_data, num_inference_steps, **kwargs)

        # Support dict (standalone sampling) and OMATGData/Batch (collator output)
        if isinstance(batch_data, dict):
            if "structure_array" in batch_data:
                sa = batch_data["structure_array"]
                num_atoms_list = sa["num_atoms"].tolist()
                atom_types = sa.get("atom_types")
            else:
                num_atoms_list = batch_data["num_atoms"].tolist()
                atom_types = batch_data.get("atom_types")
        else:
            num_atoms_list = batch_data.n_atoms.tolist()
            atom_types = batch_data.species

        batch_size = len(num_atoms_list)
        total_atoms = sum(num_atoms_list)

        if atom_types is None:
            atom_types = paddle.randint(1, 100, shape=[total_atoms])

        num_atoms_tensor = paddle.to_tensor(num_atoms_list, dtype="int64")
        node2graph = paddle.repeat_interleave(
            paddle.arange(batch_size, dtype="int64"), num_atoms_tensor
        )

        # Sample from base distribution (uniform for positions, randn for lattice)
        # Reference: omg_lightning.py:predict_step() -> x_0 = self.sampler.sample_p_0(x)
        frac_coords = paddle.rand([total_atoms, 3])
        lattices = paddle.randn([batch_size, 3, 3])

        # Scale initial lattice to reasonable values
        lattices = lattices * 2.0  # Scale to avoid extreme values

        # Integration loop using Euler method
        step_lr = 1e-5

        for step in range(num_inference_steps):
            t = paddle.full([batch_size], step / num_inference_steps)
            lattice_pred, coord_pred = self._predict(
                t, atom_types, frac_coords, lattices,
                num_atoms_tensor, node2graph,
            )

            if paddle.any(paddle.isnan(lattice_pred)) or paddle.any(
                paddle.isnan(coord_pred)
            ):
                continue

            frac_coords = frac_coords + coord_pred * step_lr
            lattices = lattices + lattice_pred * step_lr

            frac_coords = frac_coords % 1.0

        # Final clipping to ensure valid lattice
        lattices = paddle.clip(lattices, -10.0, 10.0)

        # Convert lattice matrices to lengths and angles for BuildStructure
        return self._build_sample_result(num_atoms_list, atom_types, frac_coords, lattices)

    def _predict(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        predictions = super().forward(
            t=t,
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices=lattices,
            num_atoms=num_atoms,
            node2graph=node2graph,
        )
        lattice_pred, coord_pred = predictions[0], predictions[1]
        if paddle.any(paddle.isnan(lattice_pred)):
            lattice_pred = paddle.zeros_like(lattice_pred)
        if paddle.any(paddle.isnan(coord_pred)):
            coord_pred = paddle.zeros_like(coord_pred)
        return lattice_pred, coord_pred

    def _make_model_function(self):
        def model_function(x_t, time):
            return self.forward_dict(
                t=time,
                atom_types=x_t.species,
                frac_coords=x_t.pos,
                lattices=x_t.cell,
                num_atoms=x_t.n_atoms,
                node2graph=x_t.batch,
            )
        return model_function

    @staticmethod
    def _build_sample_result(num_atoms_list, atom_types, frac_coords, lattices):
        from ppmat.utils.crystal import lattices_to_params_shape_paddle

        lengths, angles = lattices_to_params_shape_paddle(lattices)
        start_idx = 0
        result = []
        for i, n in enumerate(num_atoms_list):
            end_idx = start_idx + n
            result.append({
                "num_atoms": n,
                "atom_types": atom_types[start_idx:end_idx].tolist(),
                "frac_coords": frac_coords[start_idx:end_idx].tolist(),
                "lengths": lengths[i].tolist(),
                "angles": angles[i].tolist(),
            })
            start_idx += n
        return {"result": result}

    def _si_sample(self, batch_data, num_inference_steps, **kwargs):
        x_1 = self._data_to_omatg(batch_data)
        x_0 = self._sampler.sample_p_0(x_1)

        gen = self._si.integrate(x_0, self._make_model_function(), save_intermediate=False)

        return self._build_sample_result(
            gen.n_atoms.tolist(), gen.species, gen.pos, gen.cell
        )

    def _build_si(self, si_cfg: dict, sampler_cfg: dict) -> None:
        from ppmat.models.omatg.si.core import build_si_from_cfg, build_sampler_from_cfg
        from ppmat.models.omatg.si import StochasticInterpolants

        si_list = si_cfg.get("stochastic_interpolants", [])
        use_factory = si_list and isinstance(si_list[0], dict) and "__class_name__" in si_list[0]
        self._si = build_si_from_cfg(si_cfg) if use_factory else StochasticInterpolants(
            stochastic_interpolants=si_list,
            data_fields=si_cfg["data_fields"],
            integration_time_steps=si_cfg.get("integration_time_steps", 210),
        )
        self._relative_si_costs = si_cfg.get("relative_si_costs", {})

        if sampler_cfg:
            if any(isinstance(v, dict) and "__class_name__" in v for v in sampler_cfg.values()):
                self._sampler = build_sampler_from_cfg(sampler_cfg)
            else:
                self._sampler = IndependentSampler(
                    dataset_name=sampler_cfg.get("dataset_name"),
                    mirror_species=sampler_cfg.get("mirror_species", True),
                    mask_species=sampler_cfg.get("mask_species", False),
                )

    def _data_to_omatg(self, data):
        from ppmat.datasets.omatg_dataset import OMATGData

        if isinstance(data, OMATGData):
            return data
        return OMATGData.from_collate_dict(data)

    def _si_forward(self, data: dict) -> dict:
        """SI training step: sample x_0, interpolate, compute velocity matching loss."""
        from ppmat.models.omatg.si import SMALL_TIME, BIG_TIME

        x_1 = self._data_to_omatg(data)
        x_0 = self._sampler.sample_p_0(x_1)
        batch_size = len(x_1.n_atoms)
        t = paddle.rand([batch_size]) * (BIG_TIME - SMALL_TIME) + SMALL_TIME

        losses = self._si.losses(self._make_model_function(), t, x_0, x_1)
        total_loss = paddle.to_tensor(0.0)
        loss_dict = {}
        for key, val in losses.items():
            cost = self._relative_si_costs.get(key, 1.0)
            weighted = cost * val
            loss_dict[key] = val
            total_loss = total_loss + weighted
        loss_dict["loss"] = total_loss
        return {"loss_dict": loss_dict}


__all__ = [
    "OMATGCSPNetFull",
]
"""Independent base distribution sampler using Paddle native APIs."""

import paddle
from ase.geometry.cell import cellpar_to_cell

from ppmat.datasets.omatg_dataset import Structure, OMATGData

_LATTICE_PARAMS = {
    "carbon_24": {
        "means": [0.9852757453918457, 1.3865314722061157, 1.7068126201629639],
        "stds": [0.14957907795906067, 0.20431114733219147, 0.2403733879327774],
    },
    "mp_20": {
        "means": [1.575442910194397, 1.7017393112182617, 1.9781638383865356],
        "stds": [0.24437622725963593, 0.26526379585266113, 0.3535512685775757],
    },
    "mpts_52": {
        "means": [1.6565313339233398, 1.8407557010650635, 2.1225264072418213],
        "stds": [0.2952289581298828, 0.3340013027191162, 0.41885802149772644],
    },
    "perov_5": {
        "means": [1.419227957725525, 1.419227957725525, 1.419227957725525],
        "stds": [0.07268335670232773, 0.07268335670232773, 0.07268335670232773],
    },
    "alex_mp_20": {
        "means": [1.5808929163076058, 1.74672046352959, 2.065243388307474],
        "stds": [0.27284015410437057, 0.2944785731740152, 0.30899526911753017],
    },
}


def _sample_cell(dataset_name):
    params = _LATTICE_PARAMS.get(dataset_name)
    if params is None:
        return paddle.randn([3, 3]).numpy() * 2.0
    lengths = paddle.exp(
        paddle.randn([3]) * paddle.to_tensor(params["stds"])
        + paddle.to_tensor(params["means"])
    )
    angles = paddle.rand([3]) * 60.0 + 60.0
    return cellpar_to_cell(paddle.concat((lengths, angles)).numpy())


class IndependentSampler:
    """Sample base distributions for SI training using Paddle native APIs.

    Args:
        dataset_name: Optional dataset name for informed lattice distribution.
        mirror_species: If True, keep input species unchanged (mirror).
        mask_species: If True, replace species with zeros (mask token).
    """

    def __init__(self, dataset_name=None, mirror_species=True, mask_species=False):
        self._dataset_name = dataset_name
        self._mirror_species = mirror_species
        self._mask_species = mask_species

    def sample_p_0(self, x_1: OMATGData) -> OMATGData:
        batch_size = len(x_1.n_atoms)
        structures = []
        for i in range(batch_size):
            sl = x_1.slice(i)
            pos = paddle.rand(x_1.pos[sl].shape, dtype=x_1.pos.dtype)
            cell = paddle.to_tensor(_sample_cell(self._dataset_name), dtype=x_1.cell.dtype)
            if self._mask_species:
                species = paddle.zeros_like(x_1.species[sl])
            elif self._mirror_species:
                species = x_1.species[sl].clone()
            else:
                species = x_1.species[sl].clone()
            sampled = Structure(
                cell=cell,
                atomic_numbers=species,
                pos=pos,
                pos_is_fractional=True,
            )
            structures.append(sampled)
        return OMATGData.from_batch(structures, concatenate=True)
