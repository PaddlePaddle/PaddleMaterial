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

"""OMatG model wrapper for PaddleMaterials training framework.
"""

import paddle

from ppmat.models.omatg.model import OMATGCSPNet as CSPNet


class OMATGCSPNetFull(CSPNet):
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
        if use_si:
            self._build_si(si_cfg or {}, sampler_cfg or {})

    def forward(self, data, t=None):
        """Forward pass accepting dictionary input.

        Args:
            data: Dictionary containing atom_types, frac_coords, lattices, num_atoms, node2graph
            t: Time tensor (optional, will be sampled if None)

        Returns:
            Dictionary containing loss_dict with loss components
        """
        # Extract parameters from data dictionary
        atom_types = data.get("atom_types")
        frac_coords = data.get("frac_coords")
        lattices = data.get("lattices")
        num_atoms = data.get("num_atoms")
        node2graph = data.get("node2graph")

        # SI-based training path: velocity matching loss via StochasticInterpolants
        if self.use_si:
            return self._si_forward(data)

        # Sample time if not provided
        if t is None:
            batch_size = lattices.shape[0]
            t = paddle.rand([batch_size])

        # Call parent forward method with extracted parameters
        predictions = super().forward(
            t=t,
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices=lattices,
            num_atoms=num_atoms,
            node2graph=node2graph,
        )

        # Ground truth targets
        lattices_gt = lattices
        frac_coords_gt = frac_coords

        if self.pred_type:
            # DNG mode: parent returns 6-tuple (b/eta for lattice, coord, type)
            lattice_pred, coord_pred, type_pred = predictions[0], predictions[1], predictions[2]
            loss_lattice = paddle.nn.functional.mse_loss(lattice_pred, lattices_gt)
            loss_coord = paddle.nn.functional.mse_loss(coord_pred, frac_coords_gt)
            # Cross-entropy on species, aligned with original DiscreteFlowMatchingMask
            loss_type = paddle.nn.functional.cross_entropy(
                input=type_pred, label=atom_types - self.species_shift
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

        # CSP mode: parent returns 4-tuple (cell_b, pos_b, cell_eta, pos_eta)
        lattice_pred, coord_pred = predictions[0], predictions[1]
        loss_lattice = paddle.nn.functional.mse_loss(lattice_pred, lattices_gt)
        loss_coord = paddle.nn.functional.mse_loss(coord_pred, frac_coords_gt)
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
            batch_data: Dictionary containing either:
                - structure_array with num_atoms and optionally atom_types
                - or direct fields: atom_types, frac_coords (ignored), lattices (ignored),
                  num_atoms, node2graph (ignored)
            num_inference_steps: Number of integration steps (default 100)
            **kwargs: Additional sampling parameters
                - step_lr: step learning rate for Euler integration (default 1e-5)

        Returns:
            Dictionary containing generated structures
        """
        from ppmat.utils.crystal import lattices_to_params_shape_paddle

        if self.use_si and self._si is not None:
            return self._si_sample(batch_data, num_inference_steps, **kwargs)

        # Support both structure_array format and direct collator format
        if "structure_array" in batch_data:
            structure_array = batch_data["structure_array"]
            num_atoms_list = structure_array["num_atoms"].tolist()
        else:
            num_atoms = batch_data["num_atoms"]
            num_atoms_list = num_atoms.tolist()
        batch_size = len(num_atoms_list)
        total_atoms = sum(num_atoms_list)

        # Get atom_types if available, otherwise use random
        if "structure_array" in batch_data and "atom_types" in batch_data["structure_array"]:
            atom_types = batch_data["structure_array"]["atom_types"]
        elif "atom_types" in batch_data:
            atom_types = batch_data["atom_types"]
        else:
            atom_types = paddle.randint(1, 100, shape=[total_atoms])

        # Create node2graph mapping
        node2graph = []
        for i, num_atoms in enumerate(num_atoms_list):
            node2graph.extend([i] * num_atoms)
        node2graph = paddle.to_tensor(node2graph)

        # Sample from base distribution (uniform for positions, randn for lattice)
        # Reference: omg_lightning.py:predict_step() -> x_0 = self.sampler.sample_p_0(x)
        frac_coords = paddle.rand([total_atoms, 3])
        lattices = paddle.randn([batch_size, 3, 3])

        # Scale initial lattice to reasonable values
        lattices = lattices * 2.0  # Scale to avoid extreme values

        # Integration loop using Euler method
        step_lr = 1e-5
        num_atoms_tensor = paddle.to_tensor(num_atoms_list, dtype="int64")

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
        lengths, angles = lattices_to_params_shape_paddle(lattices)

        # Prepare results
        start_idx = 0
        result = []
        for i in range(batch_size):
            end_idx = start_idx + num_atoms_list[i]
            result.append(
                {
                    "num_atoms": num_atoms_list[i],
                    "atom_types": atom_types[start_idx:end_idx].tolist(),
                    "frac_coords": frac_coords[start_idx:end_idx].tolist(),
                    "lengths": lengths[i].tolist(),
                    "angles": angles[i].tolist(),
                }
            )
            start_idx += num_atoms_list[i]

        return {"result": result}

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

    def _si_sample(self, batch_data, num_inference_steps, **kwargs):
        """SI-based sampling: integrate from base distribution x_0 to x_1."""
        from ppmat.utils.crystal import lattices_to_params_shape_paddle

        x_1 = self._data_to_omatg(batch_data)
        x_0 = self._sampler.sample_p_0(x_1)

        def model_function(x_t, time):
            return self.forward_dict(
                t=time,
                atom_types=x_t.species,
                frac_coords=x_t.pos,
                lattices=x_t.cell,
                num_atoms=x_t.n_atoms,
                node2graph=x_t.batch,
            )

        gen = self._si.integrate(x_0, model_function, save_intermediate=False)

        num_atoms_list = gen.n_atoms.tolist()
        batch_size = len(num_atoms_list)
        atom_types = gen.species
        frac_coords = gen.pos
        lattices = gen.cell

        lengths, angles = lattices_to_params_shape_paddle(lattices)

        start_idx = 0
        result = []
        for i in range(batch_size):
            end_idx = start_idx + num_atoms_list[i]
            result.append(
                {
                    "num_atoms": num_atoms_list[i],
                    "atom_types": atom_types[start_idx:end_idx].tolist(),
                    "frac_coords": frac_coords[start_idx:end_idx].tolist(),
                    "lengths": lengths[i].tolist(),
                    "angles": angles[i].tolist(),
                }
            )
            start_idx += num_atoms_list[i]

        return {"result": result}

    def _build_si(self, si_cfg: dict, sampler_cfg: dict) -> None:
        """Build StochasticInterpolants and IndependentSampler from config dicts.

        Supports two config styles:
        1. Pre-built objects in stochastic_interpolants list
        2. PaddleMaterials style (__class_name__/__init_params__) via factory
        """
        from ppmat.models.omatg.si.factory import build_si_from_cfg, build_sampler_from_cfg

        if "stochastic_interpolants" in si_cfg and isinstance(
            si_cfg["stochastic_interpolants"], list
        ) and len(si_cfg["stochastic_interpolants"]) > 0:
            first = si_cfg["stochastic_interpolants"][0]
            if isinstance(first, dict) and "__class_name__" in first:
                self._si = build_si_from_cfg(si_cfg)
            else:
                from ppmat.models.omatg.si import StochasticInterpolants

                self._si = StochasticInterpolants(
                    stochastic_interpolants=si_cfg["stochastic_interpolants"],
                    data_fields=si_cfg["data_fields"],
                    integration_time_steps=si_cfg.get("integration_time_steps", 210),
                )
        else:
            from ppmat.models.omatg.si import StochasticInterpolants

            self._si = StochasticInterpolants(
                stochastic_interpolants=si_cfg.get("stochastic_interpolants", []),
                data_fields=si_cfg["data_fields"],
                integration_time_steps=si_cfg.get("integration_time_steps", 210),
            )
        self._relative_si_costs = si_cfg.get("relative_si_costs", {})

        if sampler_cfg and any(
            isinstance(v, dict) and "__class_name__" in v
            for v in sampler_cfg.values()
        ):
            self._sampler = build_sampler_from_cfg(sampler_cfg)
        else:
            from ppmat.models.omatg.sampler import IndependentSampler

            self._sampler = IndependentSampler(
                position_distribution=sampler_cfg.get("pos_distribution"),
                cell_distribution=sampler_cfg.get("cell_distribution"),
                species_distribution=sampler_cfg.get("species_distribution"),
            )

    def _data_to_omatg(self, data: dict):
        """Convert collate dict (atom_types/frac_coords/lattices) to OMATGData."""
        from ppmat.models.omatg.datamodule.omatg_data import OMATGData

        num_atoms = data["num_atoms"]
        batch = data["node2graph"]
        ptr = paddle.concat(
            [
                paddle.to_tensor([0], dtype="int64"),
                paddle.cumsum(num_atoms, axis=0).cast("int64"),
            ]
        )
        d = OMATGData()
        d.n_atoms = num_atoms
        d.species = data["atom_types"]
        d.cell = data["lattices"]
        d.pos = data["frac_coords"]
        d.pos_is_fractional = paddle.ones_like(num_atoms, dtype="bool")
        d.batch = batch
        d.ptr = ptr
        d.property_dict = {}
        return d

    def _si_forward(self, data: dict) -> dict:
        """SI training step: sample x_0, interpolate, compute velocity matching loss."""
        from ppmat.models.omatg.si import SMALL_TIME, BIG_TIME

        x_1 = self._data_to_omatg(data)
        x_0 = self._sampler.sample_p_0(x_1)
        batch_size = len(x_1.n_atoms)
        t = paddle.rand([batch_size]) * (BIG_TIME - SMALL_TIME) + SMALL_TIME

        def model_function(x_t, time):
            return self.forward_dict(
                t=time,
                atom_types=x_t.species,
                frac_coords=x_t.pos,
                lattices=x_t.cell,
                num_atoms=x_t.n_atoms,
                node2graph=x_t.batch,
            )

        losses = self._si.losses(model_function, t, x_0, x_1)
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
