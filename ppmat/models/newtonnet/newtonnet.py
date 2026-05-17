# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NewtonNet: Newtonian message passing network for molecular force fields.

Adapted from the PyTorch implementation at:
    https://github.com/THGLab/NewtonNet

Reference:
    Haghighatlari et al., "NewtonNet: a Newtonian message passing network
    for deep learning of interatomic potentials and forces", 2022.
"""

from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn


# ---------------------------------------------------------------------------
# Scatter utility (self-contained to avoid ppmat top-level import issues)
# Adapted from ppmat.utils.scatter
# ---------------------------------------------------------------------------

def _broadcast(src: paddle.Tensor, other: paddle.Tensor, dim: int):
    if dim < 0:
        dim = other.ndim + dim
    if src.ndim == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.ndim, other.ndim):
        src = src.unsqueeze(-1)
    src = src.expand(other.shape)
    return src


def scatter(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    """Scatter-add (or mean) operation, compatible with torch_scatter."""
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = paddle.zeros(size, dtype=src.dtype)
    out = paddle.put_along_axis(
        arr=out, indices=index, values=src, axis=dim, reduce="add"
    )
    if reduce == "mean":
        ones = paddle.ones(index.shape, dtype=src.dtype)
        count = paddle.zeros(out.shape, dtype=src.dtype)
        count = paddle.put_along_axis(
            arr=count, indices=index, values=ones, axis=dim, reduce="add"
        )
        count = paddle.clip(count, min=1)
        out = out / count
    return out


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def get_activation_by_string(key: str) -> nn.Layer:
    """Return an activation layer from a string key."""
    activations = {
        "swish": nn.Silu,
        "silu": nn.Silu,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "gelu": nn.GELU,
    }
    key_lower = key.lower()
    if key_lower not in activations:
        raise NotImplementedError(f"Unknown activation: {key}")
    return activations[key_lower]()


# ---------------------------------------------------------------------------
# Representation layers
# ---------------------------------------------------------------------------

class RadialBesselLayer(nn.Layer):
    """Radial Bessel basis functions: ``sin(freq * dist) / dist``.

    Args:
        n_basis (int): Number of radial basis functions. Default: 20.
    """

    def __init__(self, n_basis: int = 20):
        super().__init__()
        freqs = paddle.arange(1, n_basis + 1, dtype="float32") * float(np.pi)
        self.register_buffer("frequencies", freqs)

    def forward(self, dist: paddle.Tensor) -> paddle.Tensor:
        # dist: [E, 1]  →  output: [E, n_basis]
        return paddle.sin(self.frequencies * dist) / dist


class PolynomialCutoff(nn.Layer):
    """Smooth polynomial cutoff envelope.

    ``y = 1 - 0.5*(p+1)*(p+2)*x^p + p*(p+2)*x^(p+1) - 0.5*p*(p+1)*x^(p+2)``

    Args:
        p (int): Polynomial degree. Default: 9.
    """

    def __init__(self, p: int = 9):
        super().__init__()
        self.p = p

    def forward(self, dist: paddle.Tensor) -> paddle.Tensor:
        p = self.p
        x_p = dist ** p
        x_p1 = x_p * dist
        x_p2 = x_p1 * dist
        return (
            1.0
            - 0.5 * (p + 1) * (p + 2) * x_p
            + p * (p + 2) * x_p1
            - 0.5 * p * (p + 1) * x_p2
        )


class ScaledNorm(nn.Layer):
    """Compute scaled distance and unit direction vector.

    Args:
        r (float): Cutoff radius used for normalisation.
    """

    def __init__(self, r: float):
        super().__init__()
        self.r = r

    def forward(self, disp: paddle.Tensor):
        """
        Args:
            disp: Displacement vectors [E, 3].
        Returns:
            dist: Scaled distances [E, 1].
            direction: Unit direction vectors [E, 3].
        """
        dist = paddle.norm(disp, axis=-1, keepdim=True)  # [E, 1]
        direction = disp / dist
        dist = dist / self.r
        return dist, direction


class RadiusGraph(nn.Layer):
    """Build a radius graph from atomic positions with PBC support.

    Args:
        r (float): Cutoff radius.
    """

    def __init__(self, r: float):
        super().__init__()
        self.r = r

    def forward(self, pos, cell=None, batch=None):
        """
        Args:
            pos: Atomic positions [N, 3].
            cell: Lattice vectors [B, 3, 3] or None.
            batch: Batch indices [N] or None.
        Returns:
            edge_index: [2, E] source/target indices.
            disp: Displacement vectors [E, 3].
        """
        n_node = pos.shape[0]

        # Build per-molecule full graph (excluding self-loops)
        if batch is not None:
            unique_batches = paddle.unique(batch)
            src_list, dst_list = [], []
            for b in unique_batches:
                mask = (batch == b)
                nodes = paddle.nonzero(mask).flatten()
                n = nodes.shape[0]
                # Create all pairs
                row = nodes.unsqueeze(1).expand([n, n]).flatten()
                col = nodes.unsqueeze(0).expand([n, n]).flatten()
                src_list.append(row)
                dst_list.append(col)
            src_all = paddle.concat(src_list)
            dst_all = paddle.concat(dst_list)
            edge_index = paddle.stack([src_all, dst_all], axis=0)
        else:
            idx = paddle.arange(n_node, dtype="int64")
            row = idx.unsqueeze(1).expand([n_node, n_node]).flatten()
            col = idx.unsqueeze(0).expand([n_node, n_node]).flatten()
            edge_index = paddle.stack([row, col], axis=0)

        # Remove self-loops
        non_self = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, non_self]

        # Displacement vectors
        disp = pos[edge_index[0]] - pos[edge_index[1]]

        # Apply minimum-image convention for periodic boundary conditions
        if cell is not None and not paddle.all(cell == 0.0):
            if batch is not None:
                cell_per_node = cell[batch]
            else:
                cell_per_node = cell.expand([n_node, 3, 3])
            cell_per_edge = cell_per_node[edge_index[0]]  # [E, 3, 3]
            # Solve for fractional coordinates: cell^T @ s = disp
            cell_t = paddle.transpose(cell_per_edge, perm=[0, 2, 1])  # [E, 3, 3]
            scaled = paddle.linalg.solve(cell_t, disp.unsqueeze(-1)).squeeze(-1)
            # Detach before round: round() has zero gradient anyway, and
            # solve backward segfaults on CPU in some Paddle versions.
            disp = disp - paddle.bmm(
                cell_per_edge, paddle.round(scaled.detach()).unsqueeze(-1)
            ).squeeze(-1)

        # Filter by cutoff distance
        dist_sq = (disp * disp).sum(axis=-1)
        mask = dist_sq < self.r * self.r
        edge_index = edge_index[:, mask]
        disp = disp[mask]

        return edge_index, disp


class EdgeEmbedding(nn.Layer):
    """Edge embedding combining RadiusGraph, ScaledNorm, PolynomialCutoff, and RadialBessel.

    Args:
        cutoff (float): Cutoff radius.
        n_basis (int): Number of radial basis functions.
    """

    def __init__(self, cutoff: float, n_basis: int = 20):
        super().__init__()
        self.radius_graph = RadiusGraph(r=cutoff)
        self.norm = ScaledNorm(r=cutoff)
        self.envelope = PolynomialCutoff(p=9)
        self.embedding = RadialBesselLayer(n_basis=n_basis)

    def forward(self, pos, cell=None, batch=None):
        edge_index, disp = self.radius_graph(pos, cell=cell, batch=batch)
        dist_edge, dir_edge = self.norm(disp)
        dist_edge = self.envelope(dist_edge) * self.embedding(dist_edge)
        return dist_edge, dir_edge, edge_index


# ---------------------------------------------------------------------------
# Core network components
# ---------------------------------------------------------------------------

class EmbeddingNet(nn.Layer):
    """Initial atom and edge embedding layer.

    Args:
        cutoff (float): Cutoff radius for edge embedding.
        n_features (int): Feature dimension.
        n_basis (int): Number of radial basis functions.
    """

    def __init__(self, cutoff: float, n_features: int, n_basis: int):
        super().__init__()
        self.n_features = n_features
        self.node_embedding = nn.Embedding(118 + 1, n_features, padding_idx=0)
        self.edge_embedding = EdgeEmbedding(cutoff=cutoff, n_basis=n_basis)
        self.requires_dr = False

    def forward(self, z, pos, cell, batch):
        # Node embedding
        atom_node = self.node_embedding(z)  # [N, F]
        force_node = paddle.zeros(
            [z.shape[0], 3, self.n_features], dtype=pos.dtype
        )  # [N, 3, F]

        # Strain displacement for virial/stress (identity by default)
        displacement = paddle.zeros_like(cell)
        displacement[:, 0, 0] = 1.0
        displacement[:, 1, 1] = 1.0
        displacement[:, 2, 2] = 1.0

        if self.requires_dr:
            pos.stop_gradient = False
            displacement.stop_gradient = False

        symmetric_displacement = (
            displacement + paddle.transpose(displacement, perm=[0, 2, 1])
        ) / 2.0
        # Apply strain to positions: pos_displaced = pos @ symmetric_displacement
        pos_displaced = paddle.bmm(
            pos.unsqueeze(1), symmetric_displacement[batch]
        ).squeeze(1)  # [N, 3]
        cell_displaced = paddle.bmm(
            cell, symmetric_displacement
        ).squeeze(1)  # [B, 3, 3] (squeeze does nothing if already [B,3,3])

        # Edge embedding
        dist_edge, dir_edge, edge_index = self.edge_embedding(
            pos_displaced, cell_displaced, batch
        )

        return atom_node, force_node, dir_edge, dist_edge, edge_index, displacement


class InteractionNet(nn.Layer):
    """Newtonian message passing interaction layer.

    Performs both invariant (scalar) and equivariant (3D vector) message passing.

    Args:
        n_features (int): Feature dimension.
        n_basis (int): Number of radial basis functions.
        activation (nn.Layer): Activation function instance.
        layer_norm (bool): Whether to apply LayerNorm to atom features.
    """

    def __init__(
        self,
        n_features: int,
        n_basis: int,
        activation: nn.Layer,
        layer_norm: bool,
    ):
        super().__init__()
        self.n_features = n_features

        # Invariant message passing
        self.message_nodepart = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.message_edgepart = nn.Linear(n_basis, n_features, bias_attr=False)

        # Equivariant message networks
        self.equiv_message1 = nn.Sequential(
            nn.Linear(n_features, n_features, bias_attr=False),
            activation,
            nn.Linear(n_features, n_features, bias_attr=False),
        )
        self.equiv_message2 = nn.Sequential(
            nn.Linear(n_features, n_features, bias_attr=False),
            activation,
            nn.Linear(n_features, n_features, bias_attr=False),
        )

        # Force → energy update
        self.equiv_update = nn.Linear(n_features, n_features, bias_attr=False)

        # Optional layer norm
        self.layer_norm = nn.LayerNorm(n_features) if layer_norm else None

    def forward(self, atom_node, force_node, dir_edge, dist_edge, edge_index):
        """
        Args:
            atom_node: [N, F]
            force_node: [N, 3, F]
            dir_edge: [E, 3]
            dist_edge: [E, n_basis]
            edge_index: [2, E]  (row=src, col=dst)
        Returns:
            Updated atom_node [N, F] and force_node [N, 3, F].
        """
        src, dst = edge_index[0], edge_index[1]

        # --- Invariant message ---
        message_nodepart = self.message_nodepart(atom_node)  # [N, F]
        message_edgepart = self.message_edgepart(dist_edge)  # [E, F]
        message = (
            message_edgepart
            * message_nodepart[src]
            * message_nodepart[dst]
        )  # [E, F]

        inv_update = scatter(
            message, src, dim=0, dim_size=atom_node.shape[0]
        )  # [N, F]
        atom_node = atom_node + inv_update

        # --- Equivariant message 1: direction-weighted ---
        equiv_msg1_inv = self.equiv_message1(message).unsqueeze(1)  # [E, 1, F]
        equiv_msg1_eq = dir_edge.unsqueeze(2)  # [E, 3, 1]
        equiv_msg1 = equiv_msg1_inv * equiv_msg1_eq  # [E, 3, F]

        # --- Equivariant message 2: neighbor force ---
        equiv_msg2_inv = self.equiv_message2(message).unsqueeze(1)  # [E, 1, F]
        equiv_msg2_eq = force_node[dst]  # [E, 3, F]
        equiv_msg2 = equiv_msg2_inv * equiv_msg2_eq  # [E, 3, F]

        force_update = scatter(
            equiv_msg1 + equiv_msg2, src, dim=0, dim_size=force_node.shape[0]
        )  # [N, 3, F]
        force_node = force_node + force_update

        # --- Energy update from force (dot product over xyz) ---
        inv_update2 = paddle.sum(
            force_node * self.equiv_update(force_node), axis=1
        )  # [N, F]
        atom_node = atom_node + inv_update2

        # --- Layer norm ---
        if self.layer_norm is not None:
            atom_node = self.layer_norm(atom_node)

        return atom_node, force_node


# ---------------------------------------------------------------------------
# Output layers
# ---------------------------------------------------------------------------

class EnergyOutput(nn.Layer):
    """Per-atom energy prediction via MLP.

    Args:
        n_features (int): Feature dimension.
        activation (nn.Layer): Activation function instance.
    """

    def __init__(self, n_features: int, activation: nn.Layer):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, 1),
        )

    def forward(self, atom_node: paddle.Tensor) -> paddle.Tensor:
        """Return per-atom energy [N, 1]."""
        return self.layers(atom_node)


class DirectForceOutput(nn.Layer):
    """Direct force prediction from atom features and equivariant force node.

    Args:
        n_features (int): Feature dimension.
        activation (nn.Layer): Activation function instance.
    """

    def __init__(self, n_features: int, activation: nn.Layer):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

    def forward(
        self, atom_node: paddle.Tensor, force_node: paddle.Tensor
    ) -> paddle.Tensor:
        """Return forces [N, 3]."""
        weight = self.layers(atom_node).unsqueeze(1)  # [N, 1, F]
        force = weight * force_node  # [N, 3, F]
        return force.sum(axis=-1)  # [N, 3]


class ScaleShift(nn.Layer):
    """Per-element scale and shift using atomic-number embeddings.

    Args:
        scale (bool): Whether to apply scaling.
        shift (bool): Whether to apply shifting.
    """

    def __init__(self, scale: bool = True, shift: bool = True):
        super().__init__()
        if scale:
            self.scale = nn.Embedding(119, 1, padding_idx=0)
            # Initialize scale to 1
            with paddle.no_grad():
                self.scale.weight.set_value(
                    paddle.ones_like(self.scale.weight)
                )
        else:
            self.scale = None
        if shift:
            self.shift = nn.Embedding(119, 1, padding_idx=0)
            # Initialize shift to 0 (default)
        else:
            self.shift = None

    def forward(self, output, z):
        if self.scale is not None:
            output = output * self.scale(z)
        if self.shift is not None:
            output = output + self.shift(z)
        return output


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class NewtonNet(nn.Layer):
    """Newtonian message passing network for molecular force fields.

    Follows PaddleMaterials conventions with ``forward()`` returning
    ``{"loss_dict": ..., "pred_dict": ...}``.

    Args:
        cutoff (float): Cutoff radius (Å). Default: 5.0.
        n_features (int): Hidden feature dimension. Default: 128.
        n_basis (int): Number of radial basis functions. Default: 20.
        n_interactions (int): Number of message passing layers. Default: 3.
        activation (str): Activation function name. Default: ``"swish"``.
        layer_norm (bool): Apply LayerNorm after each interaction. Default: False.
        property_names (str): Target property name. Default: ``"energy"``.
        data_mean (float): Mean for target normalisation. Default: 0.0.
        data_std (float): Std for target normalisation. Default: 1.0.
        loss_type (str): ``"mse_loss"`` or ``"l1_loss"``. Default: ``"mse_loss"``.
        force_loss_weight (float): Weight for force loss relative to energy loss.
            Default: 100.0.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        n_features: int = 128,
        n_basis: int = 20,
        n_interactions: int = 3,
        activation: str = "swish",
        layer_norm: bool = False,
        property_names: str = "energy",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "mse_loss",
        force_loss_weight: float = 100.0,
    ):
        super().__init__()

        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            self.property_names = property_names

        self.register_buffer("data_mean", paddle.to_tensor(data_mean, dtype="float32"))
        self.register_buffer("data_std", paddle.to_tensor(data_std, dtype="float32"))

        act = get_activation_by_string(activation)

        # Embedding
        self.embedding_layer = EmbeddingNet(
            cutoff=cutoff,
            n_features=n_features,
            n_basis=n_basis,
        )

        # Interaction layers
        self.interaction_layers = nn.LayerList(
            [
                InteractionNet(
                    n_features=n_features,
                    n_basis=n_basis,
                    activation=get_activation_by_string(activation),
                    layer_norm=layer_norm,
                )
                for _ in range(n_interactions)
            ]
        )

        # Energy output
        self.energy_output = EnergyOutput(
            n_features, get_activation_by_string(activation)
        )

        # Per-element energy scale/shift
        self.energy_scaler = ScaleShift(scale=True, shift=True)

        # Loss
        self.force_loss_weight = force_loss_weight

        # Loss
        if loss_type == "mse_loss":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def normalize(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor: paddle.Tensor) -> paddle.Tensor:
        return tensor * self.data_std + self.data_mean

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _forward(self, data: dict, compute_forces: bool = False):
        """Core forward pass returning per-molecule energy and optional forces.

        Args:
            data: Dict with keys ``z``, ``pos``, ``batch``, ``cell`` (optional).
            compute_forces: If True, compute forces as negative energy gradient.
        Returns:
            Tuple of (energy [B], forces [N, 3] or None).
        """
        z = data["z"]
        pos = data["pos"]
        batch = data["batch"]
        cell = data.get("cell", None)

        if compute_forces:
            pos.stop_gradient = False

        if cell is None:
            n_batch = int(batch.max().item()) + 1
            cell = paddle.zeros([n_batch, 3, 3], dtype=pos.dtype)

        # Embedding
        atom_node, force_node, dir_edge, dist_edge, edge_index, displacement = (
            self.embedding_layer(z, pos, cell, batch)
        )

        # Message passing
        for interaction in self.interaction_layers:
            atom_node, force_node = interaction(
                atom_node, force_node, dir_edge, dist_edge, edge_index
            )

        # Per-atom energy
        energy_per_atom = self.energy_output(atom_node)  # [N, 1]
        energy_per_atom = self.energy_scaler(energy_per_atom, z)  # [N, 1]

        # Compute forces as F = -dE/dpos (in physical units)
        forces = None
        if compute_forces:
            grad_result = paddle.grad(
                outputs=energy_per_atom.sum(),
                inputs=pos,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            forces = -grad_result * self.data_std

        # Aggregate per molecule
        energy = scatter(
            energy_per_atom, batch, dim=0, dim_size=int(batch.max().item()) + 1
        ).reshape([-1])  # [B]

        return energy, forces

    def forward(
        self,
        data: dict,
        return_loss: bool = True,
        return_prediction: bool = True,
    ) -> dict:
        """PaddleMaterials standard forward.

        Args:
            data: Input data dict.
            return_loss: Whether to compute loss.
            return_prediction: Whether to return predictions.
        Returns:
            Dict with ``loss_dict`` and ``pred_dict``.
        """
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        compute_forces = "forces" in data or return_prediction
        energy, forces = self._forward(data, compute_forces=compute_forces)

        loss_dict = {}
        if return_loss:
            label = data[self.property_names]
            label = self.normalize(label)
            energy_loss = self.loss_fn(input=energy, label=label)

            total_loss = energy_loss
            if "forces" in data and forces is not None:
                force_loss = self.loss_fn(input=forces, label=data["forces"])
                loss_dict["force_loss"] = force_loss
                total_loss = energy_loss + self.force_loss_weight * force_loss
            loss_dict["loss"] = total_loss

        prediction = {}
        if return_prediction:
            prediction[self.property_names] = self.unnormalize(energy)
            if forces is not None:
                prediction["forces"] = forces

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    def predict(self, data: dict, compute_forces: bool = True) -> dict:
        """Predict energy and optionally forces.

        Args:
            data: Input data dict.
            compute_forces: Whether to compute forces. Default: True.
        Returns:
            Dict mapping property name to predicted value, plus ``"forces"``
            when *compute_forces* is True.
        """
        if compute_forces:
            energy, forces = self._forward(data, compute_forces=True)
            energy = self.unnormalize(energy)
            result = {self.property_names: energy.detach()}
            if forces is not None:
                result["forces"] = forces.detach()
            return result
        else:
            with paddle.no_grad():
                energy, _ = self._forward(data, compute_forces=False)
                energy = self.unnormalize(energy)
                return {self.property_names: energy}
