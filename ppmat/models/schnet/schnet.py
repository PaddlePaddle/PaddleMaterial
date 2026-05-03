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

"""SchNet: A continuous-filter convolutional neural network for modeling quantum
interactions (Schütt et al., 2017).

This module is adapted from:
  - https://github.com/atomistic-machine-learning/schnetpack (SchNetPack v0.3)
  - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.SchNet.html

Reference:
  K. T. Schütt, P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
  SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.
  NeurIPS 2017.
"""

import math
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.utils.crystal import get_pbc_distances
from ppmat.utils.scatter import scatter


def shifted_softplus(x):
    """Shifted softplus activation: softplus(x) - ln(2).

    This ensures shifted_softplus(0) = 0, making it a smoother alternative to ReLU
    that still passes through the origin.
    """
    return paddle.nn.functional.softplus(x) - math.log(2.0)


class ShiftedSoftplus(nn.Layer):
    """Module form of shifted_softplus for use in nn.Sequential."""

    def forward(self, x):
        return paddle.nn.functional.softplus(x) - math.log(2.0)


class GaussianRBF(nn.Layer):
    """Gaussian radial basis function expansion.

    Expands interatomic distances into a set of Gaussian basis functions centered
    at evenly spaced values between 0 and cutoff.

    Args:
        n_gaussians (int): Number of Gaussian basis functions.
        cutoff (float): Cutoff distance (Å) for the basis.
        start (float): Center of first Gaussian. Default: 0.0.
    """

    def __init__(self, n_gaussians: int = 50, cutoff: float = 10.0, start: float = 0.0):
        super().__init__()
        offset = paddle.linspace(start, cutoff, n_gaussians)
        self.register_buffer(tensor=offset, name="offsets")
        width = offset[1] - offset[0] if n_gaussians > 1 else paddle.to_tensor(1.0)
        self.register_buffer(tensor=width, name="widths")

    def forward(self, dist: paddle.Tensor) -> paddle.Tensor:
        """Expand distances into Gaussian basis.

        Args:
            dist: Tensor of shape [num_edges] with interatomic distances.

        Returns:
            Tensor of shape [num_edges, n_gaussians].
        """
        dist = dist.unsqueeze(-1)
        return paddle.exp(-0.5 * ((dist - self.offsets) / self.widths) ** 2)


def cosine_cutoff(distances: paddle.Tensor, cutoff: float) -> paddle.Tensor:
    """Cosine cutoff function: smoothly decays to zero at cutoff radius.

    C(r) = 0.5 * (cos(r * pi / cutoff) + 1) for r < cutoff, else 0.
    """
    cutoffs = 0.5 * (paddle.cos(distances * math.pi / cutoff) + 1.0)
    cutoffs = cutoffs * (distances < cutoff).astype(cutoffs.dtype)
    return cutoffs


class CFConv(nn.Layer):
    """Continuous-filter convolution layer.

    Applies a learned filter on interatomic distances to weight messages
    between atoms, with a cosine cutoff envelope.

    Args:
        n_atom_basis (int): Dimension of atom feature vectors.
        n_filters (int): Number of filters (same as n_atom_basis typically).
        n_gaussians (int): Number of RBF basis functions for distance expansion.
        cutoff (float): Cutoff radius for cosine cutoff function.
    """

    def __init__(self, n_atom_basis: int, n_filters: int, n_gaussians: int, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff
        self.in2f = nn.Linear(n_atom_basis, n_filters, bias_attr=False)
        self.f2out = nn.Linear(n_filters, n_atom_basis)
        self.filter_net = nn.Sequential(
            nn.Linear(n_gaussians, n_filters),
            ShiftedSoftplus(),
            nn.Linear(n_filters, n_filters),
        )

    def forward(
        self,
        x: paddle.Tensor,
        rbf: paddle.Tensor,
        edge_src: paddle.Tensor,
        edge_dst: paddle.Tensor,
        num_nodes: int,
        dist: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Apply continuous-filter convolution.

        Args:
            x: Atom features [num_atoms, n_atom_basis].
            rbf: RBF-expanded distances [num_edges, n_gaussians].
            edge_src: Source atom indices [num_edges].
            edge_dst: Destination atom indices [num_edges].
            num_nodes: Total number of atoms in the batch.
            dist: Raw distances [num_edges] for cutoff function.

        Returns:
            Aggregated messages [num_atoms, n_atom_basis].
        """
        # Generate filter from distance features
        W = self.filter_net(rbf)
        # Apply cosine cutoff envelope
        if dist is not None:
            C = cosine_cutoff(dist, self.cutoff)
            W = W * C.unsqueeze(-1)
        # Transform input features and apply filter element-wise
        y = self.in2f(x)
        y = paddle.index_select(y, edge_src, axis=0) * W
        # Aggregate messages to destination atoms
        y = scatter(y, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
        return self.f2out(y)


class SchNetInteraction(nn.Layer):
    """SchNet interaction block.

    One interaction layer: continuous-filter convolution → dense layer
    → residual connection.

    Args:
        n_atom_basis (int): Dimension of atom feature vectors.
        n_filters (int): Number of filters in CFConv.
        n_gaussians (int): Number of Gaussian RBF basis functions.
    """

    def __init__(self, n_atom_basis: int, n_filters: int, n_gaussians: int, cutoff: float = 10.0):
        super().__init__()
        self.cfconv = CFConv(n_atom_basis, n_filters, n_gaussians, cutoff=cutoff)
        self.dense = nn.Linear(n_atom_basis, n_atom_basis)

    def forward(
        self,
        x: paddle.Tensor,
        rbf: paddle.Tensor,
        edge_src: paddle.Tensor,
        edge_dst: paddle.Tensor,
        num_nodes: int,
        dist: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Apply interaction block with residual connection.

        Args:
            x: Atom features [num_atoms, n_atom_basis].
            rbf: RBF-expanded distances [num_edges, n_gaussians].
            edge_src: Source atom indices [num_edges].
            edge_dst: Destination atom indices [num_edges].
            num_nodes: Total number of atoms in batch.
            dist: Raw distances [num_edges] for cutoff function.

        Returns:
            Updated atom features [num_atoms, n_atom_basis].
        """
        v = self.cfconv(x, rbf, edge_src, edge_dst, num_nodes, dist=dist)
        v = shifted_softplus(v)
        v = self.dense(v)
        return x + v  # residual connection


class SchNet(nn.Layer):
    """SchNet: A continuous-filter convolutional neural network for modeling
    quantum interactions.

    SchNet learns a representation of atomistic systems by iteratively refining
    atom-wise features through continuous-filter convolutional layers that operate
    on interatomic distances expanded in a Gaussian RBF basis.

    This implementation follows the PaddleMaterials model interface pattern
    with ``_forward`` / ``forward`` / ``predict`` methods and supports
    energy prediction (and optionally force prediction via autograd).

    Args:
        n_atom_basis (int): Dimension of atom feature vectors. Default: 128.
        n_interactions (int): Number of interaction blocks. Default: 6.
        n_filters (int): Number of filters in CFConv layers. If None, defaults
            to n_atom_basis. Default: None.
        cutoff (float): Cutoff distance (Å) for neighbor interactions. Default: 10.0.
        n_gaussians (int): Number of Gaussian RBF basis functions. Default: 50.
        max_z (int): Maximum atomic number for embedding. Default: 100.
        readout (str): Graph-level aggregation method ("sum" or "mean"). Default: "sum".
        property_names (str): Target property name for loss computation. Default: "energy_U0".
        data_mean (float): Mean for target normalization. Default: 0.0.
        data_std (float): Std for target normalization. Default: 1.0.
        loss_type (str): Loss function type ("l1_loss" or "mse_loss"). Default: "l1_loss".
        compute_forces (bool): Whether to compute forces via autograd. Default: False.
            Requires Paddle >= 3.1.0 for stable backward through scatter.
    """

    def __init__(
        self,
        n_atom_basis: int = 128,
        n_interactions: int = 6,
        n_filters: Optional[int] = None,
        cutoff: float = 10.0,
        n_gaussians: int = 50,
        max_z: int = 100,
        readout: str = "sum",
        property_names: Optional[str] = "energy_U0",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "l1_loss",
        compute_forces: bool = False,
        **kwargs,
    ):
        super().__init__()

        if n_filters is None:
            n_filters = n_atom_basis

        self.cutoff = cutoff
        self.readout = readout
        self.compute_forces = compute_forces

        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            assert isinstance(property_names, str)
            self.property_names = property_names

        # Normalization buffers
        self.register_buffer(
            tensor=paddle.to_tensor(data_mean, dtype="float32"), name="data_mean"
        )
        self.register_buffer(
            tensor=paddle.to_tensor(data_std, dtype="float32"), name="data_std"
        )

        # Atom embedding
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # Distance expansion
        self.rbf = GaussianRBF(n_gaussians, cutoff)

        # Interaction blocks
        self.interactions = nn.LayerList(
            [
                SchNetInteraction(n_atom_basis, n_filters, n_gaussians, cutoff=cutoff)
                for _ in range(n_interactions)
            ]
        )

        # Output network: atom features → scalar energy per atom
        self.output_network = nn.Sequential(
            nn.Linear(n_atom_basis, n_atom_basis // 2),
            nn.Linear(n_atom_basis // 2, 1),
        )

        # Loss function
        if loss_type == "mse_loss":
            self.loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = paddle.nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    def _forward(self, data):
        """Core forward pass: positions → energy (and optionally forces).

        Args:
            data: Dictionary containing 'graph' key with a PGL graph object.
                The graph must have:
                  - node_feat["atom_types"]: atomic numbers [num_atoms]
                  - node_feat["cart_coords"]: Cartesian coordinates [num_atoms, 3]
                  - node_feat["lattice"]: lattice matrix [num_atoms, 3, 3] (per-atom broadcast)
                  - node_feat["num_atoms"]: number of atoms per graph
                  - edge_feat["pbc_offset"]: periodic offsets [num_edges, 3]
                  - edge_feat["num_edges"]: number of edges per graph
                  - edges: edge index array [num_edges, 2]
                  - graph_node_id: batch assignment [num_atoms]

        Returns:
            dict with "energy" key (and optionally "forces").
        """
        # Convert graph data from numpy to paddle tensors
        data["graph"] = data["graph"].tensor()
        graph = data["graph"]

        # Unpack graph
        batch = graph.graph_node_id
        atom_types = graph.node_feat["atom_types"]
        pos = graph.node_feat["cart_coords"]
        lattices = graph.node_feat["lattice"]
        frac = graph.node_feat["frac_coords"]
        edge_index = graph.edges
        to_jimages = graph.edge_feat["pbc_offset"]
        num_atoms = graph.node_feat["num_atoms"]
        num_bonds = graph.edge_feat["num_edges"]

        if self.compute_forces:
            pos.stop_gradient = False

        # Compute PBC distances
        out = get_pbc_distances(
            frac,
            edge_index.T,
            lattices,
            to_jimages,
            num_atoms,
            num_bonds,
            return_offsets=True,
        )
        edge_index_out = out["edge_index"]
        dist = out["distances"]

        edge_src = edge_index_out[0]  # source atoms
        edge_dst = edge_index_out[1]  # destination atoms
        num_nodes = atom_types.shape[0]

        # Expand distances to RBF features
        rbf = self.rbf(dist)

        # Embed atomic numbers
        x = self.embedding(atom_types)

        # Apply interaction blocks
        for interaction in self.interactions:
            x = interaction(x, rbf, edge_src, edge_dst, num_nodes, dist=dist)

        # Output: per-atom energy
        x = shifted_softplus(self.output_network[0](x))
        atom_energy = self.output_network[1](x)

        # Aggregate to per-graph energy
        energy = scatter(atom_energy, batch, dim=0, reduce=self.readout)

        result = {"energy": energy}

        # Optionally compute forces via autograd
        if self.compute_forces:
            grad = paddle.grad(
                outputs=energy.sum(),
                inputs=pos,
                create_graph=self.training,
                retain_graph=self.training,
            )
            forces = -grad[0]
            result["forces"] = forces

        return result

    def forward(self, data, return_loss=True, return_prediction=True):
        """Standard PaddleMaterials forward method.

        Args:
            data: Dictionary with 'graph' and target property.
            return_loss: Whether to compute and return loss.
            return_prediction: Whether to return predictions.

        Returns:
            dict with "loss_dict" and "pred_dict" keys.
        """
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_names]
            label = self.normalize(label)
            loss = self.loss_fn(
                input=pred["energy"],
                label=label,
            )
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            energy = self.unnormalize(pred["energy"])
            prediction[self.property_names] = energy
            if "forces" in pred:
                prediction["forces"] = pred["forces"]

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, graphs):
        """Run inference on one or more graphs.

        Args:
            graphs: A single PGL graph or a list of PGL graphs.

        Returns:
            A dict or list of dicts with predicted property values.
        """
        if isinstance(graphs, list):
            results = []
            for graph in graphs:
                result = self._forward({"graph": graph})
                energy = self.unnormalize(result["energy"]).numpy()[0, 0]
                results.append({self.property_names: energy})
            return results
        else:
            data = {"graph": graphs}
            result = self._forward(data)
            energy = self.unnormalize(result["energy"]).numpy()[0, 0]
            return {self.property_names: energy}
