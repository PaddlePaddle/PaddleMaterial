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

"""CHGNet crystal graph neural network for material structure embedding.

Cannot reuse Paddle built-in layers: crystal graph convolution (message passing
between atoms, bonds, and angles), RBF basis expansions, and crystal graph batching
(BatchedGraph) are domain-specific to materials science. PaddlePaddle has no
equivalent graph operations for periodic crystal structures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence

import paddle
from paddle import Tensor, nn
from pymatgen.core import Structure

from ppmat.models.matterchat.chgnet.graph import CrystalGraph, CrystalGraphConverter
from ppmat.models.matterchat.chgnet.graph.crystalgraph import datatype
from ppmat.models.matterchat.chgnet.model.composition_model import AtomRef
from ppmat.models.matterchat.chgnet.model.encoders import (
    AngleEncoder,
    AtomEmbedding,
    BondEncoder,
)
from ppmat.models.matterchat.chgnet.model.functions import (
    MLP,
    GatedMLP,
    find_normalization,
)
from ppmat.models.matterchat.chgnet.model.layers import (
    AngleUpdate,
    AtomConv,
    BondConv,
    GraphAttentionReadOut,
    GraphPooling,
)
from ppmat.utils import logger

if TYPE_CHECKING:
    PredTask = str


class CHGNet(nn.Layer):
    """Crystal Hamiltonian Graph neural Network
    A model that takes in a crystal graph and outputs atom feature embeddings.
    """

    def __init__(
        self,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        angle_fea_dim: int = 64,
        composition_model: str | nn.Layer = "MPtrj",
        num_radial: int = 9,
        num_angular: int = 9,
        n_conv: int = 4,
        atom_conv_hidden_dim: Sequence[int] | int = 64,
        update_bond: bool = True,
        bond_conv_hidden_dim: Sequence[int] | int = 64,
        update_angle: bool = True,
        angle_layer_hidden_dim: Sequence[int] | int = 0,
        conv_dropout: float = 0,
        read_out: str = "ave",
        mlp_hidden_dims: Sequence[int] | int = (64, 64),
        mlp_dropout: float = 0,
        mlp_first: bool = True,
        is_intensive: bool = True,
        non_linearity: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        atom_graph_cutoff: int = 5,
        bond_graph_cutoff: int = 3,
        cutoff_coeff: int = 5,
        learnable_rbf: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the CHGNet."""
        self.model_args = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "kwargs"]
        }
        self.model_args.update(kwargs)

        super().__init__()
        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_dim = bond_fea_dim
        self.is_intensive = is_intensive
        self.n_conv = n_conv

        if isinstance(composition_model, nn.Layer):
            self.composition_model = composition_model
        elif isinstance(composition_model, str):
            self.composition_model = AtomRef(is_intensive=is_intensive)
            self.composition_model.initialize_from(composition_model)
        else:
            self.composition_model = None

        if self.composition_model is not None:
            for param in self.composition_model.parameters():
                param.stop_gradient = True

        self.graph_converter = CrystalGraphConverter(
            atom_graph_cutoff=atom_graph_cutoff, bond_graph_cutoff=bond_graph_cutoff
        )

        self.atom_embedding = AtomEmbedding(atom_feature_dim=atom_fea_dim)
        self.bond_basis_expansion = BondEncoder(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            num_radial=num_radial,
            cutoff_coeff=cutoff_coeff,
            learnable=learnable_rbf,
        )
        self.bond_embedding = nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias_attr=False
        )
        self.bond_weights_ag = nn.Linear(
            in_features=num_radial, out_features=atom_fea_dim, bias_attr=False
        )
        self.bond_weights_bg = nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias_attr=False
        )
        self.angle_basis_expansion = AngleEncoder(
            num_angular=num_angular, learnable=learnable_rbf
        )
        self.angle_embedding = nn.Linear(
            in_features=num_angular, out_features=angle_fea_dim, bias_attr=False
        )

        conv_norm = kwargs.pop("conv_norm", None)
        gMLP_norm = kwargs.pop("gMLP_norm", None)
        atom_graph_layers = []
        for _i in range(n_conv):
            atom_graph_layers.append(
                AtomConv(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    hidden_dim=atom_conv_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    use_mlp_out=True,
                    resnet=True,
                )
            )
        self.atom_conv_layers = nn.LayerList(atom_graph_layers)

        if update_bond is True:
            bond_graph_layers = [
                BondConv(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=bond_conv_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    use_mlp_out=True,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.bond_conv_layers = nn.LayerList(bond_graph_layers)
        else:
            self.bond_conv_layers = [None for _ in range(n_conv - 1)]

        if update_angle is True:
            angle_layers = [
                AngleUpdate(
                    atom_fea_dim=atom_fea_dim,
                    bond_fea_dim=bond_fea_dim,
                    angle_fea_dim=angle_fea_dim,
                    hidden_dim=angle_layer_hidden_dim,
                    dropout=conv_dropout,
                    activation=non_linearity,
                    norm=conv_norm,
                    gMLP_norm=gMLP_norm,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.angle_layers = nn.LayerList(angle_layers)
        else:
            self.angle_layers = [None for _ in range(n_conv - 1)]

        self.site_wise = nn.Linear(atom_fea_dim, 1)
        self.readout_norm = find_normalization(
            name=kwargs.pop("readout_norm", None), dim=atom_fea_dim
        )
        self.mlp_first = mlp_first
        if mlp_first:
            self.read_out_type = "sum"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=False)
        elif read_out in ["attn", "weighted"]:
            self.read_out_type = "attn"
            num_heads = kwargs.pop("num_heads", 3)
            self.pooling = GraphAttentionReadOut(
                atom_fea_dim, num_head=num_heads, average=True
            )
            input_dim = atom_fea_dim * num_heads
        else:
            self.read_out_type = "ave"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=True)
        if kwargs.pop("final_mlp", "MLP") in ["normal", "MLP"]:
            self.mlp = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dims,
                output_dim=1,
                dropout=mlp_dropout,
                activation=non_linearity,
            )
        else:
            self.mlp = nn.Sequential(
                GatedMLP(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dims,
                    output_dim=mlp_hidden_dims[-1],
                    dropout=mlp_dropout,
                    activation=non_linearity,
                ),
                nn.Linear(
                    in_features=mlp_hidden_dims[-1], out_features=1
                ),
            )

        logger.message(
            f"CHGNet initialized with {sum(p.numel() for p in self.parameters()):,} "
            f"parameters"
        )

    def forward(
        self,
        graphs: Sequence[CrystalGraph],
        task: PredTask = "e",
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Get prediction associated with input graphs."""
        compute_force = "f" in task
        compute_stress = "s" in task
        site_wise = "m" in task

        comp_energy = (
            0 if self.composition_model is None else self.composition_model(graphs)
        )

        batched_graph = BatchedGraph.from_graphs(
            graphs,
            bond_basis_expansion=self.bond_basis_expansion,
            angle_basis_expansion=self.angle_basis_expansion,
            compute_stress=compute_stress,
        )

        feas_embedding = self._compute_embedding(
            batched_graph,
            site_wise=site_wise,
            compute_force=compute_force,
            compute_stress=compute_stress,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )
        return feas_embedding

    def _compute_embedding(
        self,
        g,
        site_wise: bool = False,
        compute_force: bool = False,
        compute_stress: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Get atom feature embeddings from input graphs."""
        prediction = {}
        atoms_per_graph = paddle.bincount(g.atom_owners)
        prediction["atoms_per_graph"] = atoms_per_graph

        # H is the first embedding column
        atom_feas = self.atom_embedding(g.atomic_numbers - 1)
        bond_feas = self.bond_embedding(g.bond_bases_ag)
        bond_weights_ag = self.bond_weights_ag(g.bond_bases_ag)
        bond_weights_bg = self.bond_weights_bg(g.bond_bases_bg)
        if len(g.angle_bases) != 0:
            angle_feas = self.angle_embedding(g.angle_bases)

        for idx, (atom_layer, bond_layer, angle_layer) in enumerate(
            zip(self.atom_conv_layers[:-1], self.bond_conv_layers, self.angle_layers)
        ):
            atom_feas = atom_layer(
                atom_feas=atom_feas,
                bond_feas=bond_feas,
                bond_weights=bond_weights_ag,
                atom_graph=g.batched_atom_graph,
                directed2undirected=g.directed2undirected,
            )

            if len(g.angle_bases) != 0 and bond_layer is not None:
                bond_feas = bond_layer(
                    atom_feas=atom_feas,
                    bond_feas=bond_feas,
                    bond_weights=bond_weights_bg,
                    angle_feas=angle_feas,
                    bond_graph=g.batched_bond_graph,
                )

                if angle_layer is not None:
                    angle_feas = angle_layer(
                        atom_feas=atom_feas,
                        bond_feas=bond_feas,
                        angle_feas=angle_feas,
                        bond_graph=g.batched_bond_graph,
                    )
            if idx == self.n_conv - 2:
                if return_atom_feas is True:
                    prediction["atom_fea"] = paddle.split(
                        atom_feas, atoms_per_graph.tolist()
                    )
                if site_wise:
                    magmom = paddle.abs(self.site_wise(atom_feas))
                    prediction["m"] = list(
                        paddle.split(
                            magmom.reshape([-1]), atoms_per_graph.tolist()
                        )
                    )

        atom_feas = self.atom_conv_layers[-1](
            atom_feas=atom_feas,
            bond_feas=bond_feas,
            bond_weights=bond_weights_ag,
            atom_graph=g.batched_atom_graph,
            directed2undirected=g.directed2undirected,
        )
        if self.readout_norm is not None:
            atom_feas = self.readout_norm(atom_feas)

        return atom_feas

    def predict_structure_embedding(
        self,
        structure: Structure | Sequence[Structure],
        task: PredTask = "efsm",
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 100,
    ) -> dict[str, Tensor]:
        """Predict embeddings from pymatgen.core.Structure."""
        assert (
            self.graph_converter is not None
        ), "self.graph_converter needs to be initialized first!"
        if type(structure) == Structure:
            graph = self.graph_converter(structure)
            return self.predict_graph(
                graph,
                task=task,
                return_atom_feas=return_atom_feas,
                return_crystal_feas=return_crystal_feas,
                batch_size=batch_size,
            )
        if type(structure) == list:
            graphs = [self.graph_converter(i) for i in structure]
            return self.predict_graph(
                graphs,
                task=task,
                return_atom_feas=return_atom_feas,
                return_crystal_feas=return_crystal_feas,
                batch_size=batch_size,
            )
        raise Exception("input should either be a structure or list of structures!")

    def predict_graph(
        self,
        graph: CrystalGraph | Sequence[CrystalGraph],
        task: PredTask = "efsm",
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
        batch_size: int = 100,
    ) -> dict[str, Tensor]:
        """Predict from CrystalGraph."""
        model_device = next(iter(self.parameters())).place
        self.eval()
        output = self.forward(
            [graph.to(model_device)],
            task=task,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )

        return output

    @staticmethod
    def split(x: Tensor, n: Tensor) -> Sequence[Tensor]:
        """Split a batched result Tensor into a list of Tensors."""
        start = 0
        result = []
        for i in n:
            result.append(x[start : start + i])
            start += i
        assert start == len(x), "Error: source tensor not correctly split!"
        return result

    def as_dict(self):
        """Return the CHGNet weights and args in a dictionary."""
        return {"state_dict": self.state_dict(), "model_args": self.model_args}

    @classmethod
    def from_dict(cls, dict, **kwargs):
        """Build a CHGNet from a saved dictionary."""
        chgnet = CHGNet(**dict["model_args"])
        chgnet.set_state_dict(dict["state_dict"])
        return chgnet

    @classmethod
    def from_file(cls, path, **kwargs):
        """Build a CHGNet from a saved file."""
        state = paddle.load(path)
        return CHGNet.from_dict(state["model"], **kwargs)

    @classmethod
    def load(cls, model_name="MPtrj-efsm"):
        """Load pretrained CHGNet."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if model_name == "MPtrj-efsm":
            return cls.from_file(
                os.path.join(current_dir, "../pretrained/e30f77s348m32.pth.tar")
            )
        raise Exception("model_name not supported")


@dataclass
class BatchedGraph:
    """Batched crystal graph for parallel computing."""

    atomic_numbers: Tensor
    bond_bases_ag: Tensor
    bond_bases_bg: Tensor
    angle_bases: Tensor
    batched_atom_graph: Tensor
    batched_bond_graph: Tensor
    atom_owners: Tensor
    directed2undirected: Tensor
    atom_positions: Sequence[Tensor]
    strains: Sequence[Tensor]
    volumes: Sequence[Tensor]

    @classmethod
    def from_graphs(
        cls,
        graphs: Sequence[CrystalGraph],
        bond_basis_expansion: nn.Layer,
        angle_basis_expansion: nn.Layer,
        compute_stress: bool = False,
    ) -> BatchedGraph:
        """Featurize and assemble a list of graphs."""
        atomic_numbers, atom_positions = [], []
        strains, volumes = [], []
        bond_bases_ag, bond_bases_bg, angle_bases = [], [], []
        batched_atom_graph, batched_bond_graph = [], []
        directed2undirected = []
        atom_owners = []
        atom_offset_idx = 0
        n_undirected = 0

        for graph_idx, graph in enumerate(graphs):
            n_atom = graph.atomic_number.shape[0]
            atomic_numbers.append(graph.atomic_number)

            if compute_stress:
                strain = paddle.zeros([3, 3])
                strain.stop_gradient = False
                lattice = graph.lattice @ (
                    paddle.eye(3).cast(strain.dtype) + strain
                )
            else:
                strain = None
                lattice = graph.lattice
            volumes.append(paddle.linalg.det(lattice))
            strains.append(strain)

            atom_cart_coords = graph.atom_frac_coord @ lattice
            bond_basis_ag, bond_basis_bg, bond_vectors = bond_basis_expansion(
                center=atom_cart_coords[graph.atom_graph[:, 0]],
                neighbor=atom_cart_coords[graph.atom_graph[:, 1]],
                undirected2directed=graph.undirected2directed,
                image=graph.neighbor_image,
                lattice=lattice,
            )
            atom_positions.append(atom_cart_coords)
            bond_bases_ag.append(bond_basis_ag)
            bond_bases_bg.append(bond_basis_bg)

            batched_atom_graph.append(graph.atom_graph + atom_offset_idx)
            directed2undirected.append(
                graph.directed2undirected + n_undirected
            )

            if len(graph.bond_graph) != 0:
                bond_vecs_i = paddle.index_select(
                    bond_vectors, 0, graph.bond_graph[:, 2]
                )
                bond_vecs_j = paddle.index_select(
                    bond_vectors, 0, graph.bond_graph[:, 4]
                )
                angle_basis = angle_basis_expansion(bond_vecs_i, bond_vecs_j)
                angle_bases.append(angle_basis)

                bond_graph = paddle.zeros(
                    [graph.bond_graph.shape[0], 3], dtype=graph.bond_graph.dtype
                )
                bond_graph[:, 0] = graph.bond_graph[:, 0] + atom_offset_idx
                bond_graph[:, 1] = graph.bond_graph[:, 1] + n_undirected
                bond_graph[:, 2] = graph.bond_graph[:, 3] + n_undirected
                batched_bond_graph.append(bond_graph)

            atom_owners.append(
                paddle.ones([n_atom], dtype="float32") * graph_idx
            )
            atom_offset_idx += n_atom
            n_undirected += len(bond_basis_ag)

        atomic_numbers = paddle.concat(atomic_numbers, axis=0)
        bond_bases_ag = paddle.concat(bond_bases_ag, axis=0)
        bond_bases_bg = paddle.concat(bond_bases_bg, axis=0)
        angle_bases = (
            paddle.concat(angle_bases, axis=0)
            if len(angle_bases) != 0
            else paddle.to_tensor([])
        )
        batched_atom_graph = paddle.concat(batched_atom_graph, axis=0)
        if batched_bond_graph != []:
            batched_bond_graph = paddle.concat(batched_bond_graph, axis=0)
        else:
            batched_bond_graph = paddle.to_tensor([])
        atom_owners = (
            paddle.concat(atom_owners, axis=0)
            .cast("int32")
        )
        if atomic_numbers.place.is_gpu_place():
            atom_owners = atom_owners.to(atomic_numbers.place)
        directed2undirected = paddle.concat(directed2undirected, axis=0)
        volumes = paddle.to_tensor(
            volumes, dtype=datatype
        )

        return cls(
            atomic_numbers=atomic_numbers,
            bond_bases_ag=bond_bases_ag,
            bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases,
            batched_atom_graph=batched_atom_graph,
            batched_bond_graph=batched_bond_graph,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
            atom_positions=atom_positions,
            strains=strains,
            volumes=volumes,
        )
