from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

import numpy as np
import paddle

from ppmat.datasets.structure_converter import Structure2Graph
from ppmat.models.chgnet_v2.model.composition_model import AtomRef
from ppmat.models.chgnet_v2.model.encoders import AngleEncoder
from ppmat.models.chgnet_v2.model.encoders import AtomEmbedding
from ppmat.models.chgnet_v2.model.encoders import BondEncoder
from ppmat.models.chgnet_v2.model.functions import MLP
from ppmat.models.chgnet_v2.model.functions import GatedMLP
from ppmat.models.chgnet_v2.model.functions import find_normalization
from ppmat.models.chgnet_v2.model.layers import AngleUpdate
from ppmat.models.chgnet_v2.model.layers import AtomConv
from ppmat.models.chgnet_v2.model.layers import BondConv
from ppmat.models.chgnet_v2.model.layers import GraphAttentionReadOut
from ppmat.models.chgnet_v2.model.layers import GraphPooling
from ppmat.utils import logger
from ppmat.utils.crystal import frac_to_cart_coords

if TYPE_CHECKING:
    from chgnet_v2 import PredTask
module_dir = os.path.dirname(os.path.abspath(__file__))


class CHGNet_v2(paddle.nn.Layer):
    """Crystal Hamiltonian Graph neural Network
    A model that takes in a crystal graph and output energy, force, magmom, stress.
    """

    def __init__(
        self,
        *,
        atom_fea_dim: int = 64,
        bond_fea_dim: int = 64,
        angle_fea_dim: int = 64,
        composition_model: str | paddle.nn.Layer = "MPtrj",
        num_radial: int = 31,
        num_angular: int = 31,
        n_conv: int = 4,
        atom_conv_hidden_dim: Sequence[int] | int = 64,
        update_bond: bool = True,
        bond_conv_hidden_dim: Sequence[int] | int = 64,
        update_angle: bool = True,
        angle_layer_hidden_dim: Sequence[int] | int = 0,
        conv_dropout: float = 0,
        read_out: str = "ave",
        mlp_hidden_dims: Sequence[int] | int = (64, 64, 64),
        mlp_dropout: float = 0,
        mlp_first: bool = True,
        is_intensive: bool = True,
        non_linearity: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        atom_graph_cutoff: float = 6,
        bond_graph_cutoff: float = 3,
        cutoff_coeff: int = 8,
        learnable_rbf: bool = True,
        gMLP_norm: str | None = "layer",
        readout_norm: str | None = "layer",
        task: str = "ef",
        graph_converter_cfg: dict | None = None,
        is_freeze: bool = False,
        version: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize CHGNet.

        Args:
            atom_fea_dim (int): atom feature vector embedding dimension.
                Default = 64
            bond_fea_dim (int): bond feature vector embedding dimension.
                Default = 64
            angle_fea_dim (int): angle feature vector embedding dimension.
                Default = 64
            bond_fea_dim (int): angle feature vector embedding dimension.
                Default = 64
            composition_model (nn.Module, optional): attach a composition model to
                predict energy or initialize a pretrained linear regression (AtomRef).
                The default 'MPtrj' is the atom reference energy linear regression
                trained on all Materials Project relaxation trajectories
                Default = 'MPtrj'
            num_radial (int): number of radial basis used in bond basis expansion.
                Default = 9
            num_angular (int): number of angular basis used in angle basis expansion.
                Default = 9
            n_conv (int): number of interaction blocks.
                Default = 4
                Note: last interaction block contain only an atom_conv layer
            atom_conv_hidden_dim (List or int): hidden dimensions of
                atom convolution layers.
                Default = 64
            update_bond (bool): whether to use bond_conv_layer in bond graph to
                update bond embeddings
                Default = True.
            bond_conv_hidden_dim (List or int): hidden dimensions of
                bond convolution layers.
                Default = 64
            update_angle (bool): whether to use angle_update_layer to
                update angle embeddings.
                Default = True
            angle_layer_hidden_dim (List or int): hidden dimensions of angle layers.
                Default = 0
            conv_dropout (float): dropout rate in all conv_layers.
                Default = 0
            read_out (str): method for pooling layer, 'ave' for standard
                average pooling, 'attn' for multi-head attention.
                Default = "ave"
            mlp_hidden_dims (int or list): readout multilayer perceptron
                hidden dimensions.
                Default = [64, 64]
            mlp_dropout (float): dropout rate in readout MLP.
                Default = 0.
            is_intensive (bool): whether the energy training label is intensive
                i.e. energy per atom.
                Default = True
            non_linearity ('silu' | 'relu' | 'tanh' | 'gelu'): The name of the
                activation function to use in the gated MLP.
                Default = "silu".
            mlp_first (bool): whether to apply mlp first then pooling.
                if set to True, then CHGNet is essentially calculating energy for each
                atom, them sum them up, this is used for the pretrained model
                Default = True
            atom_graph_cutoff (float): cutoff radius (A) in creating atom_graph,
                this need to be consistent with the value in training dataloader
                Default = 5
            bond_graph_cutoff (float): cutoff radius (A) in creating bond_graph,
                this need to be consistent with value in training dataloader
                Default = 3
            cutoff_coeff (float): cutoff strength used in graph smooth cutoff function.
                the smaller this coeff is, the smoother the basis is
                Default = 5
            learnable_rbf (bool): whether to set the frequencies in rbf and Fourier
                basis functions learnable.
                Default = True
            gMLP_norm (str): normalization layer to use in gate-MLP
                Default = 'layer'
            readout_norm (str): normalization layer to use before readout layer
                Default = 'layer'
            version (str): Pretrained checkpoint version.
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.atom_fea_dim = atom_fea_dim
        self.bond_fea_dim = bond_fea_dim
        self.is_intensive = is_intensive
        self.n_conv = n_conv
        self.task = task
        self.graph_converter_cfg = graph_converter_cfg
        self.is_freeze = is_freeze
        if graph_converter_cfg is not None:
            self.graph_converter = Structure2Graph(**graph_converter_cfg)
        else:
            self.graph_converter = None

        if isinstance(composition_model, paddle.nn.Layer):
            self.composition_model = composition_model
        elif isinstance(composition_model, str):
            self.composition_model = AtomRef(is_intensive=is_intensive)
            self.composition_model.initialize_from(composition_model)
        else:
            self.composition_model = None
        if self.composition_model is not None:
            for param in self.composition_model.parameters():
                param.stop_gradient = True
        self.atom_embedding = AtomEmbedding(atom_feature_dim=atom_fea_dim)
        self.bond_basis_expansion = BondEncoder(
            atom_graph_cutoff=atom_graph_cutoff,
            bond_graph_cutoff=bond_graph_cutoff,
            num_radial=num_radial,
            cutoff_coeff=cutoff_coeff,
            learnable=learnable_rbf,
        )
        self.bond_embedding = paddle.nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias_attr=False
        )
        self.bond_weights_ag = paddle.nn.Linear(
            in_features=num_radial, out_features=atom_fea_dim, bias_attr=False
        )
        self.bond_weights_bg = paddle.nn.Linear(
            in_features=num_radial, out_features=bond_fea_dim, bias_attr=False
        )
        self.angle_basis_expansion = AngleEncoder(
            num_angular=num_angular, learnable=learnable_rbf
        )
        self.angle_embedding = paddle.nn.Linear(
            in_features=num_angular, out_features=angle_fea_dim, bias_attr=False
        )
        conv_norm = kwargs.pop("conv_norm", None)
        mlp_out_bias = kwargs.pop("mlp_out_bias", False)
        atom_graph_layers = [
            AtomConv(
                atom_fea_dim=atom_fea_dim,
                bond_fea_dim=bond_fea_dim,
                hidden_dim=atom_conv_hidden_dim,
                dropout=conv_dropout,
                activation=non_linearity,
                norm=conv_norm,
                gMLP_norm=gMLP_norm,
                use_mlp_out=True,
                mlp_out_bias=mlp_out_bias,
                resnet=True,
            )
            for _ in range(n_conv)
        ]
        self.atom_conv_layers = paddle.nn.LayerList(sublayers=atom_graph_layers)
        if update_bond:
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
                    mlp_out_bias=mlp_out_bias,
                    resnet=True,
                )
                for _ in range(n_conv - 1)
            ]
            self.bond_conv_layers = paddle.nn.LayerList(sublayers=bond_graph_layers)
        else:
            self.bond_conv_layers = [None for _ in range(n_conv - 1)]
        if update_angle:
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
            self.angle_layers = paddle.nn.LayerList(sublayers=angle_layers)
        else:
            self.angle_layers = [None for _ in range(n_conv - 1)]
        self.site_wise = paddle.nn.Linear(in_features=atom_fea_dim, out_features=1)
        self.readout_norm = find_normalization(readout_norm, dim=atom_fea_dim)
        self.mlp_first = mlp_first
        if mlp_first:
            self.read_out_type = "sum"
            input_dim = atom_fea_dim
            self.pooling = GraphPooling(average=False)
        elif read_out in {"attn", "weighted"}:
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
        if kwargs.pop("final_mlp", "MLP") in {"normal", "MLP"}:
            self.mlp = MLP(
                input_dim=input_dim,
                hidden_dim=mlp_hidden_dims,
                output_dim=1,
                dropout=mlp_dropout,
                activation=non_linearity,
            )
        else:
            self.mlp = paddle.nn.Sequential(
                GatedMLP(
                    input_dim=input_dim,
                    hidden_dim=mlp_hidden_dims,
                    output_dim=mlp_hidden_dims[-1],
                    dropout=mlp_dropout,
                    norm=gMLP_norm,
                    activation=non_linearity,
                ),
                paddle.nn.Linear(in_features=mlp_hidden_dims[-1], out_features=1),
            )
        version_str = f" v{version}" if version else ""
        logger.info(
            f"CHGNet{version_str} initialized with {self.n_params:,} parameters"
        )
        if is_freeze:
            self.freeze_weights()
            logger.info("Some weights are frozen.")

    def freeze_weights(self) -> None:
        for layer in [
            self.atom_embedding,
            self.bond_embedding,
            self.angle_embedding,
            self.bond_basis_expansion,
            self.angle_basis_expansion,
            self.atom_conv_layers[:3],
            self.bond_conv_layers,
            self.angle_layers,
        ]:
            for param in layer.parameters():
                param.stop_gradient = True

    @property
    def n_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.size for p in self.parameters())

    def forward(
        self,
        batch_data,
        # graphs: Sequence[CrystalGraph],
        *,
        task: PredTask | None = "ef",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict[str, paddle.Tensor]:
        """Get prediction associated with input graphs
        Args:
            graphs (List): a list of CrystalGraphs
            task (str): the prediction task. One of 'e', 'em', 'ef', 'efs', 'efsm'.
                Default = 'e'
            return_site_energies (bool): whether to return per-site energies,
                only available if self.mlp_first == True
                Default = False
            return_atom_feas (bool): whether to return the atom features before last
                conv layer.
                Default = False
            return_crystal_feas (bool): whether to return crystal feature.
                Default = False
        Returns:
            model output (dict).
        """
        task = task if task is not None else self.task
        graphs = batch_data["graph"]
        comp_energy = (
            0 if self.composition_model is None else self.composition_model(graphs)
        )

        atom_graph = graphs.edge_feat["atom_graph"].astype("int32")
        num_atoms = graphs.node_feat["num_atoms"].astype("int32")
        num_edges = graphs.edge_feat["num_edges"].astype("int32")
        batch_size = graphs.num_graph
        atom_owners = graphs.graph_node_id
        directed2undirected = graphs.edge_feat["directed2undirected"].astype("int32")
        undirected2directed = graphs.edge_feat["undirected2directed"].astype("int32")

        atomic_numbers = graphs.node_feat["atom_types"].astype("int32")

        frac_coords = graphs.node_feat["frac_coords"]
        frac_coords.stop_gradient = False
        lattice = graphs.node_feat["lattice"]
        lattice.stop_gradient = False

        if "s" not in task:
            strains = None
            volumes = None
        else:
            strains = paddle.to_tensor(
                paddle.zeros([batch_size, 3, 3], dtype="float32"), stop_gradient=False
            )
            lattice = paddle.matmul(
                lattice, paddle.eye(3, dtype="float32")[None, :, :] + strains
            )

            volumes = paddle.dot(
                lattice[:, 0], paddle.cross(x=lattice[:, 1], y=lattice[:, 2], axis=-1)
            )
            volumes.stop_gradient = True

        if "s" not in task:
            # 使用einsum 与 @ 计算矩阵乘法有误差
            atom_positions = frac_to_cart_coords(
                frac_coords,
                num_atoms=num_atoms,
                lattices=lattice,
            )
        else:
            atom_positions = []
            start = 0
            for i in range(batch_size):
                end = start + num_atoms[i]
                atom_positions.append(frac_coords[start:end] @ lattice[i])
                start = end
            atom_positions = paddle.concat(atom_positions)

        # 存储了每个晶体图的边信息，shape=[2, N], 其中N为边数，
        # 每个元素代表了两个原子的index
        atom_graph = graphs.edge_feat["atom_graph"]
        num_atom_graph = graphs.edge_feat["num_atom_graph"]
        num_atoms_cumsum = paddle.cumsum(num_atoms)
        num_atoms_cumsum = paddle.concat(
            [paddle.zeros(1, dtype=num_atoms_cumsum.dtype), num_atoms_cumsum]
        )
        num_atoms_cumsum = num_atoms_cumsum[:-1]
        # 将index转换为整体图的index
        atom_graph_offset = paddle.repeat_interleave(num_atoms_cumsum, num_atom_graph)
        atom_graph = atom_graph + atom_graph_offset[:, None]

        # 计算晶体图中每个边的向量和距离
        center = atom_positions[atom_graph[:, 0]]
        neighbor = atom_positions[atom_graph[:, 1]]
        image = graphs.edge_feat["image"]

        if "s" not in task:
            lattice_edges = paddle.repeat_interleave(
                x=lattice, repeats=num_edges, axis=0
            )
            offset = paddle.einsum("bi,bij->bj", image, lattice_edges)
        else:
            offset = []
            start = 0
            for i in range(batch_size):
                end = start + num_edges[i]
                offset.append(image[start:end] @ lattice[i])
                start = end

            offset = paddle.concat(offset)

        neighbor = neighbor + offset
        bond_vectors = center - neighbor
        bond_lengths = paddle.linalg.norm(x=bond_vectors, axis=1)
        bond_vectors = bond_vectors / bond_lengths[:, None]

        # 对每个晶体图中的边的个数做累加，并将其作为偏移量添加到每个边向量的索引中
        num_edges_cumsum = paddle.cumsum(num_edges)
        num_edges_cumsum = paddle.concat(
            [paddle.zeros(1, dtype=num_edges_cumsum.dtype), num_edges_cumsum]
        )
        num_edges_cumsum = num_edges_cumsum[:-1]

        undirected2directed_offset = paddle.repeat_interleave(
            num_edges_cumsum, graphs.edge_feat["undirected2directed_len"]
        )
        undirected2directed = undirected2directed + undirected2directed_offset

        # 去出无向边对应的长度
        undirected_bond_lengths = paddle.gather(
            x=bond_lengths, axis=0, index=undirected2directed
        )

        bond_bases_ag = self.bond_basis_expansion.rbf_expansion_ag(
            undirected_bond_lengths
        )
        bond_bases_bg = self.bond_basis_expansion.rbf_expansion_bg(
            undirected_bond_lengths
        )

        num_bond_graph = graphs.edge_feat["num_bond_graph"]
        bond_vec_index_offset = paddle.repeat_interleave(
            num_edges_cumsum, num_bond_graph
        )

        # graphs.edge_feat["bond_graph"] = graphs.edge_feat["bond_graph"]

        undirected2directed_len_cumsum = paddle.cumsum(
            graphs.edge_feat["undirected2directed_len"]
        )
        undirected2directed_len_cumsum = paddle.concat(
            [
                paddle.zeros(1, dtype=undirected2directed_len_cumsum.dtype),
                undirected2directed_len_cumsum,
            ]
        )
        undirected2directed_len_cumsum = undirected2directed_len_cumsum[:-1]

        if num_bond_graph.max() != 0:
            bond_vecs_i_index = (
                graphs.edge_feat["bond_graph"][:, 2] + bond_vec_index_offset
            )
            bond_vecs_j_index = (
                graphs.edge_feat["bond_graph"][:, 4] + bond_vec_index_offset
            )
            bond_vecs_i = paddle.gather(x=bond_vectors, axis=0, index=bond_vecs_i_index)
            bond_vecs_j = paddle.gather(x=bond_vectors, axis=0, index=bond_vecs_j_index)
            angle_bases = self.angle_basis_expansion(bond_vecs_i, bond_vecs_j)

            bond_graph_new = paddle.zeros([graphs.edge_feat["bond_graph"].shape[0], 3])
            offset_tmp = paddle.repeat_interleave(num_atoms_cumsum, num_bond_graph)
            bond_graph_new[:, 0] = graphs.edge_feat["bond_graph"][:, 0] + offset_tmp

            offset_tmp = paddle.repeat_interleave(
                undirected2directed_len_cumsum, num_bond_graph
            )
            bond_graph_new[:, 1] = graphs.edge_feat["bond_graph"][:, 1] + offset_tmp
            bond_graph_new[:, 2] = graphs.edge_feat["bond_graph"][:, 3] + offset_tmp
        else:
            angle_bases = paddle.to_tensor(data=[])
            bond_graph_new = paddle.to_tensor(data=[])

        offset_tmp = paddle.repeat_interleave(undirected2directed_len_cumsum, num_edges)
        directed2undirected = directed2undirected + offset_tmp

        batched_graph = BatchedGraph(
            atomic_numbers=atomic_numbers,
            bond_bases_ag=bond_bases_ag,
            bond_bases_bg=bond_bases_bg,
            angle_bases=angle_bases,
            batched_atom_graph=atom_graph,
            batched_bond_graph=bond_graph_new,
            atom_owners=atom_owners,
            directed2undirected=directed2undirected,
            atom_positions=atom_positions,  # atom_positions_list,
            strains=strains,
            volumes=volumes,
        )

        prediction = self._compute(
            batched_graph,
            compute_force="f" in task,
            compute_stress="s" in task,
            compute_magmom="m" in task,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )

        prediction["e"] += comp_energy

        if return_site_energies and self.composition_model is not None:
            site_energy_shifts = self.composition_model.get_site_energies(graphs)
            prediction["site_energies"] = [
                (i + j)
                for i, j in zip(
                    prediction["site_energies"], site_energy_shifts, strict=True
                )
            ]

        return prediction

    def _compute(
        self,
        g: BatchedGraph,
        *,
        compute_force: bool = False,
        compute_stress: bool = False,
        compute_magmom: bool = False,
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict:
        """Get Energy, Force, Stress, Magmom associated with input graphs
        force = - d(Energy)/d(atom_positions)
        stress = 1/V * d(Energy)/d(strain).

        Args:
            g (BatchedGraph): batched graph
            compute_force (bool): whether to compute force.
                Default = False
            compute_stress (bool): whether to compute stress.
                Default = False
            compute_magmom (bool): whether to compute magmom.
                Default = False
            return_site_energies (bool): whether to return per-site energies,
                only available if self.mlp_first == True
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False

        Returns:
            prediction (dict): containing the fields:
                e (Tensor) : energy of structures [batch_size, 1]
                f (Tensor) : force on atoms [num_batch_atoms, 3]
                s (Tensor) : stress of structure [3 * batch_size, 3]
                m (Tensor) : magnetic moments of sites [num_batch_atoms, 3]
        """

        prediction = {}
        atoms_per_graph = paddle.bincount(x=g.atom_owners)
        prediction["atoms_per_graph"] = atoms_per_graph
        atom_feas = self.atom_embedding(g.atomic_numbers - 1)
        bond_feas = self.bond_embedding(g.bond_bases_ag)
        bond_weights_ag = self.bond_weights_ag(g.bond_bases_ag)
        bond_weights_bg = self.bond_weights_bg(g.bond_bases_bg)
        if len(g.angle_bases) != 0:
            angle_feas = self.angle_embedding(g.angle_bases)
        for idx, (atom_layer, bond_layer, angle_layer) in enumerate(
            zip(
                self.atom_conv_layers[:-1],
                self.bond_conv_layers,
                self.angle_layers,
                strict=False,
            )
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
                if return_atom_feas:
                    prediction["atom_fea"] = paddle.split(
                        x=atom_feas, num_or_sections=atoms_per_graph.tolist()
                    )
                if compute_magmom:
                    magmom = paddle.abs(x=self.site_wise(atom_feas))
                    prediction["m"] = list(
                        paddle.split(
                            x=magmom.reshape([-1]),
                            num_or_sections=atoms_per_graph.tolist(),
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
        if self.mlp_first:
            energies = self.mlp(atom_feas)
            energy = self.pooling(energies, g.atom_owners).reshape([-1])
            if return_site_energies:
                prediction["site_energies"] = paddle.split(
                    x=energies.squeeze(axis=1), num_or_sections=atoms_per_graph.tolist()
                )
            if return_crystal_feas:
                prediction["crystal_fea"] = self.pooling(atom_feas, g.atom_owners)
        else:
            crystal_feas = self.pooling(atom_feas, g.atom_owners)
            energy = self.mlp(crystal_feas).reshape([-1]) * atoms_per_graph
            if return_crystal_feas:
                prediction["crystal_fea"] = crystal_feas

        if compute_force:
            force = paddle.grad(
                outputs=energy.sum(),
                inputs=g.atom_positions,
                create_graph=self.training,
                retain_graph=self.training,
            )
            if isinstance(g.atom_positions, paddle.Tensor):
                force = force[0]
            # prediction["f"] = [(-1 * force_dim) for force_dim in force]
            prediction["f"] = -1 * force
        if compute_stress:
            stress = paddle.grad(
                outputs=energy.sum(),
                inputs=g.strains,
                create_graph=self.training,
                retain_graph=self.training,
            )
            if isinstance(g.strains, paddle.Tensor):
                stress = stress[0]
            scale = 1 / g.volumes * 160.21766208
            stress = stress * scale[:, None, None]
            # stress = [(i * j) for i, j in zip(stress, scale, strict=False)]
            prediction["s"] = stress

        if self.is_intensive:
            energy /= atoms_per_graph.cast("float32")
        prediction["e"] = energy
        return prediction

    def predict_graph(
        self,
        graph,
        *,
        task: PredTask = "efsm",
        return_site_energies: bool = False,
        return_atom_feas: bool = False,
        return_crystal_feas: bool = False,
    ) -> dict[str, paddle.Tensor] | list[dict[str, paddle.Tensor]]:
        """Predict from CrustalGraph.

        Args:
            graph (CrystalGraph | Sequence[CrystalGraph]): CrystalGraph(s) to predict.
            task (str): can be 'e' 'ef', 'em', 'efs', 'efsm'
                Default = "efsm"
            return_site_energies (bool): whether to return per-site energies.
                Default = False
            return_atom_feas (bool): whether to return atom features.
                Default = False
            return_crystal_feas (bool): whether to return crystal features.
                Default = False
            batch_size (int): batch_size for predict structures.
                Default = 16

        Returns:
            prediction (dict): dict or list of dict containing the fields:
                e (Tensor) : energy of structures float in eV/atom
                f (Tensor) : force on atoms [num_atoms, 3] in eV/A
                s (Tensor) : stress of structure [3, 3] in GPa
                m (Tensor) : magnetic moments of sites [num_atoms, 3] in Bohr
                    magneton mu_B
        """

        self.eval()
        if not graph.is_tensor():
            graph.tensor()
        # predictions: list[dict[str, paddle.Tensor]] = [{} for _ in range(len(graph))]
        prediction = self.forward(
            {"graph": graph},
            task=task,
            return_site_energies=return_site_energies,
            return_atom_feas=return_atom_feas,
            return_crystal_feas=return_crystal_feas,
        )
        for key in prediction.keys():
            if isinstance(prediction[key], list):
                prediction[key] = [
                    prediction[key][i].numpy() for i in range(len(prediction[key]))
                ]
            else:
                prediction[key] = prediction[key].numpy()
            if key == "s" and len(prediction["s"].shape) == 3:
                prediction[key] = prediction[key][0]
            if key == "m" and isinstance(prediction[key], list):
                prediction[key] = prediction[key][0]
            if key == "e" and isinstance(prediction[key], np.ndarray):
                prediction[key] = prediction[key][0]
        return prediction


@dataclass
class BatchedGraph:
    """Batched crystal graph for parallel computing.

    Attributes:
        atomic_numbers (Tensor): atomic numbers vector
            [num_batch_atoms]
        bond_bases_ag (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_ag, num_radial]
        bond_bases_bg (Tensor): bond bases vector for atom_graph
            [num_batch_bonds_bg, num_radial]
        angle_bases (Tensor): angle bases vector
            [num_batch_angles, num_angular]
        batched_atom_graph (Tensor) : batched atom graph adjacency list
            [num_batch_bonds, 2]
        batched_bond_graph (Tensor) : bond graph adjacency list
            [num_batch_angles, 3]
        atom_owners (Tensor): graph indices for each atom, used aggregate batched
            graph back to single graph
            [num_batch_atoms]
        directed2undirected (Tensor): the utility tensor used to quickly
            map directed edges to undirected edges in graph
            [num_directed]
        atom_positions (list[Tensor]): cartesian coordinates of the atoms
            from structures
            [[num_atoms_1, 3], [num_atoms_2, 3], ...]
        strains (list[Tensor]): a list of strains that's initialized to be zeros
            [[3, 3], [3, 3], ...]
        volumes (Tensor): the volume of each structure in the batch
            [batch_size]
    """

    atomic_numbers: paddle.Tensor
    bond_bases_ag: paddle.Tensor
    bond_bases_bg: paddle.Tensor
    angle_bases: paddle.Tensor
    batched_atom_graph: paddle.Tensor
    batched_bond_graph: paddle.Tensor
    atom_owners: paddle.Tensor
    directed2undirected: paddle.Tensor
    atom_positions: Sequence[paddle.Tensor]
    strains: Sequence[paddle.Tensor]
    volumes: Sequence[paddle.Tensor] | paddle.Tensor
