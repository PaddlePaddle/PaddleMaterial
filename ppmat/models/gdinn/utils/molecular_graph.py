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
Molecular graph construction utilities.

This module provides functions to convert RDKit molecules to graph representations
compatible with PGL.
"""

from functools import partial
from typing import Callable
from typing import Optional

import numpy as np
import paddle
import pgl
from rdkit import Chem


def construct_bigraph_from_mol(
    mol: Chem.rdchem.Mol, add_self_loop: bool = False
) -> tuple:
    """Construct bidirectional graph edges from a molecule.

    Args:
        mol: RDKit molecule object
        add_self_loop: Whether to add self loops

    Returns:
        Tuple of (src, dst) edge arrays
    """
    num_atoms = mol.GetNumAtoms()

    # Collect bond information
    src_list = []
    dst_list = []

    for bond in mol.GetBonds():
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()
        # Add both directions for bidirectional graph
        src_list.append(src)
        dst_list.append(dst)
        src_list.append(dst)
        dst_list.append(src)

    # Add self-loops if needed
    if add_self_loop:
        for i in range(num_atoms):
            src_list.append(i)
            dst_list.append(i)

    # Handle edge case of single atom
    if len(src_list) == 0:
        src_list = [0]
        dst_list = [0]

    return (
        paddle.to_tensor(np.array(src_list, dtype=np.int64)),
        paddle.to_tensor(np.array(dst_list, dtype=np.int64)),
    )


def mol_to_graph(
    mol: "Chem.Mol",
    graph_constructor: Callable,
    node_featurizer: Optional[Callable] = None,
    edge_featurizer: Optional[Callable] = None,
    canonical_atom_order: bool = True,
    explicit_hydrogens: bool = False,
    num_virtual_nodes: int = 0,
) -> pgl.Graph:
    """Convert RDKit molecule to pgl.Graph using a custom graph constructor.

    This is a generic function that can create different types of molecular graphs
    by using different graph construction functions.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    graph_constructor : callable
        Function that takes a molecule and returns (src, dst) edge arrays.
        This determines the graph structure (e.g., bidirectional, directed, etc.)
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule. Default to None.
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.
    explicit_hydrogens : bool
        Whether to explicitly represent hydrogens as nodes in the graph. If True,
        it will call rdkit.Chem.AddHs(mol). Default to False.
    num_virtual_nodes : int
        The number of virtual nodes to add. The virtual nodes will be connected to
        all real nodes with virtual edges. If the returned graph has any node/edge
        feature, an additional column of binary values will be used for each feature
        to indicate the identity of virtual node/edges. The features of the virtual
        nodes/edges will be zero vectors except for the additional column. Default to 0.

    Returns
    -------
    pgl.Graph
        pgl.Graph for the molecule
    """
    if mol is None:
        raise ValueError("Input molecule is None")

    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    # Apply canonical ordering if requested
    if canonical_atom_order:
        mol = Chem.RenumberAtoms(mol, list(range(mol.GetNumAtoms())))

    # Construct graph edges using the provided constructor
    src, dst = graph_constructor(mol)

    # Generate node features if featurizer is provided
    node_feat = {}
    if node_featurizer is not None:
        node_feat = node_featurizer(mol)

    # Generate edge features if featurizer is provided
    edge_feat = {}
    if edge_featurizer is not None:
        edge_data = edge_featurizer(mol)
        edge_feat = {"e": edge_data["e"]}

    # Handle virtual nodes
    if num_virtual_nodes > 0:
        num_real_nodes = mol.GetNumAtoms()
        real_nodes = list(range(num_real_nodes))

        # Add virtual node indices to node features
        for feat_name in node_feat.keys():
            real_feat = node_feat[feat_name]
            feat_dim = real_feat.shape[1]
            virtual_feat = paddle.zeros(
                (num_virtual_nodes, feat_dim), dtype=real_feat.dtype
            )
            # Add indicator column for virtual nodes
            real_indicator = paddle.zeros((num_real_nodes, 1), dtype=real_feat.dtype)
            virtual_indicator = paddle.ones(
                (num_virtual_nodes, 1), dtype=real_feat.dtype
            )
            real_feat = paddle.concat([real_feat, real_indicator], axis=1)
            virtual_feat = paddle.concat([virtual_feat, virtual_indicator], axis=1)
            node_feat[feat_name] = paddle.concat([real_feat, virtual_feat], axis=0)

        # Add virtual edges
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)

        # Concatenate edges
        src = paddle.concat(
            [src, paddle.to_tensor(np.array(virtual_src, dtype=np.int64))]
        )
        dst = paddle.concat(
            [dst, paddle.to_tensor(np.array(virtual_dst, dtype=np.int64))]
        )

        # Add indicator column to edge features
        for feat_name in edge_feat.keys():
            real_feat = edge_feat[feat_name]
            num_real_edges = real_feat.shape[0]
            feat_dim = real_feat.shape[1]
            num_virtual_edges = len(virtual_src)
            real_indicator = paddle.zeros((num_real_edges, 1), dtype=real_feat.dtype)
            virtual_indicator = paddle.ones(
                (num_virtual_edges, 1), dtype=real_feat.dtype
            )
            real_feat = paddle.concat([real_feat, real_indicator], axis=1)
            virtual_feat = paddle.zeros(
                (num_virtual_edges, feat_dim + 1), dtype=real_feat.dtype
            )
            virtual_feat[:, :-1] = 0
            virtual_feat[:, -1] = 1
            edge_feat[feat_name] = paddle.concat([real_feat, virtual_feat], axis=0)

    # Create pgl.Graph
    num_nodes = mol.GetNumAtoms() + num_virtual_nodes
    # Convert edges to list of tuples for pgl.Graph
    edges = list(zip(src.tolist(), dst.tolist()))
    graph = pgl.Graph(
        num_nodes=num_nodes, edges=edges, node_feat=node_feat, edge_feat=edge_feat
    )

    return graph


def mol_to_bigraph(
    mol: "Chem.Mol",
    add_self_loop: bool = False,
    node_featurizer: Optional[Callable] = None,
    edge_featurizer: Optional[Callable] = None,
    canonical_atom_order: bool = True,
    explicit_hydrogens: bool = False,
    num_virtual_nodes: int = 0,
) -> pgl.Graph:
    """Convert RDKit molecule to bidirectional pgl.Graph.

    This function replaces DGL's mol_to_bigraph for PaddlePaddle.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.
    explicit_hydrogens : bool
        Whether to explicitly represent hydrogens as nodes in the graph. If True,
        it will call rdkit.Chem.AddHs(mol). Default to False.
    num_virtual_nodes : int
        The number of virtual nodes to add. The virtual nodes will be connected to
        all real nodes with virtual edges. If the returned graph has any node/edge
        feature, an additional column of binary values will be used for each feature
        to indicate the identity of virtual node/edges. The features of the virtual
        nodes/edges will be zero vectors except for the additional column. Default to 0.

    Returns
    -------
    pgl.Graph or None
        Bi-directed pgl.Graph for the molecule if :attr:`mol` is valid and None otherwise.
    """
    return mol_to_graph(
        mol,
        partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
        node_featurizer,
        edge_featurizer,
        canonical_atom_order,
        explicit_hydrogens,
        num_virtual_nodes,
    )


def smiles_to_bigraph(
    smiles: str,
    add_self_loop: bool = False,
    node_featurizer: Optional[Callable] = None,
    edge_featurizer: Optional[Callable] = None,
    canonical_atom_order: bool = True,
    explicit_hydrogens: bool = False,
    num_virtual_nodes: int = 0,
) -> pgl.Graph:
    """Convert a SMILES into a bi-directed pgl.Graph and featurize for it.

    Parameters
    ----------
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    node_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for nodes like atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    edge_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for edges like bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    canonical_atom_order : bool
        Whether to use a canonical order of atoms returned by RDKit. Setting it
        to true might change the order of atoms in the graph constructed. Default
        to True.
    explicit_hydrogens : bool
        Whether to explicitly represent hydrogens as nodes in the graph. If True,
        it will call rdkit.Chem.AddHs(mol). Default to False.
    num_virtual_nodes : int
        The number of virtual nodes to add. The virtual nodes will be connected to
        all real nodes with virtual edges. If the returned graph has any node/edge
        feature, an additional column of binary values will be used for each feature
        to indicate the identity of virtual node/edges. The features of the virtual
        nodes/edges will be zero vectors except for the additional column. Default to 0.

    Returns
    -------
    pgl.Graph or None
        Bi-directed pgl.Graph for the molecule if :attr:`smiles` is valid and None otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    return mol_to_bigraph(
        mol,
        add_self_loop,
        node_featurizer,
        edge_featurizer,
        canonical_atom_order,
        explicit_hydrogens,
        num_virtual_nodes,
    )
