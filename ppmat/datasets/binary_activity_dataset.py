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
Binary activity coefficient dataset for GDI-NN.

GDI-NN Format (input_file):
    Columns: job_id, solv1, solv2, solv1_x, solv2_x, solv1_gamma, solv2_gamma,
             warnings, solv1_smiles, solv2_smiles, solv1_name, solv2_name, tpsa_binary_avg

    Example row:
        0,solvent_587,solvent_604,0.1,0.9,0.47175935,0.00025148,,CN,CC(=O)CC(C)C,METHYL AMINE,METHYL ISOBUTYL KETONE,2

Solvent List Format:
    Columns: solvent_name, solvent_id, smiles_can

Reference: GDI-NN (https://git.rwth-aachen.de/avt-svt/public/GDI-NN)
"""

import csv
import os
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import pgl
from paddle.io import Dataset
from rdkit.Chem import rdMolDescriptors

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.models.gdinn.utils.atom_feat_encoding import CanonicalAtomFeaturizer
from ppmat.models.gdinn.utils.molecular_graph import mol_to_bigraph


class BinaryActivityDataset(Dataset):
    """Binary activity coefficient dataset in GDI-NN format.

    This dataset loads binary solvent mixture data from a CSV file and converts
    molecules to graph representations. Each sample contains two molecular graphs,
    composition (x1), activity coefficients (gamma1, gamma2), and hydrogen bond features.

    GDI-NN CSV Format (input_file):
        job_id,solv1,solv2,solv1_x,solv2_x,solv1_gamma,solv2_gamma,warnings,solv1_smiles,solv2_smiles,solv1_name,solv2_name,tpsa_binary_avg
        0,solvent_587,solvent_604,0.1,0.9,0.47175935,0.00025148,,CN,CC(=O)CC(C)C,METHYL AMINE,METHYL ISOBUTYL KETONE,2

    Solvent List Format:
        solvent_name, solvent_id, smiles_can
        "1,1,1-TRICHLOROETHANE", solvent_1, CC(Cl)(Cl)Cl

    Args:
        data_path: Path to CSV file containing binary mixture data (GDI-NN format)
        solvent_list_path: Path to file containing list of solvents
            Format: solvent_name, solvent_id, smiles_can
        graph_converter: Function to convert molecules to graphs like mol_to_bigraph (default:None)
        add_self_loop: Whether to add self-loops to graphs (default: True)
        preload_graphs: Whether to preload all graphs into memory (default: False)
    """

    def __init__(
        self,
        data_path: str,
        solvent_list_path: Optional[str] = None,
        graph_converter: Optional[Callable] = None,
        add_self_loop: bool = True,
        preload_graphs: bool = False,
    ):
        """Initialize Binary Activity Dataset.

        Args:
            data_path: Path to CSV file containing binary mixture data (GDI-NN format)
            solvent_list_path: Path to file containing list of solvents
                Format: solvent_name, solvent_id, smiles_can
            graph_converter: Function to convert molecules to graphs like mol_to_bigraph (default: None)
            add_self_loop: Whether to add self-loops to graphs (default: True)
            preload_graphs: Whether to preload all graphs into memory (default: False)
        """
        super().__init__()
        self.data_path = data_path
        self.solvent_list_path = solvent_list_path
        self.add_self_loop = add_self_loop
        self.preload_graphs = preload_graphs

        # Set default graph converter with CanonicalAtomFeaturizer
        # This matches the GDI-NN implementation
        if graph_converter is None:
            self.graph_converter = lambda mol: mol_to_bigraph(
                mol,
                add_self_loop=add_self_loop,
                node_featurizer=CanonicalAtomFeaturizer(),
                edge_featurizer=None,
                canonical_atom_order=False,
                explicit_hydrogens=False,
                num_virtual_nodes=0,
            )
        else:
            self.graph_converter = graph_converter

        # Initialize BuildMolecule factory for processing SMILES
        self.build_molecule = BuildMolecule(
            format="smiles",
            sanitize=True,
            add_hs=False,
            remove_hs=False,
            kekulize=False,
        )

        # Load solvent list for caching molecular graphs
        self.solvent_info = {}  # solvent_id -> {name, smiles}
        self.solvent_smiles = {}  # solvent_id -> smiles
        if solvent_list_path and os.path.exists(solvent_list_path):
            self._load_solvent_list(solvent_list_path)
        elif solvent_list_path:
            print(f"Warning: Solvent list not found: {solvent_list_path}")

        # Load data
        self.data = self._load_csv(data_path)

        # Solvent data cache: solvent_id -> [graph, hba, hbd, intra_hb]
        self.solvent_data = {}

        # Preload graphs if requested
        self.graph_cache = {}
        if preload_graphs:
            self._preload_graphs()
            self._generate_all_solvent_data()

    def _load_csv(self, data_path: str) -> List[Dict]:
        """Load CSV data file.

        Args:
            data_path: Path to CSV file

        Returns:
            List of dictionaries, each representing a row
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        data = []
        with open(data_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        return data

    def _load_solvent_list(self, solvent_list_path: str):
        """Load solvent list for caching molecular graphs.

        GDI-NN format: solvent_name, solvent_id, smiles_can

        Args:
            solvent_list_path: Path to solvent list file
        """
        try:
            with open(solvent_list_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    solvent_id = row.get("solvent_id", "")
                    solvent_name = row.get("solvent_name", "")
                    smiles = row.get("smiles_can", "").strip()
                    if solvent_id and smiles:
                        self.solvent_info[solvent_id] = {
                            "name": solvent_name,
                            "smiles": smiles,
                        }
                        self.solvent_smiles[solvent_id] = smiles
            print(f"Loaded {len(self.solvent_info)} solvents from solvent list")
        except Exception as e:
            print(f"Warning: Failed to load solvent list: {e}")

    def _preload_graphs(self):
        """Preload all molecular graphs into memory."""
        print("Preloading molecular graphs...")

        # Use solvent_smiles dictionary to preload
        if self.solvent_smiles:
            for solvent_id, smiles in self.solvent_smiles.items():
                if smiles and smiles not in self.graph_cache:
                    try:
                        mol = self.build_molecule(smiles)
                        if mol is not None:
                            self.graph_cache[smiles] = self.graph_converter(mol)
                            self.graph_cache[solvent_id] = self.graph_cache[smiles]
                    except Exception as e:
                        print(
                            f"Warning: Failed to convert SMILES to graph: {smiles}, {e}"
                        )
        else:
            # Fallback: collect all unique SMILES from data
            all_smiles = set()
            for row in self.data:
                smiles1 = row.get("solv1_smiles", "")
                smiles2 = row.get("solv2_smiles", "")
                if smiles1:
                    all_smiles.add(smiles1)
                if smiles2:
                    all_smiles.add(smiles2)

            for smiles in all_smiles:
                if smiles not in self.graph_cache:
                    try:
                        mol = self.build_molecule(smiles)
                        if mol is not None:
                            self.graph_cache[smiles] = self.graph_converter(mol)
                    except Exception as e:
                        print(
                            f"Warning: Failed to convert SMILES to graph: {smiles}, {e}"
                        )

        print(f"Preloaded {len(self.graph_cache)} molecular graphs")

    def _generate_all_solvent_data(self):
        """Generate all solvent data including graph, HBA, HBD, and intra_hb.

        This matches the original GDI-NN implementation in generate_dataset_for_training.py.
        Each solvent_id maps to [graph, hba, hbd, intra_hb].
        """
        for solvent_id, smiles in self.solvent_smiles.items():
            if solvent_id in self.solvent_data:
                continue

            try:
                mol = self.build_molecule(smiles)
                if mol is None:
                    continue

                # Get cached graph or create new one
                if smiles in self.graph_cache:
                    graph = self.graph_cache[smiles]
                else:
                    graph = self.graph_converter(mol)
                    self.graph_cache[smiles] = graph

                # Compute hydrogen bond features
                hba = rdMolDescriptors.CalcNumHBA(mol)
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                intra_hb = min(hba, hbd)

                # Store: [graph, hba, hbd, intra_hb] - matches GDI-NN format
                self.solvent_data[solvent_id] = [graph, hba, hbd, intra_hb]

            except Exception as e:
                print(f"Warning: Failed to generate data for solvent {solvent_id}: {e}")

        print(f"Generated data for {len(self.solvent_data)} solvents")

    def _get_molecular_graph(self, smiles: str) -> pgl.Graph:
        """Get molecular graph for a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            pgl.Graph object
        """
        if self.preload_graphs and smiles in self.graph_cache:
            return self.graph_cache[smiles]

        try:
            mol = self.build_molecule(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            graph = self.graph_converter(mol)

            if self.preload_graphs:
                self.graph_cache[smiles] = graph

            return graph
        except Exception as e:
            raise ValueError(f"Failed to convert SMILES to graph: {smiles}, {e}")

    def _get_smiles(self, solvent_id: str) -> str:
        """Get SMILES for a solvent ID.

        Args:
            solvent_id: Solvent ID (e.g., 'solvent_587')

        Returns:
            SMILES string
        """
        if solvent_id in self.solvent_smiles:
            return self.solvent_smiles[solvent_id]

        for row in self.data:
            if row.get("solv1") == solvent_id:
                return row.get("solv1_smiles", "")
            elif row.get("solv2") == solvent_id:
                return row.get("solv2_smiles", "")

        raise ValueError(f"Cannot find SMILES for solvent ID: {solvent_id}")

    def _parse_value(self, value: str) -> float:
        """Parse string value to float, handling special cases.

        Args:
            value: String value

        Returns:
            Float value
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            value_lower = str(value).lower()
            if value_lower == "inf":
                return float("inf")
            elif value_lower == "-inf":
                return float("-inf")
            elif value_lower == "nan":
                return float("nan")
            else:
                return 0.0

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - g1: Molecular graph for solvent 1
                - g2: Molecular graph for solvent 2
                - x1: Composition of solvent 1 (mole fraction, solv1_x)
                - x2: Composition of solvent 2 (mole fraction, solv2_x)
                - gamma1: ln(activity coefficient) for solvent 1
                - gamma2: ln(activity coefficient) for solvent 2
                - intra_hb1: Intra-molecular hydrogen bonding capacity for solvent 1
                - intra_hb2: Intra-molecular hydrogen bonding capacity for solvent 2
                - inter_hb: Inter-molecular hydrogen bonding capacity
                - solv1_id: Solvent 1 ID
                - solv2_id: Solvent 2 ID
                - solv1_x: Composition of solvent 1 (same as x1, for GDI-NN compatibility)
        """
        row = self.data[idx]

        # Get solvent IDs
        solv1_id = row["solv1"]
        solv2_id = row["solv2"]

        # Get solvent data (from cache or compute on-the-fly)
        solv1 = self._get_solvent_data(solv1_id)
        solv2 = self._get_solvent_data(solv2_id)

        # Parse composition values
        x1 = self._parse_value(row["solv1_x"])
        x2 = self._parse_value(row["solv2_x"])

        gamma1 = self._parse_value(row["solv1_gamma"])
        gamma2 = self._parse_value(row["solv2_gamma"])

        # Build sample dictionary (consistent with GDI-NN format)
        # solvent_data format: [graph, hba, hbd, intra_hb]
        sample = {
            "g1": solv1[0],  # graph
            "g2": solv2[0],  # graph
            "x1": x1,
            "x2": x2,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "solv1_id": solv1_id,
            "solv2_id": solv2_id,
            "solv1_x": x1,  # GDI-NN uses 'solv1_x' key
            # Hydrogen bond features (computed from cached HBA/HBD values)
            # intra_hb = min(HBA, HBD)
            "intra_hb1": solv1[3],  # min(hba, hbd)
            "intra_hb2": solv2[3],  # min(hba, hbd)
            # inter_hb = min(HBA1, HBD2) + min(HBD1, HBA2)
            "inter_hb": min(solv1[1], solv2[2]) + min(solv1[2], solv2[1]),
        }

        return sample

    def _get_solvent_data(self, solvent_id: str) -> List:
        """Get solvent data (graph, hba, hbd, intra_hb) for a solvent ID.

        This matches the original GDI-NN implementation where solvent_data is
        cached with format: [graph, hba, hbd, intra_hb].

        Args:
            solvent_id: Solvent ID (e.g., 'solvent_587')

        Returns:
            List containing [graph, hba, hbd, intra_hb]
        """
        if solvent_id in self.solvent_data:
            return self.solvent_data[solvent_id]

        smiles = self._get_smiles(solvent_id)
        mol = self.build_molecule(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for solvent {solvent_id}: {smiles}")

        graph = self._get_molecular_graph(smiles)

        # Compute hydrogen bond features
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        intra_hb = min(hba, hbd)

        # Cache the result: [graph, hba, hbd, intra_hb]
        self.solvent_data[solvent_id] = [graph, hba, hbd, intra_hb]

        return self.solvent_data[solvent_id]

    def search_chemical(self, chemical_name: str) -> List:
        """Search for a chemical by name.

        Args:
            chemical_name: Name of the chemical to search for

        Returns:
            List containing solvent_id and indices of matching rows
        """
        for solvent_id, info in self.solvent_info.items():
            if chemical_name.lower() == info["name"].lower():
                print(f"{solvent_id}, {info['name']}, {info['smiles']}")
                indices = [
                    i
                    for i, row in enumerate(self.data)
                    if row["solv1"] == solvent_id or row["solv2"] == solvent_id
                ]
                return [solvent_id, indices]
        return [None, []]

    def search_chemical_pair(self, chemical_list: List[str]) -> List:
        """Search for a pair of chemicals.

        Args:
            chemical_list: List of two chemical names

        Returns:
            List containing solvent IDs and indices of matching rows
        """
        solv1_match = self.search_chemical(chemical_list[0])[0]
        solv2_match = self.search_chemical(chemical_list[1])[0]

        if solv1_match is None or solv2_match is None:
            return [[None, None], []]

        indices = [
            i
            for i, row in enumerate(self.data)
            if (row["solv1"] == solv1_match and row["solv2"] == solv2_match)
            or (row["solv1"] == solv2_match and row["solv2"] == solv1_match)
        ]

        return [[solv1_match, solv2_match], indices]

    @staticmethod
    def generate_solvsys(batch_size: int) -> pgl.Graph:
        """Generate an empty solvent system graph for global interaction.

        This creates a bipartite graph connecting solvent 1 and solvent 2 representations
        for each batch sample, matching the original GDI-NN architecture.

        The graph has:
        - 2 * batch_size nodes (two solvent nodes per batch)
        - Bidirectional edges between solvent pairs
        - Self-loops on each node

        Args:
            batch_size: Number of samples in the batch

        Returns:
            pgl.Graph with the solvent system topology
        """
        import paddle

        n_solv = 2
        num_nodes = n_solv * batch_size

        # Create edges matching original DGL order:
        #   src = arange(batch_size)           -> [0, 1, ..., batch-1]
        #   dst = arange(batch_size, 2*batch)  -> [batch, batch+1, ..., 2*batch-1]
        #   add_edges(cat(src, dst), cat(dst, src))  -> all src->dst then all dst->src
        #   add_edges(arange(2*batch), arange(2*batch))  -> self-loops
        #
        # Edge order matters because hb_features are indexed by position:
        #   [0..batch-1]: inter_hb (solv1->solv2)
        #   [batch..2*batch-1]: inter_hb (solv2->solv1)
        #   [2*batch..3*batch-1]: intra_hb1 (self-loops on solv1)
        #   [3*batch..4*batch-1]: intra_hb2 (self-loops on solv2)
        src_range = paddle.arange(batch_size, dtype="int64")
        dst_range = paddle.arange(batch_size, num_nodes, dtype="int64")
        all_range = paddle.arange(num_nodes, dtype="int64")

        # Bidirectional edges: cat(src, dst) -> cat(dst, src)
        edge_src = paddle.concat([paddle.concat([src_range, dst_range]), all_range])
        edge_dst = paddle.concat([paddle.concat([dst_range, src_range]), all_range])

        # Convert to list of tuples for pgl.Graph
        edges = list(zip(edge_src.tolist(), edge_dst.tolist()))

        graph = pgl.Graph(
            num_nodes=num_nodes,
            edges=edges,
            node_feat={"h": paddle.zeros([num_nodes, 1])},  # Dummy features
            edge_feat={"e": paddle.zeros([len(edges), 1])},  # Dummy edge features
        )

        return graph
