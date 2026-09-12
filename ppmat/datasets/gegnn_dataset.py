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

from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import paddle
import paddle.distributed as dist
import pandas as pd
import pgl
from paddle.io import Dataset
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.misc import is_equal

__all__ = ["BinaryActivityDataset"]

_ATOM_TYPES = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
]
_DEGREES = list(range(11))
_VALENCES = list(range(7))
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_NUM_H = [0, 1, 2, 3, 4]
_MOLECULAR_GRAPH_VOCAB = {
    "atom": {
        "token_to_id": {atom: i for i, atom in enumerate(_ATOM_TYPES)},
        "num_embeddings": len(_ATOM_TYPES),
    },
    "bond": {
        "token_to_id": {
            "NO_BOND": 0,
            "SINGLE": 1,
            "DOUBLE": 2,
            "TRIPLE": 3,
            "AROMATIC": 4,
        },
        "num_embeddings": 5,
    },
}
_MOLECULAR_GRAPH_CFG = {
    "__class_name__": "MolecularGraphConverter",
    "__init_params__": {
        "vocab": _MOLECULAR_GRAPH_VOCAB,
        "remove_h": False,
        "add_self_loops": True,
        "edge_mode": "bidirectional",
    },
}


def _one_hot(value, values):
    return [int(value == item) for item in values]


def _canonical_atom_feats(atom):
    """Match the upstream DGL-LifeSci ``CanonicalAtomFeaturizer`` (74 dims)."""
    return np.asarray(
        _one_hot(atom.GetSymbol(), _ATOM_TYPES)
        + _one_hot(atom.GetDegree(), _DEGREES)
        + _one_hot(atom.GetImplicitValence(), _VALENCES)
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic())]
        + _one_hot(atom.GetTotalNumHs(), _NUM_H),
        dtype="float32",
    )


def build_molecular_graph(molecule, converter):
    """Build the molecular graph and GE-GNN atom features."""
    graph = converter(molecule)
    if graph is not None:
        graph.node_feat["h"] = np.stack(
            [
                _canonical_atom_feats(molecule.GetAtomWithIdx(index))
                for index in range(molecule.GetNumAtoms())
            ]
        )
    return graph


class BinaryActivityDataset(Dataset):
    """Binary-mixture activity-coefficient dataset.

    Molecular graphs are built once and cached as indexed pickle files.

    Args:
        path: Path to the binary activity CSV. If it does not exist, the
            dataset will be downloaded.
        solvent_list_path: Path to the solvent metadata CSV. Defaults to a
            ``solvent_list.csv`` sibling of ``path``.
        build_graph_cfg: Config for building molecular graphs. Defaults to
            ``_MOLECULAR_GRAPH_CFG``.
        cache_path: Directory for cached molecular graphs. Defaults to a
            ``_cache`` sibling of the data directory.
        overwrite: Whether to rebuild the cache even if it exists.
    """

    name = "binary_activity"
    md5 = ""
    url = (
        "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/"
        "thermodynamic_data_of_binary_mixtures/"
    )
    data_file = "output_binary_with_inf_all.csv"
    solvent_file = "solvent_list.csv"
    _REQUIRED_COLUMNS = {
        "solv1",
        "solv2",
        "solv1_x",
        "solv1_gamma",
        "solv2_gamma",
    }

    def __init__(
        self,
        path: str = "./data/binary_activity/output_binary_with_inf_all.csv",
        solvent_list_path: Optional[str] = None,
        build_graph_cfg: Optional[Dict] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__()

        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            path = download.get_path_from_url(
                self.url + osp.basename(path),
                osp.dirname(path),
                decompress=False,
            )
        if solvent_list_path is None:
            solvent_list_path = osp.join(osp.dirname(path), self.solvent_file)
        if not osp.exists(solvent_list_path):
            solvent_list_path = download.get_path_from_url(
                self.url + self.solvent_file,
                osp.dirname(solvent_list_path),
                decompress=False,
            )

        self.path = path
        self.solvent_list_path = solvent_list_path
        self.build_graph_cfg = build_graph_cfg or _MOLECULAR_GRAPH_CFG
        self.build_molecule = BuildMolecule(format="smiles")

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache",
                osp.splitext(osp.basename(path))[0],
            )
        logger.info(f"Cache path: {self.cache_path}")

        self.cache_exists = osp.exists(self.cache_path)
        self.overwrite = overwrite

        self.dataset = self.read_data(path)
        self.num_samples = len(self.dataset)
        logger.info(f"Load {self.num_samples} binary-mixture samples from {path}")

        self.solvent_smiles = self.read_solvent_smiles(solvent_list_path)
        self.solvent_ids = list(self.solvent_smiles)
        self.solvent_index = {
            solvent_id: index
            for index, solvent_id in enumerate(self.solvent_ids)
        }
        self.num_solvents = len(self.solvent_ids)

        graph_cache_path = osp.join(self.cache_path, "graphs")
        config_cache_path = osp.join(self.cache_path, "build_graph_cfg.pkl")

        cache_config = {
            "build_graph_cfg": self.build_graph_cfg,
            "solvent_ids": self.solvent_ids,
            "solvent_smiles": [
                self.solvent_smiles[solvent_id] for solvent_id in self.solvent_ids
            ],
        }

        need_rebuild = overwrite or not self.cache_exists
        if self.cache_exists and not overwrite:
            try:
                graph_paths = [
                    osp.join(graph_cache_path, f"{index:010d}.pkl")
                    for index in range(self.num_solvents)
                ]
                if not all(osp.exists(p) for p in graph_paths):
                    raise FileNotFoundError("The cached graph files are incomplete.")
                cached_config = self.load_from_cache(config_cache_path)
                if is_equal(cached_config, cache_config):
                    logger.info(
                        "The cached graph configuration matches the current "
                        "settings. Reusing previously generated molecular graphs."
                    )
                else:
                    logger.warning(
                        "build_graph_cfg or solvent metadata differs from cache. "
                        "Will rebuild the graphs."
                    )
                    need_rebuild = True
            except Exception as error:
                logger.warning(error)
                logger.warning(
                    "Failed to load graph cache metadata. Will rebuild the graphs."
                )
                need_rebuild = True

        if need_rebuild:
            if not dist.is_initialized() or dist.get_rank() == 0:
                os.makedirs(graph_cache_path, exist_ok=True)
                self.save_to_cache(config_cache_path, cache_config)
                converter = build_graph_converter(self.build_graph_cfg)
                for index, solvent_id in enumerate(self.solvent_ids):
                    molecule = self.build_molecule(self.solvent_smiles[solvent_id])
                    if molecule is None:
                        raise ValueError(
                            f"Invalid SMILES for solvent {solvent_id}: "
                            f"{self.solvent_smiles[solvent_id]}"
                        )
                    self.save_to_cache(
                        osp.join(graph_cache_path, f"{index:010d}.pkl"),
                        self.build_solvent(molecule, converter),
                    )
                logger.info(f"Save {self.num_solvents} graphs to {graph_cache_path}")
            if dist.is_initialized():
                dist.barrier()

        self.graphs = [
            osp.join(graph_cache_path, f"{index:010d}.pkl")
            for index in range(self.num_solvents)
        ]

    def read_data(self, path):
        """Read and validate the binary-mixture records."""
        dataset = pd.read_csv(path, low_memory=False)
        missing_columns = self._REQUIRED_COLUMNS.difference(dataset.columns)
        if missing_columns:
            raise ValueError(
                "Binary activity CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )
        return dataset.reset_index(drop=True)

    def read_solvent_smiles(self, solvent_list_path):
        """Read SMILES for solvent IDs used by the dataset."""
        solvents = pd.read_csv(solvent_list_path, index_col="solvent_id")
        if "smiles_can" not in solvents:
            raise ValueError("Solvent metadata must contain 'smiles_can'.")
        solvent_ids = pd.unique(
            self.dataset[["solv1", "solv2"]].to_numpy().ravel()
        ).tolist()
        return {
            solvent_id: solvents.loc[solvent_id, "smiles_can"]
            for solvent_id in solvent_ids
        }

    @staticmethod
    def build_solvent(molecule, converter):
        """Build one cached solvent entry: graph + H-bond descriptors."""
        graph = build_molecular_graph(molecule, converter)
        if graph is None:
            raise ValueError("Failed to build molecular graph.")
        hba = rdMolDescriptors.CalcNumHBA(molecule)
        hbd = rdMolDescriptors.CalcNumHBD(molecule)
        return {
            "graph": graph,
            "hba": hba,
            "hbd": hbd,
            "intra_hb": min(hba, hbd),
        }

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as file:
            pickle.dump(data, file)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as file:
                return pickle.load(file)
        raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def __getitem__(self, idx: int):
        """Get one binary-mixture sample."""
        if isinstance(idx, paddle.Tensor):
            idx = idx.item()
        row = self.dataset.iloc[idx]
        solvent1 = self.load_from_cache(self.graphs[self.solvent_index[row.solv1]])
        solvent2 = self.load_from_cache(self.graphs[self.solvent_index[row.solv2]])
        return {
            "g1": solvent1["graph"],
            "g2": solvent2["graph"],
            "x1": np.asarray([float(row.solv1_x)], dtype="float32"),
            "x2": np.asarray([1.0 - float(row.solv1_x)], dtype="float32"),
            "gamma1": np.asarray([float(row.solv1_gamma)], dtype="float32"),
            "gamma2": np.asarray([float(row.solv2_gamma)], dtype="float32"),
            "gamma": np.asarray(
                [float(row.solv1_gamma), float(row.solv2_gamma)], dtype="float32"
            ),
            "intra_hb1": np.asarray([solvent1["intra_hb"]], dtype="float32"),
            "intra_hb2": np.asarray([solvent2["intra_hb"]], dtype="float32"),
            "inter_hb": np.asarray(
                [
                    min(solvent1["hba"], solvent2["hbd"])
                    + min(solvent1["hbd"], solvent2["hba"])
                ],
                dtype="float32",
            ),
            "empty_solvsys": self.generate_solvsys(1),
        }

    def __len__(self):
        return self.num_samples

    @staticmethod
    def generate_solvsys(batch_size=5):
        nodes = 2 * batch_size
        source = (
            list(range(batch_size))
            + list(range(batch_size, nodes))
            + list(range(nodes))
        )
        target = (
            list(range(batch_size, nodes))
            + list(range(batch_size))
            + list(range(nodes))
        )
        return pgl.Graph(
            num_nodes=nodes,
            edges=np.asarray(list(zip(source, target)), dtype=np.int64),
        )
