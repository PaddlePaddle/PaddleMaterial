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

from __future__ import absolute_import
from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.distributed as dist
import pandas as pd
import pgl
from paddle.io import Dataset
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType as BT

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.datasets.build_spectrum import build_spectrum_converter
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.models.diffnmr.utils import diffgraphformer_utils as utils
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.ext_rdkit import build_molecule_with_partial_charges
from ppmat.utils.ext_rdkit import compute_molecular_metrics
from ppmat.utils.ext_rdkit import mol2smiles
from ppmat.utils.misc import is_equal


class MSDnmrDataset(Dataset):
    """Multimodal Spectrum Dataset‑Nuclear Magnetic Resonance subset handler.

    This class provides utilities for loading and processing the MSD NMR dataset.
    The dataset contains preprocess dataset includes SMILES of molecules, tokenized
    input of NMR and atom counts of molecules. Tokenized input includs chemical shift,
    multiplicity, intensity etc.
    The total dataset is divided into three parts: training, validation, and testing
    and devided into 4 parts by number of atoms per molecules.

    **Dataset Overview**
    - **Source**: Original data available at
    https://github.com/rxn4chemistry/multimodal-spectroscopic-dataset
    - **Preprocessed Version**:
    ```
    ┌───────────────────┬─────────┬─────────┬─────────┬──────────┐
    │ Dataset Partition │ Train   │ Val     │ Test    │ Total    │
    ├───────────────────┼─────────┼─────────┼─────────┼──────────┤
    │       n<15        │ 109,358 │ 6,076   │ 6,075   │ 121509   │
    ├───────────────────┼─────────┼─────────┼─────────┼──────────┤
    │       n<20        │ 235,512 │ 13,085  │ 13,084  │ 261681   │
    ├───────────────────┼─────────┼─────────┼─────────┼──────────┤
    │       n<25        │ 351,273 │ 19,516  │ 19,515  │ 390,304  │
    ├───────────────────┼─────────┼─────────┼─────────┼──────────┤
    │       n<35        │ 517,319 │ 28,741  │ 28,739  │ 574,799  │
    └───────────────────┴─────────┴─────────┴─────────┴──────────┘
    ```
    Download preprocessed data:
    https://paddle-org.bj.bcebos.com/paddlematerial/datasets/msd/msd_nmr.zip

    **Data Format**
    The dataset is stored in CSV format with the following structure:
    ```CSV
    smiles,tokenized_input,atom_count

    The tokenized_input is stored as a JSON-style dictionary with two top-level keys:
    "1HNMR"  : a list of proton (^1H) NMR signals
    "13CNMR" : a list of carbon (^13C) NMR chemical shifts

    Each element in the "1HNMR" list represents a single proton signal and is itself
    a five-element array in the form:
    [chemical_shift_ppm, line_width_ppm, multiplicity, integration, coupling_constants]

    - chemical_shift_ppm: float – the chemical shift δ value in parts per million.
    - line_width_ppm   : float – the peak width (or half-height width) in ppm.
    - multiplicity     : str   – the splitting pattern, e.g.:
        * "t"   : triplet
        * "dd"  : doublet of doublets
        * "td"  : triplet of doublet
        * "ddt" : doublet of doublet of triplet
        * "qt"  : quartet of triplet
        (other patterns may appear depending on the spectrum).
    - integration      : str   – the number of protons represented by the signal,
        expressed as a string like "1H", "2H", "3H", etc.
    - coupling_constants: list of floats – a list of J couplings (Hz) associated
        with this signal; an empty list means no couplings were reported.

    The "13CNMR" entry is simply a list of float values, where each value is a carbon
    chemical shift in ppm. No additional information (such as line widths or couplings)
    is provided for the carbon spectra.
    ```

    Args:
            path (str or List[str]): Path to a CSV file (or list of CSV files)
                containing the raw dataset. Each file should have columns such
                as 'smiles', 'tokenized_input' and 'atom_count'. If multiple
                files are provided, they will be concatenated.
            vocab_peakwidth_path (str): Path to a CSV file defining the
                vocabulary for NMR peak widths. The file should have a column
                named 'Value' whose unique entries are mapped to integer IDs.
            vocab_split_path (str): Path to a CSV file defining the vocabulary
                for NMR splitting types. The file should have a column named
                'Type' whose unique entries are mapped to integer IDs.
            remove_h (bool): Whether to remove hydrogen atoms from the graph
                representation. When ``True``, hydrogens are stripped and the
                remaining node features are shifted accordingly.
            seq_len_H1 (int): Maximum sequence length for ¹H NMR tokens. 1H
                spectra shorter than this will be padded; longer sequences are
                truncated.
            seq_len_C13 (int): Maximum sequence length for ¹³C NMR tokens.
            cache (bool, optional): If ``True``, processed graphs will be cached
                to a ``*.pkl`` file next to the input CSV. Subsequent runs
                reuse the cache when the file exists, speeding up initialization.
                Defaults to ``True``.
            **kwargs: Additional keyword arguments to configure dataset behaviour.
                Recognised keys include:
                  - ``guidance_target`` (str): one of {'mu','homo','both'}, used
                    when training a regressor to select which target(s) to return.
                  - ``regressor``: boolean or object indicating whether a
                    regression model is being trained, which affects the
                    transform applied to the labels.

    """

    name = "msd_nmr"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/msd/msd_nmr.zip"
    md5 = "bcd731f6d4075a93c11641fdebd1d6bd"

    def __init__(
        self,
        path: Union[str, List[str]],
        vocab_peakwidth_path: str,
        vocab_split_path: str,
        data_flag: str,
        max_atoms: int,
        build_molecule_cfg: Optional[Dict[str, Any]] = None,
        build_graph_cfg: Optional[Dict[str, Any]] = None,
        build_spectrum_cfg: Optional[Dict[str, Any]] = None,
        transforms: Optional[Any] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Download the dataset if the provided path does not exist
        if not osp.exists(path):
            logger.message("The dataset is not found. Will download it now.")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            if data_flag == "n<15":
                subdataset_name = "msd_nmr_nless15"
            elif data_flag == "n<20":
                subdataset_name = "msd_nmr_nless20"
            elif data_flag == "n<30":
                subdataset_name = "msd_nmr_nless30"
            elif data_flag == "n<35":
                subdataset_name = "msd_nmr_nless35"
            else:
                raise ValueError(
                    f"Unknown data_flag: {data_flag}. "
                    "Expected one of {'n<15', 'n<20', 'n<30', 'n<35'}."
                )
            path = osp.join(root_path, self.name, subdataset_name, osp.basename(path))

        self.path = path

        # Config dicts controlling molecule and graph construction
        if build_molecule_cfg is None:
            build_molecule_cfg = {
                "format": "smiles",
                "sanitize": False,
                "add_hs": False,
                "remove_hs": False,
                "kekulize": False,
            }
            logger.message(
                "The build_molecule_cfg is not set, will use the default "
                f"configs: {build_molecule_cfg}"
            )
        self.build_molecule_cfg = build_molecule_cfg

        if build_graph_cfg is None:
            build_graph_cfg = {
                "atom_vocab": {
                    "H": 0,
                    "C": 1,
                    "N": 2,
                    "O": 3,
                    "F": 4,
                    "P": 5,
                    "S": 6,
                    "Cl": 7,
                    "Br": 8,
                    "I": 9,
                },
                "bond_vocab": {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3},
                "remove_h": False,
                "add_self_loops": False,
                "edge_mode": "bidirectional",
                "num_cpus": 1,
            }
            logger.message(
                "The build_graph_cfg is not set, will use the default "
                f"configs: {build_graph_cfg}"
            )
        self.build_graph_cfg = build_graph_cfg

        if build_spectrum_cfg is None:
            build_spectrum_cfg = {
                "__class_name__": "BuildSpectrumNMR",  # 指定要实例化的类名
                "__init_params__": {  # 类初始化参数
                    "seq_len_H1": 32,  # 1H谱序列长度
                    "seq_len_C13": 32,  # 13C谱序列长度
                    "j_len": 6,  # 耦合常数维度
                    "unk_token": "<unk>",  # 未知token
                    "integral_offset": 1,  # 积分偏移量
                    "num_cpus": 1,  # 并行线程数
                },
            }
            logger.message(
                "The build_spectrum_cfg is not set, will use the default "
                f"configs: {build_spectrum_cfg}"
            )
        self.build_spectrum_cfg = build_spectrum_cfg

        self.transforms = transforms
        self.vocabs = self._build_vocab(vocab_peakwidth_path, vocab_split_path)

        if cache_path is not None:
            self.cache_path = cache_path
        else:
            self.cache_path = osp.join(
                osp.split(path)[0] + "_cache", osp.splitext(osp.basename(path))[0]
            )
        logger.info(f"Cache path: {self.cache_path}")

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        self.cache_exists = True if osp.exists(self.cache_path) else False
        self.raw_data, self.num_samples = self.read_data(path)
        logger.info(f"Load {self.num_samples} samples from {path}")

        if self.cache_exists and not overwrite:
            logger.warning(
                "Cache enabled. If a cache file exists, it will be automatically "
                "read and current settings will be ignored. Please ensure that the "
                "settings used in match your current settings."
            )
            try:
                build_molecule_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_molecule_cfg.pkl")
                )
                if is_equal(build_molecule_cfg_cache, build_molecule_cfg):
                    logger.info(
                        "The cached build_molecule_cfg configuration matches "
                        "the current settings. Reusing previously generated"
                        " structural data to optimize performance."
                    )
                else:
                    logger.warning(
                        "build_molecule_cfg is different from "
                        "build_molecule_cfgg_cache. Will rebuild the molecules and "
                        "graphs."
                    )
                    logger.warning(
                        "If you want to use the cached molecules and graphs, please "
                        "ensure that the settings used in match your current settings."
                    )
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning(
                    "Failed to load builded_molecules_cfg.pkl from cache. "
                    "Will rebuild the molecules and graphs(if need)."
                )
                overwrite = True

            if build_graph_cfg is not None and not overwrite:
                try:
                    build_graph_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_graph_cfg.pkl")
                    )
                    if is_equal(build_graph_cfg_cache, build_graph_cfg):
                        logger.info(
                            "The cached build_molecule_cfg configuration "
                            "matches the current settings. Reusing previously "
                            "generated molecular data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "build_graph_cfg is different from build_graph_cfg_cache"
                            ". Will rebuild the graphs."
                        )
                        logger.warning(
                            "If you want to use the cached molecules and graphs, "
                            "please ensure that the settings used in match your "
                            "current settings."
                        )
                        overwrite = True

                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load builded_graph_cfg.pkl from cache. "
                        "Will rebuild the graphs."
                    )
                    overwrite = True

            if build_spectrum_cfg is not None and not overwrite:
                try:
                    build_spectrum_cfg_cache = self.load_from_cache(
                        osp.join(self.cache_path, "build_spectrum_cfg.pkl")
                    )
                    if is_equal(build_spectrum_cfg_cache, build_spectrum_cfg):
                        logger.info(
                            "The cached build_spectrum_cfg configuration "
                            "matches the current settings. Reusing previously "
                            "generated spectrum data to optimize performance."
                        )
                    else:
                        logger.warning(
                            "build_spectrum_cfg is different from "
                            "build_spectrum_cfg_cache. Will rebuild the spectrums."
                        )
                        logger.warning(
                            "If you want to use the cached spectrums, "
                            "please ensure that the settings used in match your "
                            "current settings."
                        )
                        overwrite = True

                except Exception as e:
                    logger.warning(e)
                    logger.warning(
                        "Failed to load builded_spectrum_cfg.pkl from cache. "
                        "Will rebuild the spectrums."
                    )
                    overwrite = True

        molecule_cache_path = osp.join(self.cache_path, "molecules")
        graph_cache_path = osp.join(self.cache_path, "graphs")
        spectrum_cache_path = osp.join(self.cache_path, "spectrums")

        if overwrite or not self.cache_exists:
            # convert strucutes and graphs
            # only rank 0 process do the conversion
            if dist.get_rank() == 0:
                # save build_molecule_cfg and build_graph_cfg and build_spechtrum_cfg
                # to cache file
                os.makedirs(self.cache_path, exist_ok=True)

                self.save_to_cache(
                    osp.join(self.cache_path, "build_molecule_cfg.pkl"),
                    build_molecule_cfg,
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )
                self.save_to_cache(
                    osp.join(self.cache_path, "build_spectrum_cfg.pkl"),
                    build_spectrum_cfg,
                )

                # convert strucutes
                molecules = BuildMolecule(**build_molecule_cfg)(self.raw_data["smiles"])
                # save molecules to cache file
                os.makedirs(molecule_cache_path, exist_ok=True)
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(molecule_cache_path, f"{i:010d}.pkl"),
                        molecules[i],
                    )
                logger.info(
                    f"Save {self.num_samples} molecules to {molecule_cache_path}"
                )
                # convert graphs
                if build_graph_cfg is not None:
                    converter = build_graph_converter(build_graph_cfg)
                    graphs = converter(molecules)
                    # save graphs to cache file
                    os.makedirs(graph_cache_path, exist_ok=True)
                    for i in range(self.num_samples):
                        self.save_to_cache(
                            osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                        )
                    logger.info(f"Save {self.num_samples} graphs to {graph_cache_path}")
                # convert spectrums
                if build_spectrum_cfg is not None:
                    converter = build_spectrum_converter(
                        build_spectrum_cfg, vocabs=self.vocabs, strict=True
                    )
                    spectrums = converter(self.raw_data["tokenized_nmr"])
                    # save spectrums to cache file
                    os.makedirs(spectrum_cache_path, exist_ok=True)
                    for i in range(self.num_samples):
                        self.save_to_cache(
                            osp.join(spectrum_cache_path, f"{i:010d}.pkl"), spectrums[i]
                        )
                    logger.info(
                        f"Save {self.num_samples} spectrums to {spectrum_cache_path}"
                    )

            # sync all processes
            if dist.is_initialized():
                dist.barrier()
        self.molecules = [
            osp.join(molecule_cache_path, f"{i:010d}.pkl")
            for i in range(self.num_samples)
        ]
        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.graphs = None
        if build_spectrum_cfg is not None:
            self.spectrums = [
                osp.join(spectrum_cache_path, f"{i:010d}.pkl")
                for i in range(self.num_samples)
            ]
        else:
            self.spectrums = None
        self.properties = self.get_property_data(self.raw_data)  # represent "y"

        assert (
            len(self.molecules) == self.num_samples
        ), "The number of molecules must be equal to the number of samples."
        assert (
            self.graphs is None or len(self.graphs) == self.num_samples
        ), "The number of graphs must be equal to the number of samples."

        # filter data by specific requirement such as max atom number
        if filter_unvalid:
            self.filter_by_atom_count_raw(max_atoms=max_atoms)

    def __getitem__(self, idx: int) -> Tuple[pgl.Graph, Dict[str, Any]]:
        """Get item at index idx."""
        data = {}

        # get graph
        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph
        else:
            molecule = self.molecules[idx]
            if isinstance(molecule, str):
                molecule = self.load_from_cache(molecule)
            data["molecule_array"] = self.get_molecule_array(molecule)

        # get spectrum
        if self.spectrums is not None:
            spectrum = self.spectrums[idx]
            if isinstance(spectrum, str):
                spectrum = self.load_from_cache(spectrum)
            data["spectrum"] = spectrum

        # get property-like data "y"
        if self.properties is not None:
            property = self.properties[idx]
            atom_count = self.raw_data["atom_count"][idx]
            if isinstance(property, str):
                property = self.load_from_cache(property)
            data["property"] = {
                "y": property,
                "atom_count": atom_count,
            }

        # data = self.transforms(data) if self.transforms is not None else data

        return data

    def __len__(self) -> int:
        return self.num_samples

    def read_data(
        self, csv_path: Union[str, List[str]]
    ) -> Tuple[List[str], List[dict], List[int], int]:
        """Read MSD-NMR raw CSV file(s) and return parsed columns.

        Expected CSV schema (3 columns in order):
            1) smiles                      (str)
            2) tokenized_input (JSON str)  (e.g., '{"1HNMR":[...], "13CNMR":[...]}')
            3) atom_count                  (int)

        Args:
            csv_path (str | List[str]): Path to a CSV file, a directory containing CSVs,
                or a list of CSV file paths.

        Returns:
            raw_data (dict): {
                "smiles":         List[str],
                "tokenized_nmr":  List[dict|list],   # parsed from tokenized_input
                "atom_count":     List[int],
            }
            num_samples (int)
        """
        # 1) Collect CSV files
        if isinstance(csv_path, (list, tuple)):
            file_list = list(csv_path)
        elif osp.isdir(csv_path):
            file_list = [
                osp.join(csv_path, f)
                for f in os.listdir(csv_path)
                if f.endswith(".csv")
            ]
            file_list.sort()
        else:
            file_list = [csv_path]

        if len(file_list) == 0:
            return [], [], [], 0

        # 2) Read and concatenate
        frames = []
        for p in file_list:
            df = pd.read_csv(p)
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)

        # 3) Normalize/rename columns if needed
        # Preferred canonical names: 'smiles', 'tokenized_input', 'atom_count'
        cols = [c.lower().strip() for c in df.columns.tolist()]
        rename_map = {}
        # Attempt to map by known names; fall back to position
        # (0: smiles, 1: tokenized_input, 2: atom_count)
        if "smiles" not in cols:
            rename_map[df.columns[0]] = "smiles"
        else:
            # Make sure exact canonical name
            rename_map[df.columns[cols.index("smiles")]] = "smiles"

        if "tokenized_input" not in cols:
            # If user used a different header, assume second column
            rename_map[df.columns[1]] = "tokenized_input"
        else:
            rename_map[df.columns[cols.index("tokenized_input")]] = "tokenized_input"

        if "atom_count" not in cols:
            rename_map[df.columns[2]] = "atom_count"
        else:
            rename_map[df.columns[cols.index("atom_count")]] = "atom_count"

        df = df.rename(columns=rename_map)

        # 4) Parse tokenized_input JSON (if it's a string)
        if "tokenized_input" not in df.columns:
            raise ValueError("Column 'tokenized_input' not found after normalization.")

        def _parse_json(x):
            if isinstance(x, (dict, list)):
                return x
            if pd.isna(x):
                return None
            # ensure it's a JSON string
            if not isinstance(x, str):
                x = str(x)
            return json.loads(x)

        df["tokenized_input"] = df["tokenized_input"].apply(_parse_json)

        # 5) Ensure atom_count is integer
        if "atom_count" not in df.columns:
            raise ValueError("Column 'atom_count' not found after normalization.")

        df["atom_count"] = pd.to_numeric(df["atom_count"], errors="coerce").astype(
            "Int64"
        )

        # 6) Drop invalid rows (any of the three columns invalid)
        valid_mask = (
            df["smiles"].astype(str).str.len().gt(0)
            & df["tokenized_input"].notna()
            & df["atom_count"].notna()
        )
        df = df.loc[valid_mask].reset_index(drop=True)

        # 7) Build outputs
        raw_data: Dict[str, List[Any]] = {
            "smiles": df["smiles"].astype(str).tolist(),
            "tokenized_nmr": df["tokenized_input"].tolist(),
            "atom_count": [int(v) for v in df["atom_count"].tolist()],
        }
        num_samples = len(df)

        return raw_data, num_samples

    def save_to_cache(self, cache_path: str, data: Any):
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No such file or directory: {cache_path}")

    def filter_by_atom_count_raw(
        self,
        min_atoms: int | None = None,
        max_atoms: int | None = None,
        allowed: set[int] | list[int] | tuple[int, ...] | None = None,
        inplace: bool = True,
    ):
        """
        Filter samples based on raw_data['atom_count'].

        Criteria (AND):
        - min_atoms: keep if count >= min_atoms (if provided)
        - max_atoms: keep if count <= max_atoms (if provided)
        - allowed:   keep if count ∈ allowed (if provided)

        Returns:
            reserve_idx (List[int]): kept indices (when inplace=True, also mutates
                datasets).

        Usage:
            # filter by atom count range [5, 20]
            dataset.filter_by_atom_count_raw(min_atoms=5, max_atoms=20)

            # filter by specific atom counts ∈ {10, 12, 14}
            dataset.filter_by_atom_count_raw(allowed={10, 12, 14})

            # return filtered indices without modifying data
            idx = dataset.filter_by_atom_count_raw(min_atoms=6, inplace=False)
        """
        # 1) Read atom counts from raw data (required source) raw_data['atom_count']
        if not hasattr(self, "raw_data") or "atom_count" not in self.raw_data:
            raise ValueError("raw_data['atom_count'] is required for filtering.")

        counts = np.asarray(self.raw_data["atom_count"], dtype=np.int64).reshape(-1)
        n = counts.shape[0]
        keep = np.ones(n, dtype=bool)

        # 2) Apply filtering criteria (combined with AND)
        if min_atoms is not None:
            keep &= counts >= int(min_atoms)
        if max_atoms is not None:
            keep &= counts <= int(max_atoms)
        if allowed is not None:
            allowed_arr = np.asarray(list(allowed), dtype=np.int64)
            keep &= np.isin(counts, allowed_arr)

        reserve_idx = np.nonzero(keep)[0].tolist()
        filtered_out = n - len(reserve_idx)

        # If not mutating the dataset, return the indices only
        if not inplace:
            return reserve_idx

        # 3) Slice all containers that are aligned to raw_data length
        def _slice_list(lst):
            return [lst[i] for i in reserve_idx]

        # 3.1 raw_data: slice list-like values with length n
        for k, v in list(self.raw_data.items()):
            if isinstance(v, list) and len(v) == n:
                self.raw_data[k] = _slice_list(v)

        # 3.2 property_data: slice when aligned
        if hasattr(self, "property_data") and isinstance(self.property_data, dict):
            for k, v in list(self.property_data.items()):
                if isinstance(v, list) and len(v) == n:
                    self.property_data[k] = _slice_list(v)

        # 3.3 other parallel containers (if present and aligned)
        for attr in ("raw_data", "molecules", "graphs", "metas"):
            if hasattr(self, attr):
                seq = getattr(self, attr)
                if isinstance(seq, list) and len(seq) == n:
                    setattr(self, attr, _slice_list(seq))

        # 4) Update dataset size and log
        if hasattr(self, "num_samples"):
            self.num_samples = len(reserve_idx)

        try:
            from ppmat.utils import logger

            logger.warning(
                f"[filter_by_atom_count_raw] filtered_out={filtered_out}, "
                f"remaining={self.num_samples}, criteria: "
                f"min={min_atoms}, max={max_atoms}, "
                f"allowed={set(allowed) if allowed is not None else None}"
            )
        except Exception:
            pass

        return reserve_idx

    def get_property_data(self, data):
        property = np.zeros([len(data["smiles"]), 0], dtype=np.float32)
        return property

    def get_molecule_array(self, molecule):
        """
        Return graph-ready arrays (not a pgl.Graph):
        - num_nodes: [1] int64
        - edges:     [E, 2] int64 (sorted)
        - node_feat: [N, A] float32  (A = len(atom_vocab))
        - edge_feat: [E, K] float32  (K = len(bond_vocab) + 1, 0 reserved)
        """
        # ---- config / defaults (reuse same knobs as MolecularGraphConverter) ----
        atom_vocab = getattr(
            self,
            "atom_vocab",
            {
                "H": 0,
                "C": 1,
                "N": 2,
                "O": 3,
                "F": 4,
                "P": 5,
                "S": 6,
                "Cl": 7,
                "Br": 8,
                "I": 9,
            },
        )
        bond_vocab = getattr(
            self, "bond_vocab", (BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC)
        )
        remove_h = bool(getattr(self, "remove_h", False))
        add_self = bool(getattr(self, "add_self_loops", False))
        edge_mode = getattr(self, "edge_mode", "bidirectional")

        # ---- RDKit Mol ----
        mol = Chem.MolFromSmiles(molecule) if isinstance(molecule, str) else molecule
        if mol is None:
            raise ValueError(f"Invalid molecule/SMILES: {molecule}")
        if remove_h:
            mol = Chem.RemoveHs(mol)
        N = mol.GetNumAtoms()
        if N == 0:
            # empty arrays for consistency
            return {
                "num_nodes": ConcatData(np.asarray([0], dtype=np.int64)),
                "edges": ConcatData(np.zeros((0, 2), dtype=np.int64)),
                "node_feat": ConcatData(
                    np.zeros((0, len(atom_vocab)), dtype=np.float32)
                ),
                "edge_feat": ConcatData(
                    np.zeros((0, len(bond_vocab) + 1), dtype=np.float32)
                ),
            }

        # ---- node_feat (one-hot over atom_vocab) ----
        idxs = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in atom_vocab:
                raise ValueError(f"Unknown atom symbol '{sym}' not in atom_vocab")
            idxs.append(atom_vocab[sym])
        idxs = np.asarray(idxs, dtype=np.int64)  # [N]
        node_feat = np.eye(len(atom_vocab), dtype=np.float32)[idxs]  # [N, A]

        # ---- edges & edge_feat (bond one-hot over bond_vocab + 0) ----
        rows, cols, etypes = [], [], []
        bt2id = {bt: i + 1 for i, bt in enumerate(bond_vocab)}  # 0 reserved

        def push(u, v, et):
            rows.append(u)
            cols.append(v)
            etypes.append(et)

        for b in mol.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            et = bt2id.get(b.GetBondType(), 0)

            if edge_mode == "directed":
                push(u, v, et)
            elif edge_mode == "undirected":
                uu, vv = (u, v) if u < v else (v, u)
                push(uu, vv, et)
            elif edge_mode == "bidirectional":
                push(u, v, et)
                push(v, u, et)
            else:
                raise ValueError(f"Unknown edge_mode: {edge_mode}")

        # dedup for undirected
        if edge_mode == "undirected" and rows:
            pair = np.stack([np.asarray(rows), np.asarray(cols)], axis=1)  # [E,2]
            ety = np.asarray(etypes)
            view = pair.view([("r", pair.dtype), ("c", pair.dtype)])[:, 0]
            _, keep = np.unique(view, return_index=True)
            pair, ety = pair[keep], ety[keep]
            rows, cols, etypes = pair[:, 0].tolist(), pair[:, 1].tolist(), ety.tolist()

        if add_self:
            for i in range(N):
                rows.append(i)
                cols.append(i)
                etypes.append(0)

        if rows:
            row_np = np.asarray(rows, dtype=np.int64)
            col_np = np.asarray(cols, dtype=np.int64)
            et_np = np.asarray(etypes, dtype=np.int64)
            # deterministic order
            order = np.argsort(row_np * N + col_np, kind="mergesort")
            row_np, col_np, et_np = row_np[order], col_np[order], et_np[order]
            edges = np.stack([row_np, col_np], axis=1).astype(np.int64)  # [E,2]
            edge_feat = np.eye(len(bond_vocab) + 1, dtype=np.float32)[et_np]  # [E,K]
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            edge_feat = np.zeros((0, len(bond_vocab) + 1), dtype=np.float32)

        # ---- pack arrays (mirrors the 4 graph arguments) ----
        molecule_array = {
            "num_nodes": ConcatData(np.asarray([N], dtype=np.int64)),
            "edges": ConcatData(edges),  # [E, 2] int64
            "node_feat": ConcatData(node_feat),  # [N, A] float32
            "edge_feat": ConcatData(edge_feat),  # [E, K] float32
        }
        return molecule_array

    # ------------------------------------------------------------------
    # Internal data preparation methods
    # These mirror the functionality previously provided by ``MMSnmrData`` and
    # are kept private within the dataset class to simplify usage.

    def _build_vocab(self, peakwidth_path: str, split_path: str):
        """
        Populate the peak width and split vocabularies from CSV files.
        Return {'peakwidth': {...}, 'split': {...}} vocab dicts from CSVs.
        """

        def uniq_keep_order(xs):
            seen, out = set(), []
            for x in xs:
                if x is None:
                    continue
                s = str(x).strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            return out

        df_pw = pd.read_csv(peakwidth_path)
        df_sp = pd.read_csv(split_path)

        pw_tokens = uniq_keep_order(df_pw["Value"].tolist())
        sp_tokens = uniq_keep_order(df_sp["Type"].tolist())

        vocab_peakwidth = {"<pad>": 0, "<unk>": 1}
        vocab_peakwidth.update({t: i + 2 for i, t in enumerate(pw_tokens)})

        vocab_split = {"<pad>": 0, "<unk>": 1}
        vocab_split.update({t: i + 2 for i, t in enumerate(sp_tokens)})

        return {"peakwidth": vocab_peakwidth, "split": vocab_split}


class MSDnmrinfos:
    def __init__(self, dataloaders, cfg, recompute_statistics=False):
        self.remove_h = cfg["build_graph_cfg"]["__init_params__"]["remove_h"]
        self.dataflag = cfg["data_flag"]
        self.need_to_strip = (
            False  # to indicate whether we need to ignore one output from the model
        )

        self.atom_encoder = (
            {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
            if not self.remove_h
            else {
                "C": 0,
                "N": 1,
                "O": 2,
                "F": 3,
                "P": 4,
                "S": 5,
                "Cl": 6,
                "Br": 7,
                "I": 8,
            }
        )
        self.atom_decoder = list(self.atom_encoder.keys())
        self.num_atom_types = len(self.atom_encoder)
        self.valencies = (
            [1, 4, 3, 2, 1] if not self.remove_h else [4, 3, 2, 1, 3, 2, 1, 1, 1]
        )
        self.atom_weights = (
            {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            if not self.remove_h
            else {
                0: 12,
                1: 14,
                2: 16,
                3: 19,
                4: 30.97,
                5: 32.07,
                6: 35.45,
                7: 79.9,
                8: 126.9,
            }
        )
        if self.dataflag == "n<15":
            self.max_n_nodes = 29 if not self.remove_h else 15
            self.max_weight = 390 if not self.remove_h else 564

            self.n_nodes = (
                paddle.to_tensor(
                    [
                        0,
                        0,
                        0,
                        1.5287e-05,
                        3.0574e-05,
                        3.8217e-05,
                        9.1721e-05,
                        0.00015287,
                        0.00049682,
                        0.0013147,
                        0.0036918,
                        0.0080486,
                        0.016732,
                        0.03078,
                        0.051654,
                        0.078085,
                        0.10566,
                        0.1297,
                        0.13332,
                        0.1387,
                        0.094802,
                        0.10063,
                        0.033845,
                        0.048628,
                        0.0054421,
                        0.014698,
                        0.00045096,
                        0.0027211,
                        0.0,
                        0.00026752,
                    ]
                )
                if not self.remove_h
                else paddle.to_tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.000657983182463795,
                        0.0034172674641013145,
                        0.009784846566617489,
                        0.019774870947003365,
                        0.04433957487344742,
                        0.07253380119800568,
                        0.10895635187625885,
                        0.14755095541477203,
                        0.17605648934841156,
                        0.19964483380317688,
                        0.21728302538394928,
                    ]
                )
            )
            self.node_types = (
                paddle.to_tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
                if not self.remove_h
                else paddle.to_tensor(
                    [
                        0.7162184715270996,
                        0.09598348289728165,
                        0.12478094547986984,
                        0.01828921213746071,
                        0.0004915347089990973,
                        0.014545895159244537,
                        0.01616295613348484,
                        0.011324135586619377,
                        0.002203370677307248,
                    ]
                )
            )
            self.edge_types = (
                paddle.to_tensor([0.88162, 0.11062, 0.0059875, 0.0017758, 0])
                if not self.remove_h
                else paddle.to_tensor(
                    [
                        0.8293983340263367,
                        0.09064729511737823,
                        0.011958839371800423,
                        0.0011387828271836042,
                        0.0668567642569542,
                    ]
                )
            )
        elif self.dataflag == "n<20":
            self.max_n_nodes = 29 if not self.remove_h else 20
            self.max_weight = 390 if not self.remove_h else 631
            self.n_nodes = paddle.to_tensor(
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    1.465040404582396150e-04,
                    7.087133126333355904e-04,
                    2.005274174734950066e-03,
                    4.010548349469900131e-03,
                    9.273706004023551941e-03,
                    1.550195924937725067e-02,
                    2.318426594138145447e-02,
                    3.164304420351982117e-02,
                    3.758744522929191589e-02,
                    4.201003536581993103e-02,
                    4.522579908370971680e-02,
                    4.758085310459136963e-02,
                    4.873823374509811401e-02,
                    5.004395171999931335e-02,
                    4.889755696058273315e-02,
                    4.859539121389389038e-02,
                    4.685382544994354248e-02,
                    4.636486992239952087e-02,
                    4.473684355616569519e-02,
                    4.392923787236213684e-02,
                    4.205032438039779663e-02,
                    4.190198704600334167e-02,
                    3.956525027751922607e-02,
                    3.861114010214805603e-02,
                    3.698311373591423035e-02,
                    3.511701896786689758e-02,
                    3.210086748003959656e-02,
                    2.951690368354320526e-02,
                    2.601728774607181549e-02,
                    2.254880405962467194e-02,
                    1.854924298822879791e-02,
                ]
            )
            self.node_types = paddle.to_tensor(
                [
                    7.415896058082580566e-01,
                    9.485986828804016113e-02,
                    1.080681160092353821e-01,
                    2.368708699941635132e-02,
                    3.370510821696370840e-04,
                    1.273731887340545654e-02,
                    1.297908369451761246e-02,
                    4.853925667703151703e-03,
                    8.879197412170469761e-04,
                ]
            )
            self.edge_types = paddle.to_tensor(
                [
                    9.066669344902038574e-01,
                    4.404582828283309937e-02,
                    5.253293085843324661e-03,
                    3.737418155651539564e-04,
                    4.366017505526542664e-02,
                ]
            )
        elif self.dataflag == "n<25":
            self.max_n_nodes = 29 if not self.remove_h else 25
            self.max_weight = 390 if not self.remove_h else 998
            self.n_nodes = paddle.to_tensor(
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    1.465040404582396150e-04,
                    7.087133126333355904e-04,
                    2.005274174734950066e-03,
                    4.010548349469900131e-03,
                    9.273706004023551941e-03,
                    1.550195924937725067e-02,
                    2.318426594138145447e-02,
                    3.164304420351982117e-02,
                    3.758744522929191589e-02,
                    4.201003536581993103e-02,
                    4.522579908370971680e-02,
                    4.758085310459136963e-02,
                    4.873823374509811401e-02,
                    5.004395171999931335e-02,
                    4.889755696058273315e-02,
                    4.859539121389389038e-02,
                    4.685382544994354248e-02,
                    4.636486992239952087e-02,
                    4.473684355616569519e-02,
                    4.392923787236213684e-02,
                    4.205032438039779663e-02,
                    4.190198704600334167e-02,
                    3.956525027751922607e-02,
                    3.861114010214805603e-02,
                    3.698311373591423035e-02,
                    3.511701896786689758e-02,
                    3.210086748003959656e-02,
                    2.951690368354320526e-02,
                    2.601728774607181549e-02,
                    2.254880405962467194e-02,
                    1.854924298822879791e-02,
                ]
            )
            self.node_types = paddle.to_tensor(
                [
                    7.415896058082580566e-01,
                    9.485986828804016113e-02,
                    1.080681160092353821e-01,
                    2.368708699941635132e-02,
                    3.370510821696370840e-04,
                    1.273731887340545654e-02,
                    1.297908369451761246e-02,
                    4.853925667703151703e-03,
                    8.879197412170469761e-04,
                ]
            )
            self.edge_types = paddle.to_tensor(
                [
                    9.066669344902038574e-01,
                    4.404582828283309937e-02,
                    5.253293085843324661e-03,
                    3.737418155651539564e-04,
                    4.366017505526542664e-02,
                ]
            )
        elif self.dataflag == "n<35":
            self.max_n_nodes = 29 if not self.remove_h else 35
            self.max_weight = 390 if not self.remove_h else 1094
            self.n_nodes = paddle.to_tensor(
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    1.465040404582396150e-04,
                    7.087133126333355904e-04,
                    2.005274174734950066e-03,
                    4.010548349469900131e-03,
                    9.273706004023551941e-03,
                    1.550195924937725067e-02,
                    2.318426594138145447e-02,
                    3.164304420351982117e-02,
                    3.758744522929191589e-02,
                    4.201003536581993103e-02,
                    4.522579908370971680e-02,
                    4.758085310459136963e-02,
                    4.873823374509811401e-02,
                    5.004395171999931335e-02,
                    4.889755696058273315e-02,
                    4.859539121389389038e-02,
                    4.685382544994354248e-02,
                    4.636486992239952087e-02,
                    4.473684355616569519e-02,
                    4.392923787236213684e-02,
                    4.205032438039779663e-02,
                    4.190198704600334167e-02,
                    3.956525027751922607e-02,
                    3.861114010214805603e-02,
                    3.698311373591423035e-02,
                    3.511701896786689758e-02,
                    3.210086748003959656e-02,
                    2.951690368354320526e-02,
                    2.601728774607181549e-02,
                    2.254880405962467194e-02,
                    1.854924298822879791e-02,
                ]
            )
            self.node_types = paddle.to_tensor(
                [
                    7.415896058082580566e-01,
                    9.485986828804016113e-02,
                    1.080681160092353821e-01,
                    2.368708699941635132e-02,
                    3.370510821696370840e-04,
                    1.273731887340545654e-02,
                    1.297908369451761246e-02,
                    4.853925667703151703e-03,
                    8.879197412170469761e-04,
                ]
            )
            self.edge_types = paddle.to_tensor(
                [
                    9.066669344902038574e-01,
                    4.404582828283309937e-02,
                    5.253293085843324661e-03,
                    3.737418155651539564e-04,
                    4.366017505526542664e-02,
                ]
            )
        else:
            logger.message("invalid dataflag: %s", self.dataflag)
        self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
        self.valency_distribution = paddle.zeros(3 * self.max_n_nodes - 2)
        if self.dataflag == "n<15":
            if not self.remove_h:
                self.valency_distribution[0:6] = paddle.to_tensor(
                    [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
                )
            else:
                self.valency_distribution[0:7] = paddle.to_tensor(
                    [
                        0.000000000000000000e00,
                        1.856458932161331177e-01,
                        2.707855999469757080e-01,
                        3.008204102516174316e-01,
                        2.362315803766250610e-01,
                        3.544347826391458511e-03,
                        2.972166286781430244e-03,
                    ]
                )
        elif self.dataflag == "n<20":
            if not self.remove_h:
                self.valency_distribution[0:6] = paddle.to_tensor(
                    [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
                )
            else:
                self.valency_distribution[0:7] = paddle.to_tensor(
                    [
                        0.000000000000000000e00,
                        1.382219046354293823e-01,
                        2.489367425441741943e-01,
                        3.354085683822631836e-01,
                        2.695656120777130127e-01,
                        3.342652227729558945e-03,
                        4.524504765868186951e-03,
                    ]
                )
        elif self.dataflag == "n<25":
            if not self.remove_h:
                self.valency_distribution[0:6] = paddle.to_tensor(
                    [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
                )
            else:
                self.valency_distribution[0:7] = paddle.to_tensor(
                    [
                        0.000000000000000000e00,
                        1.382219046354293823e-01,
                        2.489367425441741943e-01,
                        3.354085683822631836e-01,
                        2.695656120777130127e-01,
                        3.342652227729558945e-03,
                        4.524504765868186951e-03,
                    ]
                )
        elif self.dataflag == "n<35":
            if not self.remove_h:
                self.valency_distribution[0:6] = paddle.to_tensor(
                    [0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012]
                )
            else:
                self.valency_distribution[0:7] = paddle.to_tensor(
                    [
                        0.000000000000000000e00,
                        1.382219046354293823e-01,
                        2.489367425441741943e-01,
                        3.354085683822631836e-01,
                        2.695656120777130127e-01,
                        3.342652227729558945e-03,
                        4.524504765868186951e-03,
                    ]
                )
        if recompute_statistics:
            self.n_nodes = dataloaders.node_counts()
            self.node_types = dataloaders.node_types()
            self.edge_types = dataloaders.edge_counts()
            self.valency_distribution = dataloaders.valency_count(self.max_n_nodes)

        self.train_smiles = get_train_smiles(
            cfg, dataloaders.train_dataloader, self, evaluate_dataset=False
        )

    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(
        self, dataloader, extra_features, domain_features, conditionDim=0
    ):
        data = next(iter(dataloader()))
        graph = data["graph"]
        spectrum = data["spectrum"]
        property = data["property"]
        ex_dense, node_mask = utils.to_dense(
            paddle.to_tensor(graph.node_feat["feat"]),
            paddle.to_tensor(graph.edges.T),
            paddle.to_tensor(graph.edge_feat["feat"]),
            paddle.to_tensor(graph.graph_node_id),
        )
        example_data = {
            "X_t": ex_dense.X,
            "E_t": ex_dense.E,
            "y_t": spectrum,
            "node_mask": node_mask,
        }

        self.input_dims = {
            "X": graph.node_feat["feat"].shape[1],
            "E": graph.edge_feat["feat"].shape[1],
            "y": property["y"].shape[1] + 1,
        }  # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims["X"] += ex_extra_feat.X.shape[-1]
        self.input_dims["E"] += ex_extra_feat.E.shape[-1]
        self.input_dims["y"] += ex_extra_feat.y.shape[-1]

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims["X"] += ex_extra_molecular_feat.X.shape[-1]
        self.input_dims["E"] += ex_extra_molecular_feat.E.shape[-1]
        self.input_dims["y"] += ex_extra_molecular_feat.y.shape[-1]

        self.input_dims["y"] += conditionDim

        self.output_dims = {
            "X": graph.node_feat["feat"].shape[1],
            "E": graph.edge_feat["feat"].shape[1],
            "y": 0,
        }


def get_train_smiles(cfg, dataloader, dataset_infos, evaluate_dataset=False):
    if evaluate_dataset:
        assert (
            dataset_infos is not None
        ), "If wanting to evaluate dataset, need to pass dataset_infos"
    if not osp.exists(cfg["datadir"]):
        logger.message(
            "The dataset directory is not found. Will save it to default path now."
        )
        root_path = download.get_datasets_path_from_url(
            MSDnmrDataset.url, MSDnmrDataset.md5
        )
        path = osp.join(root_path, MSDnmrDataset.name, osp.basename(cfg["datadir"]))
        if cfg["data_flag"] == "n<15":
            subdataset_name = "msd_nmr_nless15"
        elif cfg["data_flag"] == "n<20":
            subdataset_name = "msd_nmr_nless20"
        elif cfg["data_flag"] == "n<30":
            subdataset_name = "msd_nmr_nless30"
        elif cfg["data_flag"] == "n<35":
            subdataset_name = "msd_nmr_nless35"
        else:
            raise ValueError(
                f"Unknown data_flag: {cfg['data_flag']}. Expected one of "
                f"{'n<15', 'n<20', 'n<30', 'n<35'}."
            )
        path = osp.join(root_path, MSDnmrDataset.name, subdataset_name)

    remove_h = cfg["build_graph_cfg"]["__init_params__"]["remove_h"]
    atom_decoder = dataset_infos.atom_decoder

    smiles_file_name = "train_smiles_no_h.npy" if remove_h else "train_smiles_h.npy"
    smiles_path = os.path.join(path + "_cache", "train", smiles_file_name)
    if os.path.exists(smiles_path):
        logger.message("Dataset smiles were found")
        train_smiles = np.load(smiles_path)
    else:
        logger.message("Computing dataset smiles...")
        train_smiles = compute_MSDnmr_smiles(atom_decoder, dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        all_molecules = []
        for i, data in enumerate(dataloader):
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.graph_node_id
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E
            for k in range(X.shape[0]):
                n = int(paddle.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])
        logger.message(
            "Evaluating the dataset -- number of molecules to evaluate",
            len(all_molecules),
        )
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )
        logger.info(metrics[0])
    return train_smiles


def compute_MSDnmr_smiles(atom_decoder, dataloader, remove_h):
    logger.message(f"Converting MSDnmr dataset to SMILES for remove_h={remove_h}...")
    mols_smiles = []
    len_train = len(dataloader)
    invalid = 0
    disconnected = 0
    for i, batch in enumerate(dataloader):
        RDLogger.DisableLog("rdApp.*")
        if i % 1000 == 0:
            logger.message(
                f"Converting MSDnmr dataset to SMILES {float(i)/len_train:.2%}"
            )

        logger.info(f"compute_MSDnmr_smiles i: {i:d}")
        dense_data, node_mask = utils.to_dense(
            paddle.to_tensor(batch["graph"].node_feat["feat"], dtype="float32"),
            paddle.to_tensor(batch["graph"].edges.T, dtype="int64"),
            paddle.to_tensor(batch["graph"].edge_feat["feat"], dtype="float32"),
            paddle.to_tensor(batch["graph"].graph_node_id, dtype="int64"),
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E
        n_nodes = [int(paddle.sum((X != -1)[j, :])) for j in range(X.shape[0])]
        molecule_list = []
        for k in range(X.shape[0]):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
        for _, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(
                molecule[0], molecule[1], atom_decoder
            )
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=True
                )
                if len(mol_frags) > 1:
                    logger.info(f"Disconnected molecule {mol}, {mol_frags}")
                    disconnected += 1
            else:
                logger.info("Invalid molecule obtained.")
                invalid += 1

    logger.info(f"Number of invalid molecules {invalid}")
    logger.info(f"Number of disconnected molecules {disconnected}")

    return mols_smiles


class DataLoaderCollection:
    def __init__(self, train_dataloader, val_dataloader=None, test_dataloader=None):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def node_counts(self, max_nodes_possible=300):
        all_counts = paddle.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data, other_data in loader:
                unique, counts = np.unique(data.graph_node_id, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data, other_data in self.train_dataloader():
            num_classes = data.node_feat["feat"].shape[1]
            break

        counts = paddle.zeros(num_classes)

        for i, (data, other_data) in enumerate(self.train_dataloader()):
            counts += data.node_feat["feat"].sum(axis=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data, other_data in self.train_dataloader():
            num_classes = data.edge_feat["feat"].shape[1]
            break

        d = paddle.zeros(num_classes, dtype=paddle.float32)

        for i, (data, other_data) in enumerate(self.train_dataloader()):
            unique, counts = np.unique(data.graph_node_id, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edges.T.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_feat["feat"].sum(axis=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d

    def valency_count(self, max_n_nodes):
        valencies = paddle.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = paddle.to_tensor([0, 1, 2, 3, 1.5])

        for data, other_data in self.train_dataloader():
            n = data.node_feat["feat"].shape[0]

            for atom in range(n):
                edges = data.edge_feat["feat"][data.edges.T[0] == atom]
                edges_total = edges.sum(axis=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.astype("int64").item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class DistributionNodes(object):
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset,
            and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """
        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = paddle.zeros(shape=max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram
        self.prob = prob / prob.sum()
        self.m = paddle.distribution.Categorical(prob)

    def sample_n(self, n_samples):
        idx = self.m.sample((n_samples,))
        return idx

    def log_prob(self, batch_n_nodes):
        assert len(tuple(batch_n_nodes.shape)) == 1
        p = self.prob.to(batch_n_nodes.place)
        probas = p[batch_n_nodes]
        log_p = paddle.log(x=probas + 1e-30)
        return log_p


class SelecTargetTransform:
    """Dynamically select specific dimensions or targets from the data."""

    def __init__(
        self,
        target_indices: Union[int, Tuple[int, ...]],
        apply_keys: Tuple[str, ...] = ("input", "label"),
    ):
        if isinstance(target_indices, int):
            target_indices = (target_indices,)
        self.target_indices = target_indices
        self.apply_keys = apply_keys

    def __call__(self, data):
        for key in self.apply_keys:
            assert key in data, f"Key {key} does not exist in data."
            target = data[key]
            if isinstance(target, np.ndarray):
                data[key] = target[..., self.target_indices]
        return data


class RemoveYTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        data.y = np.zeros((1, 0), dtype="float32")
        return data


class SelectMuTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data
