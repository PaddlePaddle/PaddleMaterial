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

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from p_tqdm import p_map

from ppmat.utils import download as download_utils
from ppmat.utils import logger


def build_spectrum_converter(
    cfg: Dict,
    *,
    vocabs: Optional[Dict[str, Dict[str, int]]] = None,
    strict: bool = True,
):
    """Build spectrum converter.
    If 'vocabs' is provided (e.g., {'peakshape': {...}, 'intensity': {...}}),
    inject/merge it into __init_params__['vocabs'].

    Args:
        cfg (Dict): Spectrum converter config.
    """
    if cfg is None:
        return None
    cfg = copy.deepcopy(cfg)

    class_name = cfg.pop("__class_name__")
    if not class_name:
        raise ValueError(
            "Spectrum converter class name is not specified in the configuration."
        )

    init_params = cfg.pop("__init_params__")
    if vocabs:
        init_params["vocabs"] = {**(init_params.get("vocabs") or {}), **vocabs}

    cls = _locate_class(class_name)

    # Optional strict check: ensure required vocabs exist and contain unk_token
    if strict and hasattr(cls, "REQUIRED_VOCABS"):
        req = set(cls.REQUIRED_VOCABS)
        got = set((init_params.get("vocabs") or {}).keys())
        miss = req - got
        if miss:
            raise ValueError(
                f"{class_name} is missing required vocabularies: {sorted(miss)}"
            )

        unk = str(init_params.get("unk_token", "<unk>"))

        for name, vb in (init_params.get("vocabs") or {}).items():
            if unk not in vb:
                raise ValueError(
                    f"Vocabulary '{name}' must include the unknown token '{unk}'"
                )

    spectrum_converter = eval(class_name)(**init_params)
    logger.debug(str(spectrum_converter))

    return spectrum_converter


def _parse_factory_cfg(
    cfg: Optional[Dict[str, Any] | str],
    *,
    default_class_name: str,
) -> Tuple[str, Dict[str, Any]]:
    if cfg is None:
        return default_class_name, {}

    if isinstance(cfg, str):
        return cfg, {}

    if not isinstance(cfg, dict):
        raise TypeError(
            "Factory cfg must be None, str, or dict, "
            f"but got type={type(cfg).__name__}."
        )

    cfg = copy.deepcopy(cfg)
    class_name = (
        cfg.pop("__class_name__", None)
        or cfg.pop("class_name", None)
        or cfg.pop("type", None)
    )
    if not class_name:
        raise ValueError(
            "Factory cfg must include class name key, e.g. "
            "{'__class_name__': 'StrictIndexSampleBuilder', '__init_params__': {...}}."
        )

    init_params = (
        cfg.pop("__init_params__", None)
        if "__init_params__" in cfg
        else cfg.pop("init_params", None)
        if "init_params" in cfg
        else cfg.pop("params", None)
        if "params" in cfg
        else {}
    )
    if init_params is None:
        init_params = {}
    if not isinstance(init_params, dict):
        raise TypeError(
            f"Factory init params must be dict, but got type={type(init_params).__name__}."
        )

    if cfg:
        raise ValueError(
            f"Unsupported keys in factory cfg for '{class_name}': {list(cfg.keys())}"
        )
    return class_name, init_params


def _build_component(
    cfg: Optional[Dict[str, Any] | str],
    *,
    default_class_name: str,
    required_methods: List[str],
):
    class_name, init_params = _parse_factory_cfg(
        cfg,
        default_class_name=default_class_name,
    )

    cls = _locate_class(class_name)
    component = cls(**init_params)
    for method_name in required_methods:
        if not hasattr(component, method_name):
            raise TypeError(
                f"Component '{class_name}' must implement method '{method_name}'."
            )
    return component


class StrictIndexSampleBuilder:
    def build(
        self,
        noisy_dir: Path,
        target_dir: Path,
        file_suffix: str,
        data_count: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        samples: List[Dict[str, str]] = []
        if data_count is None:
            available = sorted(noisy_dir.glob(f"*{file_suffix}"))
            data_count = len(available)

        for idx in range(int(data_count)):
            name = f"{idx}{file_suffix}"
            noisy_path = noisy_dir / name
            target_path = target_dir / name
            if not noisy_path.exists():
                raise FileNotFoundError(f"Noisy image not found: {noisy_path}")
            if not target_path.exists():
                raise FileNotFoundError(f"Target image not found: {target_path}")
            samples.append(
                {
                    "name": name,
                    "noisy_path": str(noisy_path),
                    "target_path": str(target_path),
                }
            )
        return samples


class MatchedNameSampleBuilder:
    def build(
        self,
        noisy_dir: Path,
        target_dir: Path,
        file_suffix: str,
        data_count: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        noisy_files = {
            p.name: p for p in noisy_dir.glob(f"*{file_suffix}") if p.is_file()
        }
        target_files = {
            p.name: p for p in target_dir.glob(f"*{file_suffix}") if p.is_file()
        }
        common_names = sorted(set(noisy_files.keys()) & set(target_files.keys()))
        if data_count is not None:
            common_names = common_names[: int(data_count)]

        return [
            {
                "name": name,
                "noisy_path": str(noisy_files[name]),
                "target_path": str(target_files[name]),
            }
            for name in common_names
        ]


class DefaultSTEMDatasetDownloader:
    def __init__(self, datasets_home: Optional[str] = None):
        self.datasets_home = datasets_home or download_utils.DATASETS_HOME

    def download(
        self, url: str, md5: Optional[str] = None, force_download: bool = False
    ) -> Path:
        if force_download:
            downloaded_root = download_utils.get_path_from_url(
                url,
                self.datasets_home,
                md5sum=md5,
                check_exist=False,
                decompress=True,
            )
        else:
            downloaded_root = download_utils.get_datasets_path_from_url(url, md5)
        return Path(downloaded_root)


class PairDirectoryDataRootResolver:
    def __init__(self, max_depth: int = 2):
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, but got {max_depth}")
        self.max_depth = int(max_depth)

    @staticmethod
    def _contains_pair_dirs(root: Path, noisy_subdir: str, target_subdir: str) -> bool:
        return (
            root.is_dir()
            and (root / noisy_subdir).exists()
            and (root / target_subdir).exists()
        )

    def find_data_roots(
        self,
        base_root: Path,
        split: Optional[str],
        noisy_subdir: str,
        target_subdir: str,
    ) -> List[Path]:
        if not base_root.exists():
            return []

        candidate_roots: List[Path] = [base_root]
        frontier: List[Path] = [base_root]
        for _ in range(self.max_depth):
            next_frontier: List[Path] = []
            for root in frontier:
                for child in root.iterdir():
                    if child.is_dir():
                        candidate_roots.append(child)
                        next_frontier.append(child)
            frontier = next_frontier

        matches: List[Path] = []
        for root in candidate_roots:
            if split is not None:
                split_root = root / split
                if self._contains_pair_dirs(split_root, noisy_subdir, target_subdir):
                    matches.append(split_root)
            if self._contains_pair_dirs(root, noisy_subdir, target_subdir):
                matches.append(root)

        seen = set()
        unique_matches = []
        for path in matches:
            path_str = str(path)
            if path_str in seen:
                continue
            seen.add(path_str)
            unique_matches.append(path)
        return unique_matches


def build_stem_sample_builder(
    cfg: Optional[Dict[str, Any] | str],
    *,
    strict_index_naming: bool,
):
    default_class_name = (
        "StrictIndexSampleBuilder"
        if strict_index_naming
        else "MatchedNameSampleBuilder"
    )
    sample_builder = _build_component(
        cfg,
        default_class_name=default_class_name,
        required_methods=["build"],
    )
    logger.debug(f"Use sample builder: {sample_builder.__class__.__name__}")
    return sample_builder


def build_stem_downloader(cfg: Optional[Dict[str, Any] | str]):
    downloader = _build_component(
        cfg,
        default_class_name="DefaultSTEMDatasetDownloader",
        required_methods=["download"],
    )
    logger.debug(f"Use downloader: {downloader.__class__.__name__}")
    return downloader


def build_stem_data_root_resolver(cfg: Optional[Dict[str, Any] | str]):
    resolver = _build_component(
        cfg,
        default_class_name="PairDirectoryDataRootResolver",
        required_methods=["find_data_roots"],
    )
    logger.debug(f"Use data root resolver: {resolver.__class__.__name__}")
    return resolver


class BuildSpectrumNMR:
    """
    Convert tokenized NMR JSON into fixed-size numeric arrays for 1H and 13C.

    Input format (per sample, e.g. from CSV "tokenized_input" JSON):
        {
          "1HNMR": [
            [chem_shift, peak_width_token, split_token, "nH", [J1, J2, ...]],
            ...
          ],
          "13CNMR": [c_shift1, c_shift2, ...]
        }

    Output (dict of NumPy arrays and counts):
        {
          "H_nmr":      float32 [seq_len_H1, 4 + j_len],
            # [δ, peakwidth_id, split_id, integral, J*]
          "num_H_peak": int,
          "C_nmr":      float32 [seq_len_C13],
            # δ (ppm), padded/truncated
          "num_C_peak": int,
        }

    Notes:
        - Unknown tokens map to vocab["<unk>"] (you must include it).
        - Peaks beyond the sequence length are truncated; missing slots are zero-padded.
        - Integral like "3H" is parsed as 3; you can add a constant offset if that
            matches your training setup.
    """

    REQUIRED_VOCABS = ("peakwidth", "split")

    def __init__(
        self,
        vocabs: Dict[str, int],
        seq_len_H1: int,
        seq_len_C13: int,
        *,
        j_len: int = 6,
        integral_offset: int = 1,
        unk_token: str = "<unk>",
        dtype: str = "float32",
        num_cpus: int = 1,
    ) -> None:
        self.vocab_peakwidth = dict(vocabs["peakwidth"])
        self.vocab_split = dict(vocabs["split"])
        self.seq_len_H1 = int(seq_len_H1)
        self.seq_len_C13 = int(seq_len_C13)
        self.j_len = int(j_len)
        self.integral_offset = int(integral_offset)
        self.unk_token = unk_token
        self.dtype = np.dtype(dtype)
        self.num_cpus = int(num_cpus)

        if (
            self.unk_token not in self.vocab_peakwidth
            or self.unk_token not in self.vocab_split
        ):
            raise ValueError(
                f"Both vocabs must contain the unknown token '{self.unk_token}'."
            )

    @staticmethod
    def _parse_integral(h_str: Union[str, int, float], offset: int) -> int:
        # Accept "3H" or 3; treat NaN/None as 0
        if h_str is None:
            val = 0
        elif isinstance(h_str, (int, float)):
            val = int(h_str)
        else:
            s = str(h_str).upper().replace("H", "").strip()
            try:
                val = int(float(s))
            except Exception:
                val = 0
        return max(0, val + offset)

    @staticmethod
    def build_one(
        nmrdata: Dict[str, Any],
        vocab_peakwidth: Dict[str, int],
        vocab_split: Dict[str, int],
        seq_len_H1: int,
        seq_len_C13: int,
        j_len: int,
        integral_offset: int,
        unk_token: str,
        dtype: np.dtype,
    ) -> Dict[str, Any]:
        # ----- 1H NMR -----
        Hnmr = nmrdata.get("1HNMR", []) or []
        num_h = len(Hnmr)

        # Allocate [seq_len_H1, 4 + j_len]: [δ, peakwidth_id, split_id, integral,
        # J1..Jj_len]
        H_arr = np.zeros((seq_len_H1, 4 + j_len), dtype=dtype)

        # Fill rows up to seq_len_H1
        limit_h = min(seq_len_H1, num_h)
        for i in range(limit_h):
            peak = Hnmr[i]
            # Expected: [chem_shift (float), peakwidth_token (str), split_token (str),
            # "nH", [J...]]
            chem_shift = float(peak[0])
            peakwidth_tok = str(peak[1])
            split_tok = str(peak[2])
            integral_str = peak[3]
            j_list = (
                peak[4] if len(peak) > 4 and isinstance(peak[4], (list, tuple)) else []
            )

            peakwidth_id = vocab_peakwidth.get(
                peakwidth_tok, vocab_peakwidth[unk_token]
            )
            split_id = vocab_split.get(split_tok, vocab_split[unk_token])
            integral = BuildSpectrumNMR._parse_integral(integral_str, integral_offset)

            row = [chem_shift, float(peakwidth_id), float(split_id), float(integral)]
            if len(j_list) >= j_len:
                row += [float(x) for x in j_list[:j_len]]
            else:
                row += [float(x) for x in j_list] + [0.0] * (j_len - len(j_list))

            H_arr[i, : len(row)] = np.asarray(row, dtype=dtype)

        # ----- 13C NMR -----
        Cnmr = nmrdata.get("13CNMR", []) or []
        num_c = len(Cnmr)
        C_arr = np.zeros((seq_len_C13,), dtype=dtype)
        if num_c > 0:
            C_vals = np.asarray([float(x) for x in Cnmr[:seq_len_C13]], dtype=dtype)
            C_arr[: len(C_vals)] = C_vals

        return {
            "H_nmr": H_arr,  # [seq_len_H1, 4 + j_len] float32
            "num_H_peak": int(num_h),
            "C_nmr": C_arr,  # [seq_len_C13] float32
            "num_C_peak": int(num_c),
        }

    def __call__(
        self, nmr_list: Union[Sequence[Dict[str, Any]], Dict[str, Any]]
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Vectorized/batched conversion with p_tqdm.p_map (or single sample)."""
        if isinstance(nmr_list, (list, tuple)):
            if len(nmr_list) == 0:
                return []
            return p_map(
                BuildSpectrumNMR.build_one,
                nmr_list,
                [self.vocab_peakwidth] * len(nmr_list),
                [self.vocab_split] * len(nmr_list),
                [self.seq_len_H1] * len(nmr_list),
                [self.seq_len_C13] * len(nmr_list),
                [self.j_len] * len(nmr_list),
                [self.integral_offset] * len(nmr_list),
                [self.unk_token] * len(nmr_list),
                [self.dtype] * len(nmr_list),
                num_cpus=self.num_cpus,
                desc="Building spectrums",
                dynamic_ncols=True,
                mininterval=0.2,
            )
        # single sample
        return BuildSpectrumNMR.build_one(
            nmr_list,
            self.vocab_peakwidth,
            self.vocab_split,
            self.seq_len_H1,
            self.seq_len_C13,
            self.j_len,
            self.integral_offset,
            self.unk_token,
            self.dtype,
        )


def _locate_class(class_name: str):
    """Resolve 'pkg.mod.Class' or a bare class name in the current globals()."""
    if "." in class_name:
        mod, cls = class_name.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    return globals()[class_name]
