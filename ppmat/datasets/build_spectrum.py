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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from p_tqdm import p_map

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
