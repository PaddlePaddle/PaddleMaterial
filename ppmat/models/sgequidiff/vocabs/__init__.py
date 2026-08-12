# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bundled SGEquiDiff static resources (element / space-group / Wyckoff embeddings).

This package owns the four resource JSON files and provides explicit construction
of embedding tables. Consumers should call `build_embedding_tools()` with explicit
paths (or use defaults) and pass the instance to downstream modules.
"""
import json
from pathlib import Path
from typing import Optional

import paddle
import paddle.nn.functional as F

from ppmat.models.sgequidiff.sgequidiff_meta import MAX_WYCKOFF_POSITIONS
from ppmat.models.sgequidiff.sgequidiff_meta import NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
from ppmat.utils.asu_dataset_meta import ELEMENT_ENCODING_SIZE

RESOURCE_DIR = Path(__file__).resolve().parent

_DEFAULT_SPACE_GROUP_EMBEDDING_JSON = (
    "init_tokens/space_group_features/space_group_embeddings_62dim.json"
)
_DEFAULT_ELEMENT_EMBEDDING_JSON = "cgcnn_atom_init.json"
_DEFAULT_WYCKOFF_EMBEDDING_JSON = (
    "init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json"
)


class EmbeddingTools:
    """Element/space-group/Wyckoff embedding tables.

    Construct explicitly via `EmbeddingTools(...)` or `build_embedding_tools()`.
    No global singleton.
    """

    @paddle.no_grad()
    def __init__(
        self,
        space_group_embedding_json_path: Optional[str] = None,
        element_embedding_json_path: Optional[str] = None,
        wyckoff_embedding_json_path: Optional[str] = None,
    ):
        self.space_group_embedding_dict = None
        self.element_embedding_dict = None
        self.wyckoff_embedding_dict = None

        def _resolve_json(json_path: str) -> Path:
            return Path(RESOURCE_DIR / json_path)

        if space_group_embedding_json_path is not None:
            fp = _resolve_json(space_group_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.space_group_embedding_dict = json.load(file)
            self.space_group_embedding_length = len(
                self.space_group_embedding_dict["1"]
            )
            self.space_group_embedding_tensor = paddle.to_tensor(
                [
                    self.space_group_embedding_dict[str(sg_num)]
                    for sg_num in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1)
                ],
                dtype=paddle.float32,
            )
        else:
            self.space_group_embedding_length = NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS

        if element_embedding_json_path is not None:
            fp = _resolve_json(element_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.element_embedding_dict = json.load(file)
            self.element_embedding_length = len(self.element_embedding_dict["0"])
            self.element_embedding_tensor = paddle.to_tensor(
                [
                    self.element_embedding_dict[str(atomic_number)]
                    for atomic_number in range(ELEMENT_ENCODING_SIZE + 1)
                ],
                dtype=paddle.float32,
            )
        else:
            self.element_embedding_length = ELEMENT_ENCODING_SIZE

        if wyckoff_embedding_json_path is not None:
            fp = _resolve_json(wyckoff_embedding_json_path).as_posix()
            with open(fp, "r") as file:
                self.wyckoff_embedding_dict = json.load(file)
            self.wyckoff_embedding_length = len(self.wyckoff_embedding_dict["1"]["a"])

            wyckoff_emb_list = []
            n_wyckoffs_list = []
            for sg_num in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1):
                wyckoff_dict_of_sg = self.wyckoff_embedding_dict[str(sg_num)]
                letters = list(wyckoff_dict_of_sg.keys())

                wyckoff_ascii = [ord(letter) for letter in letters]
                wyckoff_idxs = [
                    ai - 97 if ai >= 97 else ai - 65 + 26 for ai in wyckoff_ascii
                ]
                sorted_letters = [
                    letter
                    for letter, _ in sorted(
                        zip(letters, wyckoff_idxs), key=lambda pair: pair[1]
                    )
                ]

                emb_array = paddle.to_tensor(
                    [wyckoff_dict_of_sg[letter] for letter in sorted_letters],
                    dtype=paddle.float32,
                )
                padding = paddle.zeros(
                    [
                        MAX_WYCKOFF_POSITIONS - len(letters),
                        self.wyckoff_embedding_length,
                    ]
                )
                wyckoff_emb_list.append(paddle.concat([emb_array, padding], axis=0))
                n_wyckoffs_list.append(len(letters))

            self.wyckoff_embedding_tensor = paddle.stack(wyckoff_emb_list, axis=0)
            self.n_wyckoffs_per_space_group = paddle.to_tensor(
                n_wyckoffs_list, dtype=paddle.int64
            )
        else:
            self.wyckoff_embedding_length = MAX_WYCKOFF_POSITIONS

    def get_space_group_embedding(
        self, space_group_index: paddle.Tensor
    ) -> paddle.Tensor:
        """Get space group embedding."""
        assert space_group_index.dtype == paddle.int64
        if self.space_group_embedding_dict is None:
            return F.one_hot(space_group_index, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS).cast(
                paddle.float32
            )
        else:
            return self.space_group_embedding_tensor[space_group_index]

    @paddle.no_grad()
    def get_element_embedding(self, atomic_number: paddle.Tensor) -> paddle.Tensor:
        """Get element embedding."""
        assert atomic_number.dtype == paddle.int64
        if self.element_embedding_dict is None:
            return F.one_hot(atomic_number - 1, ELEMENT_ENCODING_SIZE).cast(
                paddle.float32
            )
        else:
            return self.element_embedding_tensor[atomic_number]

    @paddle.no_grad()
    def get_wyckoff_embedding(
        self,
        wyckoff_index: paddle.Tensor,
        space_group_index: paddle.Tensor,
    ) -> paddle.Tensor:
        """Get Wyckoff embedding by index and space group."""
        if self.wyckoff_embedding_dict is None:
            return F.one_hot(wyckoff_index, MAX_WYCKOFF_POSITIONS).cast(paddle.float32)
        else:
            valid_mask = (
                self.n_wyckoffs_per_space_group[space_group_index] > wyckoff_index
            )
            assert valid_mask.all().item(), "Invalid space group-Wyckoff index pairs"
            return self.wyckoff_embedding_tensor[space_group_index, wyckoff_index, :]


def build_embedding_tools(
    space_group_embedding_json_path: Optional[
        str
    ] = _DEFAULT_SPACE_GROUP_EMBEDDING_JSON,
    element_embedding_json_path: Optional[str] = _DEFAULT_ELEMENT_EMBEDDING_JSON,
    wyckoff_embedding_json_path: Optional[str] = _DEFAULT_WYCKOFF_EMBEDDING_JSON,
) -> EmbeddingTools:
    """Build EmbeddingTools with explicit paths (defaults to bundled resources)."""
    return EmbeddingTools(
        space_group_embedding_json_path=space_group_embedding_json_path,
        element_embedding_json_path=element_embedding_json_path,
        wyckoff_embedding_json_path=wyckoff_embedding_json_path,
    )
