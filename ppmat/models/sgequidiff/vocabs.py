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

"""SGEquiDiff embedding tables built from the registered vocabulary.

The element / space-group / Wyckoff embedding tables come from the
``sgequidiff`` vocabulary (see ``ppmat.vocab.build_vocab`` and the
``Vocabulary:`` config section). Consumers should call
``build_embedding_tools()`` and pass the instance to downstream modules.

Vocabulary roles (content derived from the upstream SGEquiDiff repository;
vector payloads are byte-equivalent to the upstream values, conversion script
lives under the sgequidiff history workspace):

- ``element``: atomic numbers 0-100 -> 92-dim embeddings
  (upstream ``data/init_tokens/cgcnn_atom_init.json``).
- ``space_group``: space-group numbers 1-230 -> 62-dim embeddings
  (upstream ``data/init_tokens/space_group_features/space_group_embeddings_62dim.json``).
- ``wyckoff``: per-space-group letter -> 231-dim embeddings
  (upstream ``data/init_tokens/wyckoff_features/wyckoff_embeddings_231dim.json``).
- ``asu_sites``: ASU Wyckoff-site geometry payload under ``data``
  (upstream ``data/wyckoff_positions/clean_wyckoffs_in_asu_v6.json``),
  consumed by ``wyckoff_geometry.WyckoffGeometry``.
"""

from __future__ import annotations

import paddle

from ppmat.models.sgequidiff.sgequidiff_meta import ELEMENT_ENCODING_SIZE
from ppmat.models.sgequidiff.sgequidiff_meta import MAX_WYCKOFF_POSITIONS
from ppmat.models.sgequidiff.sgequidiff_meta import NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS
from ppmat.vocab import build_vocab

VOCAB_NAME = "sgequidiff"


class EmbeddingTools:
    """Element / space-group / Wyckoff embedding tables.

    Construct explicitly via ``EmbeddingTools(vocab)`` or
    ``build_embedding_tools()``. No global singleton.
    """

    @paddle.no_grad()
    def __init__(self, vocab: dict):
        element_vectors = vocab["element"]["vectors"][: ELEMENT_ENCODING_SIZE + 1]
        self.element_embedding_length = len(element_vectors[0])
        self.element_embedding_tensor = paddle.to_tensor(
            element_vectors,
            dtype=paddle.float32,
        )

        space_group_vectors = vocab["space_group"]["vectors"]
        self.space_group_embedding_length = len(space_group_vectors[0])
        self.space_group_embedding_tensor = paddle.to_tensor(
            space_group_vectors,
            dtype=paddle.float32,
        )

        wyckoff_dict = vocab["wyckoff"]["vectors"]
        self.wyckoff_embedding_length = len(wyckoff_dict["1"]["a"])

        wyckoff_emb_list = []
        n_wyckoffs_list = []
        for sg_num in range(1, NUM_CRYSTALLOGRAPHIC_SPACE_GROUPS + 1):
            wyckoff_dict_of_sg = wyckoff_dict[str(sg_num)]
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

    def get_space_group_embedding(
        self, space_group_index: paddle.Tensor
    ) -> paddle.Tensor:
        """Get space group embedding."""
        assert space_group_index.dtype == paddle.int64
        return self.space_group_embedding_tensor[space_group_index]

    @paddle.no_grad()
    def get_element_embedding(self, atomic_number: paddle.Tensor) -> paddle.Tensor:
        """Get element embedding."""
        assert atomic_number.dtype == paddle.int64
        return self.element_embedding_tensor[atomic_number]

    @paddle.no_grad()
    def get_wyckoff_embedding(
        self,
        wyckoff_index: paddle.Tensor,
        space_group_index: paddle.Tensor,
    ) -> paddle.Tensor:
        """Get Wyckoff embedding by index and space group."""
        valid_mask = (
            self.n_wyckoffs_per_space_group[space_group_index] > wyckoff_index
        )
        assert valid_mask.all().item(), "Invalid space group-Wyckoff index pairs"
        return self.wyckoff_embedding_tensor[space_group_index, wyckoff_index, :]


def build_embedding_tools(vocab: dict | None = None) -> EmbeddingTools:
    """Build EmbeddingTools from the registered vocabulary."""
    if vocab is None:
        vocab = build_vocab(VOCAB_NAME)
    return EmbeddingTools(vocab)
