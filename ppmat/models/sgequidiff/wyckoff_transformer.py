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

"""
Wyckoff and Element Transformer: autoregressive sampling of Wyckoff positions and elements.

"""
import dataclasses
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import ppmat.models.sgequidiff.global_vars as global_vars
from ppmat.models.sgequidiff.constants import (
    NUM_ELEMENTS, NUM_SPACE_GROUPS, MAX_WYCKOFF_SITES,
    lattice_parameter_ranges,
    max_atoms_per_dataset,
)
from ppmat.models.sgequidiff.lattice_sampler import SpaceGroupEncoder
from ppmat.models.sgequidiff.non_equivariant_drift_modules import FourierLinear, Swish


class CustomMultiheadAttention(nn.Layer):
    """Custom MHA matching _qkv_weight/_qkv_bias format from PT weights."""
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 kdim=None, vdim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self._qkv_weight = paddle.create_parameter(
            shape=[3 * embed_dim, self.kdim],
            dtype="float32",
            default_initializer=nn.initializer.XavierUniform(),
        )
        self._qkv_bias = paddle.create_parameter(
            shape=[3 * embed_dim],
            dtype="float32",
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, need_weights=False,
                attn_mask=None, key_padding_mask=None):
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        embed_dim = self.embed_dim

        W_q = self._qkv_weight[:embed_dim]
        W_k = self._qkv_weight[embed_dim:2*embed_dim]
        W_v = self._qkv_weight[2*embed_dim:]
        b_q = self._qkv_bias[:embed_dim]
        b_k = self._qkv_bias[embed_dim:2*embed_dim]
        b_v = self._qkv_bias[2*embed_dim:]

        q = paddle.matmul(query, W_q) + b_q
        k = paddle.matmul(key, W_k) + b_k
        v = paddle.matmul(value, W_v) + b_v

        q = q.reshape([batch_size, seq_q, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        k = k.reshape([batch_size, seq_k, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        v = v.reshape([batch_size, seq_k, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

        scale = math.sqrt(self.head_dim)
        attn_weights = paddle.matmul(q, k.transpose([0, 1, 3, 2])) / scale

        if attn_mask is not None:
            if attn_mask.dtype == paddle.bool:
                attn_weights = paddle.where(
                    attn_mask.unsqueeze(1) if attn_mask.dim() == 3 else attn_mask,
                    paddle.full_like(attn_weights, float('-inf')),
                    attn_weights,
                )
            else:
                attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = paddle.where(
                mask,
                paddle.full_like(attn_weights, float('-inf')),
                attn_weights,
            )

        attn_weights = F.softmax(attn_weights, axis=-1)

        attn_weights = paddle.where(
            paddle.isnan(attn_weights),
            paddle.zeros_like(attn_weights),
            attn_weights,
        )

        attn_output = paddle.matmul(attn_weights, v)
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape(
            [batch_size, seq_q, self.embed_dim]
        )

        output = self.out_proj(attn_output)

        if need_weights:
            avg_weights = attn_weights.mean(axis=1)
            return output, avg_weights

        return output, None

class SpaceGroupAndLatticeEncoder(nn.Layer):
    """Encode space group index and lattice parameters."""
    def __init__(self, hidden_dim: int, dataset_name: str):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset_name = dataset_name
        lattice_param_range_dict = lattice_parameter_ranges[self.dataset_name]
        self.lattice_length_range = (
            lattice_param_range_dict["max_lattice_length"]
            - lattice_param_range_dict["min_lattice_length"]
        )
        self.lattice_angle_range = (
            lattice_param_range_dict["max_lattice_angle"]
            - lattice_param_range_dict["min_lattice_angle"]
        )
        self.space_group_encoder = SpaceGroupEncoder(
            hidden_channels=self.hidden_dim,
            space_group_embedding_dim=math.floor(self.hidden_dim / 2),
        )
        self.lattice_encoder = FourierLinear(
            input_dim=6,
            num_fourier_frequencies=64,
            scale=1.0,
            output_dim=math.ceil(self.hidden_dim / 2),
            use_bias=True,
        )

    def forward(self, space_group_indices, lattice_lengths, lattice_angles):
        normed_lattice_params = paddle.concat(
            [
                lattice_lengths / self.lattice_length_range,
                lattice_angles / self.lattice_angle_range,
            ],
            axis=-1,
        )
        emb = paddle.concat(
            [
                self.space_group_encoder(space_group_indices),
                self.lattice_encoder(normed_lattice_params),
            ],
            axis=-1,
        )
        return emb

class TransformerDecoderLayer(nn.Layer):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.mha = CustomMultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x, attn_mask=None, key_padding_mask=None, **kwargs):
        x_norm = self.layernorm(x)
        attn_out = self.mha(
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        x = x + attn_out
        x = self.dropout(x)
        x = x + self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)

@dataclasses.dataclass
class WyckoffElementTransformerConfig:
    hidden_dim: int
    dataset_name: str
    num_heads: int = 4
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0

class WyckoffElementTransformer(nn.Layer):
    """Matches original PT code structure and weight format exactly."""
    def __init__(self, config: WyckoffElementTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.dropout_rate = config.dropout_rate

        assert self.hidden_dim % 2 == 0
        self.wyckoff_dim = int(self.hidden_dim / 2)
        self.element_dim = int(self.hidden_dim / 2)

        self.wyckoff_emb = nn.Linear(
            global_vars.embedding_tools.wyckoff_embedding_length, self.wyckoff_dim
        )
        self.element_emb = nn.Linear(
            global_vars.embedding_tools.element_embedding_length, self.element_dim
        )

        nn.initializer.XavierUniform()(self.element_emb.weight)

        self.stop_key = paddle.create_parameter(
            shape=[1, 1, self.wyckoff_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(),
        )
        self.seed_wyckoff_embedding = paddle.create_parameter(
            shape=[1, 1, self.wyckoff_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(),
        )
        self.seed_element_embedding = paddle.create_parameter(
            shape=[1, 1, self.element_dim],
            dtype="float32",
            default_initializer=nn.initializer.Normal(),
        )

        self.space_group_and_lattice_emb = SpaceGroupAndLatticeEncoder(
            self.hidden_dim, config.dataset_name
        )

        self.atom_embedder = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.wyckoff_dim + self.element_dim,
                self.hidden_dim,
            ),
            Swish(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Swish(),
        )

        self.hidden_layers = nn.LayerList([
            TransformerDecoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_hidden_layers=self.num_hidden_layers,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_hidden_layers)
        ])

        self.wyckoff_mha = CustomMultiheadAttention(
            embed_dim=int(self.hidden_dim / 2),
            num_heads=1,
            kdim=self.wyckoff_dim,
            vdim=self.wyckoff_dim,
        )

        d = int(self.hidden_dim / 2) + self.wyckoff_dim
        self.mix_xtal_and_wyckoff_mlp = nn.Sequential(
            nn.Linear(d, int(self.hidden_dim / 2)),
            Swish(),
            nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2)),
            Swish(),
            nn.Linear(int(self.hidden_dim / 2), int(self.hidden_dim / 2)),
            Swish(),
        )

        self.element_keys_mlp = nn.Sequential(
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
            nn.Linear(self.element_dim, self.element_dim),
            Swish(),
        )

        self.element_mha = CustomMultiheadAttention(
            embed_dim=int(self.hidden_dim / 2),
            num_heads=1,
            kdim=self.element_dim,
            vdim=self.element_dim,
        )

        self.register_buffer(
            "valid_wyckoff_positions_mask",
            paddle.zeros([NUM_SPACE_GROUPS, MAX_WYCKOFF_SITES], dtype="bool"),
        )
        self.register_buffer(
            "zero_dimensional_wyckoff_mask",
            paddle.zeros([NUM_SPACE_GROUPS, MAX_WYCKOFF_SITES], dtype="bool"),
        )

    @paddle.no_grad()
    def sample_and_log_prob(
        self,
        lattice_lengths,
        lattice_angles,
        space_group_indices,
        temperature: float = 1.0,
    ):
        """Autoregressively sample Wyckoff positions and elements."""
        assert temperature > 0.0
        max_allowed_atoms = max_atoms_per_dataset[self.config.dataset_name]
        n_crystals = space_group_indices.shape[0]

        global_context = self.space_group_and_lattice_emb(
            space_group_indices, lattice_lengths, lattice_angles
        )[:, None, :]

        padded_wyckoff_embeddings = self.seed_wyckoff_embedding.expand(
            [n_crystals, 1, -1]
        )
        padded_element_embeddings = self.seed_element_embedding.expand(
            [n_crystals, 1, -1]
        )

        atom_tokens = paddle.concat(
            [
                global_context,
                padded_wyckoff_embeddings,
                padded_element_embeddings,
            ],
            axis=-1,
        )

        wyckoff_exists = self.valid_wyckoff_positions_mask[space_group_indices]

        _xtal_indices_list = []
        _existing_wyckoff_indices_list = []
        for b in range(n_crystals):
            sg_idx = int(space_group_indices[b])
            for wp_idx in range(MAX_WYCKOFF_SITES):
                if wyckoff_exists[b, wp_idx]:
                    _xtal_indices_list.append(b)
                    _existing_wyckoff_indices_list.append(wp_idx)

        if len(_xtal_indices_list) > 0:
            _xtal_indices = paddle.to_tensor(_xtal_indices_list, dtype="int64")
            _existing_wyckoff_indices = paddle.to_tensor(
                _existing_wyckoff_indices_list, dtype="int64"
            )
            wyckoff_keys = self.wyckoff_emb(
                global_vars.embedding_tools.get_wyckoff_embedding(
                    wyckoff_index=_existing_wyckoff_indices,
                    space_group_index=space_group_indices[_xtal_indices],
                )
            )
        else:
            wyckoff_keys = paddle.zeros([0, self.wyckoff_dim])

        padded_wyckoff_keys = paddle.zeros(
            [n_crystals, MAX_WYCKOFF_SITES, self.wyckoff_dim]
        )
        if len(_xtal_indices_list) > 0:
            padded_wyckoff_keys[_xtal_indices, _existing_wyckoff_indices] = wyckoff_keys

        padded_wyckoff_keys = paddle.concat(
            [self.stop_key.expand([n_crystals, 1, -1]), padded_wyckoff_keys],
            axis=1,
        )

        wyckoff_padding_mask = paddle.concat(
            [
                paddle.zeros([n_crystals, 1], dtype="bool"),
                ~wyckoff_exists,
            ],
            axis=1,
        )

        _element_keys = self.element_keys_mlp(
            self.element_emb(
                global_vars.embedding_tools.get_element_embedding(
                    atomic_number=1 + paddle.arange(NUM_ELEMENTS, dtype="int64")
                )
            )
        )[None, ...]

        wyckoff_attn_mask = paddle.zeros(
            [n_crystals, 1, 1 + MAX_WYCKOFF_SITES], dtype="bool"
        )
        wyckoff_attn_mask[:, :, 0] = True

        crystal_is_complete = paddle.zeros([n_crystals], dtype="bool")
        n_asu_atoms_per_xtal = paddle.zeros([n_crystals], dtype="int64")
        padded_wyckoff_indices = paddle.full(
            [n_crystals, max_allowed_atoms], fill_value=-1, dtype="int64"
        )
        padded_element_indices = paddle.full(
            [n_crystals, max_allowed_atoms], fill_value=-1, dtype="int64"
        )
        padded_wyckoff_probs = paddle.full(
            [n_crystals, max_allowed_atoms], fill_value=-1.0
        )
        padded_element_probs = paddle.full(
            [n_crystals, max_allowed_atoms], fill_value=-1.0
        )
        termination_probs = paddle.zeros([n_crystals])

        iteration = 0
        arange_n_crystals = paddle.arange(n_crystals, dtype="int64")

        while iteration < max_allowed_atoms and not crystal_is_complete.all():
            atom_tokens = self.atom_embedder(atom_tokens)

            seq_len = atom_tokens.shape[1]
            causal_mask = paddle.triu(
                paddle.ones([seq_len, seq_len], dtype="bool"), diagonal=1
            )

            for layer in self.hidden_layers:
                atom_tokens = layer(
                    atom_tokens,
                    attn_mask=causal_mask,
                )

            incomplete_mask = ~crystal_is_complete
            n_incomplete = int(incomplete_mask.sum())

            last_pos = n_asu_atoms_per_xtal[incomplete_mask]
            incomplete_indices = arange_n_crystals[incomplete_mask]

            atom_tokens_last = atom_tokens[incomplete_mask]
            _idx = paddle.arange(n_incomplete, dtype="int64")
            atom_tokens_flat = atom_tokens_last[_idx, last_pos]

            atom_z_wyckoff = atom_tokens_flat[:, :self.wyckoff_dim]
            atom_z_element = atom_tokens_flat[:, self.wyckoff_dim:]

            wyckoff_keys_batch = padded_wyckoff_keys[incomplete_mask]
            wyckoff_padding_batch = wyckoff_padding_mask[incomplete_mask]
            wyckoff_attn_batch = wyckoff_attn_mask

            wyckoff_and_stop_probs = self.wyckoff_mha(
                query=atom_z_wyckoff[:, None, :],
                key=wyckoff_keys_batch,
                value=wyckoff_keys_batch,
                need_weights=True,
                key_padding_mask=wyckoff_padding_batch,
                attn_mask=wyckoff_attn_batch,
            )[1]

            if temperature != 1.0:
                wyckoff_and_stop_probs = F.softmax(
                    paddle.log(wyckoff_and_stop_probs + 1e-12) / temperature, axis=-1
                )

            wyckoff_or_stop_sample = paddle.multinomial(
                wyckoff_and_stop_probs.squeeze(1), num_samples=1
            ).squeeze(-1)

            sampled_stop_token = (wyckoff_or_stop_sample == 0)
            sampled_wyckoff_indices = wyckoff_or_stop_sample[~sampled_stop_token] - 1

            if (~sampled_stop_token).sum() > 0:
                sampled_wyckoff_probs = paddle.take_along_axis(
                    wyckoff_and_stop_probs[~sampled_stop_token].squeeze(1),
                    (1 + sampled_wyckoff_indices[:, None]),
                    axis=1,
                ).squeeze(-1)
            else:
                sampled_wyckoff_probs = paddle.zeros([0])

            termination_probs[incomplete_mask] = (
                sampled_stop_token.cast("float32")
                * wyckoff_and_stop_probs[:, 0, 0]
            )

            idxs_of_xtals_to_update = arange_n_crystals[incomplete_mask][~sampled_stop_token]
            if idxs_of_xtals_to_update.shape[0] > 0:
                padded_wyckoff_indices[idxs_of_xtals_to_update, iteration] = sampled_wyckoff_indices
                padded_wyckoff_probs[idxs_of_xtals_to_update, iteration] = sampled_wyckoff_probs
                n_asu_atoms_per_xtal[idxs_of_xtals_to_update] += 1

            iteration += 1
            crystal_is_complete[incomplete_mask] = sampled_stop_token

            if (~crystal_is_complete).sum() > 0:
                if sampled_wyckoff_indices.shape[0] > 0:
                    sampled_wyckoff_embeddings = self.wyckoff_emb(
                        global_vars.embedding_tools.get_wyckoff_embedding(
                            wyckoff_index=sampled_wyckoff_indices,
                            space_group_index=space_group_indices[idxs_of_xtals_to_update],
                        )
                    )
                else:
                    sampled_wyckoff_embeddings = paddle.zeros([0, self.wyckoff_dim])

                if sampled_wyckoff_indices.shape[0] > 0:
                    atom_z_element_mixed = self.mix_xtal_and_wyckoff_mlp(
                        paddle.concat(
                            [atom_z_element[~sampled_stop_token], sampled_wyckoff_embeddings],
                            axis=-1,
                        )
                    )
                else:
                    atom_z_element_mixed = paddle.zeros([0, int(self.hidden_dim / 2)])

                element_keys = _element_keys.expand(
                    [atom_z_element_mixed.shape[0], -1, -1]
                )

                element_attn_mask = paddle.zeros(
                    [atom_z_element_mixed.shape[0], 1, NUM_ELEMENTS],
                    dtype="bool",
                )

                element_probs = self.element_mha(
                    query=atom_z_element_mixed[:, None, :],
                    key=element_keys,
                    value=element_keys,
                    need_weights=True,
                    attn_mask=element_attn_mask,
                )[1]

                if temperature != 1.0:
                    element_probs = F.softmax(
                        paddle.log(element_probs + 1e-12) / temperature, axis=-1
                    )

                sampled_element_indices = paddle.multinomial(
                    element_probs.squeeze(1), num_samples=1
                ).squeeze(-1)
                sampled_element_probs = paddle.take_along_axis(
                    element_probs.squeeze(1),
                    sampled_element_indices[:, None],
                    axis=1,
                ).squeeze(-1)

                padded_element_indices[idxs_of_xtals_to_update, iteration - 1] = sampled_element_indices
                padded_element_probs[idxs_of_xtals_to_update, iteration - 1] = sampled_element_probs

                sampled_element_embeddings = self.element_emb(
                    global_vars.embedding_tools.get_element_embedding(
                        atomic_number=1 + sampled_element_indices
                    )
                )

                padded_sampled_wyckoff = paddle.zeros([n_crystals, self.wyckoff_dim])
                padded_sampled_element = paddle.zeros([n_crystals, self.element_dim])
                padded_sampled_wyckoff[idxs_of_xtals_to_update] = sampled_wyckoff_embeddings
                padded_sampled_element[idxs_of_xtals_to_update] = sampled_element_embeddings

                padded_wyckoff_embeddings = paddle.concat(
                    [
                        padded_wyckoff_embeddings,
                        padded_sampled_wyckoff[:, None, :],
                    ],
                    axis=1,
                )
                padded_element_embeddings = paddle.concat(
                    [
                        padded_element_embeddings,
                        padded_sampled_element[:, None, :],
                    ],
                    axis=1,
                )

                incomplete_mask_new = ~crystal_is_complete
                if incomplete_mask_new.sum() > 0:
                    global_context_expanded = global_context[incomplete_mask_new].expand(
                        [-1, 1 + iteration, -1]
                    )
                    atom_tokens = paddle.concat(
                        [
                            global_context_expanded,
                            padded_wyckoff_embeddings[incomplete_mask_new],
                            padded_element_embeddings[incomplete_mask_new],
                        ],
                        axis=-1,
                    )

        atom_mask = (
            paddle.arange(max_allowed_atoms, dtype="int64")[None, :]
            < n_asu_atoms_per_xtal[:, None]
        )

        element_indices = padded_element_indices[atom_mask]
        wyckoff_indices = padded_wyckoff_indices[atom_mask]
        _element_probs = padded_element_probs[atom_mask]
        _wyckoff_probs = padded_wyckoff_probs[atom_mask]

        elements_log_prob = paddle.log(_element_probs + 1e-12)
        wyckoffs_log_prob = paddle.log(_wyckoff_probs + 1e-12)
        termination_log_prob = paddle.log(termination_probs + 1e-12)

        return (
            element_indices,
            wyckoff_indices,
            n_asu_atoms_per_xtal,
            elements_log_prob,
            wyckoffs_log_prob,
            termination_log_prob,
        )