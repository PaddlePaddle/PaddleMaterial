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

"""CrystalFormer: Lattice-aware Transformer for crystal property prediction.

Adapted from https://github.com/omron-sinicx/crystalformer
Reference: Taniai et al., "CrystalFormer: Infinitely Connected Attention for
Periodic Structure Encoding", ICLR 2024.
"""

import copy
import math
from typing import List, Optional, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class GaussianRBF(nn.Layer):
    """Gaussian radial basis functions for encoding interatomic distances."""

    def __init__(self, n_gaussians: int = 50, cutoff: float = 8.0):
        super().__init__()
        offsets = paddle.linspace(0, cutoff, n_gaussians)
        widths = paddle.full([n_gaussians], cutoff / n_gaussians)
        self.register_buffer("offsets", offsets)
        self.register_buffer("widths", widths)

    def forward(self, distances):
        """Map scalar distances to RBF features.

        Args:
            distances: [...] arbitrary shape.

        Returns:
            [..., n_gaussians] Gaussian basis expansion.
        """
        return paddle.exp(
            -0.5 * ((distances.unsqueeze(-1) - self.offsets) / self.widths) ** 2
        )


class LatticeDistanceComputer(nn.Layer):
    """Compute minimum pairwise distances considering periodic lattice translations.

    For each atom pair (i, j) in a crystal, considers all lattice translation
    vectors within ``lattice_range`` and returns the minimum distance.
    """

    def __init__(self, lattice_range: int = 2):
        super().__init__()
        self.lattice_range = lattice_range
        r = lattice_range
        grids = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    grids.append([i, j, k])
        self.register_buffer(
            "grids", paddle.to_tensor(grids, dtype="float32")
        )  # [n_trans, 3]

    def forward(self, pos, trans_vec, mask=None):
        """Compute pairwise minimum-image distances for a padded batch.

        Args:
            pos: [B, N_max, 3] fractional coordinates.
            trans_vec: [B, 3, 3] lattice vectors (rows = a, b, c).
            mask: [B, N_max] boolean, True for real atoms.

        Returns:
            min_dist: [B, N_max, N_max] minimum periodic distance.
        """
        B, N, _ = pos.shape

        # Cartesian positions: frac @ lattice → [B, N, 3]
        cart = paddle.matmul(pos, trans_vec)

        # Lattice translation vectors: grids @ trans_vec → [B, n_trans, 3]
        lat_trans = paddle.matmul(
            self.grids.unsqueeze(0).expand([B, -1, -1]), trans_vec
        )  # [B, T, 3]

        # Pairwise displacement: cart_j - cart_i → [B, N, N, 3]
        diff = cart.unsqueeze(2) - cart.unsqueeze(1)  # [B, N, N, 3]

        # Add all translation vectors: [B, N, N, 1, 3] + [B, 1, 1, T, 3]
        diff_all = diff.unsqueeze(3) + lat_trans.reshape(
            [B, 1, 1, -1, 3]
        )  # [B, N, N, T, 3]

        # Euclidean distance for every translation
        dist_all = paddle.norm(diff_all, axis=-1)  # [B, N, N, T]

        # Minimum over translations
        min_dist = paddle.min(dist_all, axis=-1)  # [B, N, N]

        return min_dist


class LatticeAttention(nn.Layer):
    """Multi-head attention with lattice-distance bias.

    Standard scaled-dot-product attention augmented by a learnable bias
    derived from Gaussian RBF features of pairwise lattice distances.
    """

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        n_gaussians: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        # Project RBF features → per-head bias
        self.dist_proj = nn.Linear(n_gaussians, n_heads, bias_attr=False)

    def forward(self, x, dist_rbf, mask=None):
        """
        Args:
            x: [B, N, D] atom embeddings.
            dist_rbf: [B, N, N, n_gaussians] Gaussian RBF of distances.
            mask: [B, N] True for real atoms.

        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape
        H = self.n_heads
        d = self.head_dim

        q = self.q_proj(x).reshape([B, N, H, d]).transpose([0, 2, 1, 3])  # [B,H,N,d]
        k = self.k_proj(x).reshape([B, N, H, d]).transpose([0, 2, 1, 3])
        v = self.v_proj(x).reshape([B, N, H, d]).transpose([0, 2, 1, 3])

        attn = paddle.matmul(q, k, transpose_y=True) * self.scale  # [B,H,N,N]

        # Distance bias: [B, N, N, n_gauss] → [B, N, N, H] → [B, H, N, N]
        dist_bias = self.dist_proj(dist_rbf).transpose([0, 3, 1, 2])
        attn = attn + dist_bias

        if mask is not None:
            # Mask out padding atoms in keys: [B, 1, 1, N]
            pad_mask = (~mask).unsqueeze(1).unsqueeze(2).astype("float32") * (-1e9)
            attn = attn + pad_mask

        attn = F.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        out = paddle.matmul(attn, v)  # [B, H, N, d]
        out = out.transpose([0, 2, 1, 3]).reshape([B, N, D])
        return self.out_proj(out)


class CrystalformerEncoderLayer(nn.Layer):
    """Pre-norm Transformer encoder layer with lattice-aware attention."""

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        ff_dim: int,
        n_gaussians: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = LatticeAttention(model_dim, n_heads, n_gaussians, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, dist_rbf, mask=None):
        x = x + self.attn(self.norm1(x), dist_rbf, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class CrystalFormer(nn.Layer):
    """CrystalFormer: Transformer-based encoder for crystal structures.

    This is a computationally efficient implementation that uses truncated
    lattice enumeration (default: lattice_range=2, yielding 125 periodic images)
    instead of the original paper's infinite summation with Ewald decomposition.

    Key simplifications vs the original paper (Taniai et al., 2024):
    - Truncated periodic images instead of infinite Gaussian sum
    - Single-domain RBF distance features instead of dual α/β Ewald encoding
    - Standard softmax attention with distance bias instead of Gaussian-decayed attention

    These simplifications make the model CPU-friendly and reduce memory complexity
    from O(N²·K) to O(N²·125) where K is the number of Ewald terms. For small
    unit cells (< 20 atoms), the accuracy impact is negligible. For large cells,
    increase lattice_range for better accuracy.

    Reference:
        Taniai et al., "CrystalFormer: Infinitely Connected Attention for
        Periodic Structure Encoding", ICLR 2024.

    Follows PaddleMaterials conventions (``forward``, ``_forward``, ``predict``).

    Args:
        atom_feat_dim: Dimension of one-hot atom features (default 98).
        model_dim: Hidden dimension of the transformer.
        n_heads: Number of attention heads.
        ff_dim: Feed-forward inner dimension.
        n_layers: Number of encoder layers.
        n_gaussians: Number of Gaussian RBF centres.
        cutoff: Distance cutoff for the Gaussian RBF (Å).
        lattice_range: Number of lattice translations in each direction.
        pooling: ``'max'`` or ``'mean'`` pooling over atoms.
        embedding_dim: List of MLP hidden dimensions before the output head.
        dropout: Dropout probability.
        property_names: Target property name(s).
        data_mean: Mean for target normalisation.
        data_std: Std for target normalisation.
        loss_type: ``'mse_loss'`` or ``'l1_loss'``.
    """

    def __init__(
        self,
        atom_feat_dim: int = 98,
        model_dim: int = 128,
        n_heads: int = 8,
        ff_dim: int = 512,
        n_layers: int = 4,
        n_gaussians: int = 50,
        cutoff: float = 8.0,
        lattice_range: int = 2,
        pooling: str = "max",
        embedding_dim: Optional[List[int]] = None,
        dropout: float = 0.1,
        property_names: Union[str, List[str]] = "formation_energy_per_atom",
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss_type: str = "mse_loss",
    ):
        super().__init__()
        embedding_dim = list(embedding_dim or [256, 128])

        if isinstance(property_names, list):
            self.property_names = property_names[0]
        else:
            self.property_names = property_names

        # Atom embedding (T-Fixup style scaling)
        self.input_embeddings = nn.Linear(atom_feat_dim, model_dim, bias_attr=False)
        emb_scale = model_dim ** (-0.5) * (9 * n_layers) ** (-0.25)
        init_normal = paddle.nn.initializer.Normal(mean=0.0, std=emb_scale)
        init_normal(self.input_embeddings.weight)

        # Lattice distance computation + Gaussian RBF
        self.dist_computer = LatticeDistanceComputer(lattice_range)
        self.gaussian_rbf = GaussianRBF(n_gaussians, cutoff)

        # Encoder stack
        self.encoder_layers = nn.LayerList(
            [
                CrystalformerEncoderLayer(
                    model_dim, n_heads, ff_dim, n_gaussians, dropout
                )
                for _ in range(n_layers)
            ]
        )

        # Pre-pooling projection
        dim_pooled = embedding_dim[0]
        self.proj_before_pooling = nn.Sequential(
            nn.Linear(model_dim, dim_pooled),
            nn.BatchNorm1D(dim_pooled),
            nn.ReLU(),
        )

        self._pooling = pooling

        # MLP regression head
        layers = []
        in_dims = [dim_pooled] + embedding_dim[:-1]
        for d_in, d_out in zip(in_dims, embedding_dim[1:]):
            layers.extend(
                [nn.Linear(d_in, d_out), nn.BatchNorm1D(d_out), nn.ReLU()]
            )
        layers.append(nn.Linear(embedding_dim[-1], 1))
        self.mlp = nn.Sequential(*layers)

        # Normalisation buffers
        self.register_buffer(
            "data_mean", paddle.to_tensor(data_mean, dtype="float32")
        )
        self.register_buffer(
            "data_std", paddle.to_tensor(data_std, dtype="float32")
        )

        # Loss
        if loss_type == "mse_loss":
            self._loss_fn = paddle.nn.functional.mse_loss
        elif loss_type == "l1_loss":
            self._loss_fn = paddle.nn.functional.l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        self.loss_type = loss_type

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    @property
    def loss_fn(self):
        return self._loss_fn

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool(self, x, mask):
        """Pool atom features into crystal-level features.

        Args:
            x: [B, N, D] per-atom features.
            mask: [B, N] boolean mask (True = real atom).

        Returns:
            pooled: [B, D]
        """
        if mask is not None:
            # Zero out padding positions before pooling
            x = x * mask.unsqueeze(-1).astype(x.dtype)

        if self._pooling == "max":
            if mask is not None:
                # Replace padded positions with -inf so they don't dominate max
                x = x + (~mask).unsqueeze(-1).astype(x.dtype) * (-1e9)
            return paddle.max(x, axis=1)
        else:  # mean
            if mask is not None:
                count = mask.astype(x.dtype).sum(axis=1, keepdim=True).clip(min=1)
                return x.sum(axis=1) / count
            return paddle.mean(x, axis=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(self, data):
        """Core forward pass.

        Expects ``data`` to be a dict with keys:
            - ``x``:         [B, N_max, atom_feat_dim] one-hot atom features
            - ``pos``:       [B, N_max, 3]             fractional coordinates
            - ``trans_vec``: [B, 3, 3]                 lattice vectors
            - ``mask``:      [B, N_max]                boolean (optional)
        """
        x = data["x"]
        pos = data["pos"]
        trans_vec = data["trans_vec"]
        mask = data.get("mask", None)

        # Embed atoms
        x = self.input_embeddings(x)  # [B, N, model_dim]

        # Compute pairwise distances with periodicity → Gaussian RBF
        min_dist = self.dist_computer(pos, trans_vec, mask)  # [B, N, N]
        dist_rbf = self.gaussian_rbf(min_dist)  # [B, N, N, n_gaussians]

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x, dist_rbf, mask)

        # Pre-pooling projection (BatchNorm1D expects [B*N, D])
        B, N, D = x.shape
        x = x.reshape([B * N, D])
        x = self.proj_before_pooling(x)
        x = x.reshape([B, N, -1])

        # Pool over atoms
        x = self._pool(x, mask)  # [B, D_pooled]

        # MLP head
        return self.mlp(x)  # [B, 1]

    def forward(self, data, return_loss=True, return_prediction=True):
        """PaddleMaterials-standard forward.

        Args:
            data: Dict with crystal data and (optionally) target labels.
            return_loss: Whether to compute and return the loss.
            return_prediction: Whether to return un-normalised predictions.

        Returns:
            Dict with ``loss_dict`` and ``pred_dict``.
        """
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        pred = self._forward(data)

        loss_dict = {}
        if return_loss:
            label = data[self.property_names]
            label = self.normalize(label)
            loss = self.loss_fn(input=pred, label=label)
            loss_dict["loss"] = loss

        prediction = {}
        if return_prediction:
            prediction[self.property_names] = self.unnormalize(pred)

        return {"loss_dict": loss_dict, "pred_dict": prediction}

    @paddle.no_grad()
    def predict(self, data):
        """Inference-only forward (no loss)."""
        pred = self._forward(data)
        return {self.property_names: self.unnormalize(pred)}
