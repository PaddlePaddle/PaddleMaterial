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

"""MOFDiff: Coarse-Grained Diffusion for Metal-Organic Framework Generation.

This module implements a simplified PaddlePaddle port of MOFDiff
(https://github.com/microsoft/MOFDiff). The original relies on
torch_geometric, torch_scatter, GemNetOC, and hydra; this version
substitutes those with simple MLPs and the ppmat scatter utilities so
that it is fully self-contained and CPU-testable.

The three-stage pipeline:
    1. **Encoder** — encodes per-node features into a graph-level latent
       via a VAE bottleneck (fc_mu / fc_var → reparameterize).
    2. **CG Diffusion** — VP (Variance Preserving) noise for both
       building-block type embeddings and fractional coordinates, with a
       denoiser that predicts the noise components.
    3. **Lattice predictor** — an MLP that maps the latent to 6 lattice
       parameters (3 lengths + 3 angles).

Reference: Yao *et al.*, "Coarse-Grained Diffusion for Metal-Organic
Framework Generation", *ICLR 2024*.
"""

import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.utils.scatter import scatter_mean

EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_mlp(in_dim, hidden_dim, num_layers, out_dim):
    """Build a simple feed-forward MLP with ReLU activations."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class GaussianFourierProjection(nn.Layer):
    """Gaussian Fourier embeddings for noise levels.

    Maps a scalar timestep *t* to a ``2 * embedding_size``-dimensional
    vector via random Fourier features:

        out = [sin(2π · t · W), cos(2π · t · W)]

    where *W* is drawn once from N(0, scale²) and kept frozen.
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        W = paddle.randn([embedding_size]) * scale
        self.register_buffer("W", W)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape ``[B]`` or ``[B, 1]``.

        Returns:
            Tensor of shape ``[B, 2 * embedding_size]``.
        """
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        x_proj = x * self.W.unsqueeze(0) * 2 * math.pi
        return paddle.concat([paddle.sin(x_proj), paddle.cos(x_proj)], axis=-1)


# ---------------------------------------------------------------------------
# Diffusion schedules
# ---------------------------------------------------------------------------

class VP(nn.Layer):
    """Variance Preserving diffusion with a cosine schedule.

    Forward process::

        h_t = sqrt(ᾱ_t) · h_0  +  sqrt(1 − ᾱ_t) · ε

    Reverse (DDPM) step::

        h_{t-1} = (1/√α_t)(h_t − (β_t / √(1 − ᾱ_t)) · ε̂)  +  σ_t · z
    """

    def __init__(self, num_steps=1000, s=0.0001, power=2, clipmax=0.999):
        super().__init__()
        self.num_steps = num_steps

        t = np.arange(0, num_steps + 1, dtype=np.float64)
        f_t = np.cos((np.pi / 2) * ((t / num_steps) + s) / (1 + s)) ** power
        alpha_bars = f_t / f_t[0]

        betas = np.concatenate([[0.0], 1 - (alpha_bars[1:] / alpha_bars[:-1])])
        betas = np.clip(betas, 0, clipmax)

        # Posterior variance  σ²_t = β_t · (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
        sigmas_sq = betas[1:] * ((1 - alpha_bars[:-1]) / (1 - alpha_bars[1:] + EPSILON))
        sigmas = np.sqrt(np.concatenate([[0.0], sigmas_sq]))

        self.register_buffer(
            "alpha_bars", paddle.to_tensor(alpha_bars, dtype="float32")
        )
        self.register_buffer(
            "betas", paddle.to_tensor(betas, dtype="float32")
        )
        self.register_buffer(
            "sigmas", paddle.to_tensor(sigmas, dtype="float32")
        )

    def forward(self, h0, t):
        """Forward diffusion: add noise at timestep *t*.

        Args:
            h0: Clean signal ``[N, D]``.
            t:  Integer timesteps ``[N]``.

        Returns:
            (h_t, eps): noised signal and the noise that was added.
        """
        alpha_bar = paddle.gather(self.alpha_bars, t)  # [N]
        eps = paddle.randn(h0.shape)
        sqrt_ab = paddle.sqrt(alpha_bar).unsqueeze(-1)
        sqrt_1_ab = paddle.sqrt(1.0 - alpha_bar).unsqueeze(-1)
        ht = sqrt_ab * h0 + sqrt_1_ab * eps
        return ht, eps

    def reverse(self, ht, eps_h, t):
        """Single DDPM reverse step.

        Args:
            ht:    Noised signal ``[N, D]``.
            eps_h: Predicted noise ``[N, D]``.
            t:     Integer timesteps ``[N]``.

        Returns:
            h_{t-1}: denoised one step.
        """
        alpha = (1 - paddle.gather(self.betas, t)).unsqueeze(-1)
        alpha_bar = paddle.gather(self.alpha_bars, t).unsqueeze(-1)
        sigma = paddle.gather(self.sigmas, t).unsqueeze(-1)

        z = paddle.where(
            (t > 1).unsqueeze(-1).expand_as(ht),
            paddle.randn(ht.shape),
            paddle.zeros(ht.shape),
        )
        coef = (1.0 - alpha) / paddle.sqrt(1.0 - alpha_bar + EPSILON)
        return (1.0 / paddle.sqrt(alpha + EPSILON)) * (ht - coef * eps_h) + sigma * z


# ---------------------------------------------------------------------------
# Encoder / Decoder (MLP stand-ins for GemNetOC)
# ---------------------------------------------------------------------------

class SimpleGNNEncoder(nn.Layer):
    """MLP encoder that replaces GemNetOC for CPU-testable builds.

    Per-node features are projected through an MLP, then mean-pooled
    per graph to produce a graph-level representation.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.mlp = build_mlp(input_dim, hidden_dim, num_layers, output_dim)

    def forward(self, node_features, batch_indices, num_graphs):
        """
        Args:
            node_features: ``[total_nodes, input_dim]``
            batch_indices: ``[total_nodes]`` — graph id for each node.
            num_graphs:    int — number of graphs in the batch.

        Returns:
            ``[num_graphs, output_dim]``
        """
        h = self.mlp(node_features)
        out = scatter_mean(h, batch_indices, dim=0, dim_size=num_graphs)
        return out


class SimpleGNNDecoder(nn.Layer):
    """MLP decoder predicting coordinate noise and type noise."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_coord_dim=3,
        output_type_dim=100,
        num_layers=3,
    ):
        super().__init__()
        self.coord_mlp = build_mlp(input_dim, hidden_dim, num_layers, output_coord_dim)
        self.type_mlp = build_mlp(input_dim, hidden_dim, num_layers, output_type_dim)

    def forward(self, node_features):
        """
        Args:
            node_features: ``[total_nodes, input_dim]``

        Returns:
            (eps_x, eps_h): predicted noise for coords ``[N,3]``
            and types ``[N, num_bb_types]``.
        """
        eps_x = self.coord_mlp(node_features)
        eps_h = self.type_mlp(node_features)
        return eps_x, eps_h


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MOFDiff(nn.Layer):
    """MOFDiff: Coarse-Grained Diffusion for MOF generation.

    A simplified PaddleMaterials implementation that keeps the same
    three-stage architecture as the reference while replacing the heavy
    GNN backbone (GemNetOC) with lightweight MLPs.

    **Training forward** returns ``{"loss_dict": {...}}`` following the
    PaddleMaterials convention (see ``DiffCSP``).

    Args:
        node_feat_dim:  Dimension of input per-node features.
        hidden_dim:     Width of all hidden MLPs.
        latent_dim:     Dimension of the VAE latent space.
        num_bb_types:   Number of building-block type classes.
        max_num_bbs:    Maximum number of building blocks in a MOF.
        num_diffusion_steps:  Number of VP diffusion steps.
        fc_num_layers:  Depth of each MLP sub-network.
        kl_weight:      Weight for the KL divergence loss term.
    """

    def __init__(
        self,
        node_feat_dim=64,
        hidden_dim=128,
        latent_dim=64,
        num_bb_types=50,
        max_num_bbs=20,
        num_diffusion_steps=100,
        fc_num_layers=3,
        kl_weight=0.1,
    ):
        super().__init__()

        # -- timestep embedding ------------------------------------------------
        self.time_embedding = GaussianFourierProjection(embedding_size=128)
        time_emb_dim = 256  # 128 × 2 (sin + cos)

        # -- encoder -----------------------------------------------------------
        self.encoder = SimpleGNNEncoder(
            node_feat_dim, hidden_dim, latent_dim, fc_num_layers
        )

        # -- VAE bottleneck ----------------------------------------------------
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

        # -- lattice predictor (6 = 3 lengths + 3 angles) ---------------------
        self.fc_lattice = build_mlp(latent_dim, hidden_dim, fc_num_layers, 6)

        # -- number-of-BBs classifier -----------------------------------------
        self.fc_num_bbs = build_mlp(
            latent_dim, hidden_dim, fc_num_layers, max_num_bbs + 1
        )

        # -- denoiser ----------------------------------------------------------
        # Input: noisy coords (3) + one-hot bb type + time emb + latent
        denoiser_input = 3 + num_bb_types + time_emb_dim + latent_dim
        self.denoiser = SimpleGNNDecoder(
            denoiser_input, hidden_dim, 3, num_bb_types, fc_num_layers
        )

        # -- diffusion process -------------------------------------------------
        self.vp_diffusion = VP(num_diffusion_steps)
        self.num_diffusion_steps = num_diffusion_steps
        self.num_bb_types = num_bb_types
        self.latent_dim = latent_dim
        self.max_num_bbs = max_num_bbs
        self.kl_weight = kl_weight

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    def reparameterize(self, mu, log_var):
        """Sample *z* ~ N(mu, σ²) via the reparameterization trick."""
        std = paddle.exp(0.5 * log_var)
        eps = paddle.randn(std.shape)
        return mu + eps * std

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, node_features, batch_indices, num_graphs):
        """Encode a batch of graphs into latent (mu, log_var, z).

        Returns:
            (mu, log_var, z) — each ``[num_graphs, latent_dim]``.
        """
        h = self.encoder(node_features, batch_indices, num_graphs)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z

    # ------------------------------------------------------------------
    # Core forward (training)
    # ------------------------------------------------------------------

    def _forward(self, batch):
        """Core training forward: encode → diffuse → denoise → losses.

        Expected keys in *batch*:

        * ``node_features`` — ``[total_nodes, node_feat_dim]``
        * ``frac_coords``   — ``[total_nodes, 3]``
        * ``bb_types``      — ``[total_nodes]``  (int, 0-based class ids)
        * ``batch``         — ``[total_nodes]``  (graph index per node)
        * ``num_atoms``     — ``[B]``
        * ``lattice_params``— ``[B, 6]``
        """
        node_features = batch["node_features"]
        frac_coords = batch["frac_coords"]
        bb_types = batch["bb_types"]
        batch_idx = batch["batch"]
        num_atoms = batch["num_atoms"]
        lattice_params = batch["lattice_params"]

        batch_size = num_atoms.shape[0]
        total_nodes = node_features.shape[0]

        # --- 1. Encode -------------------------------------------------------
        mu, log_var, z = self.encode(node_features, batch_idx, batch_size)

        # --- 2. Lattice prediction loss ---------------------------------------
        pred_lattice = self.fc_lattice(z)
        loss_lattice = F.mse_loss(pred_lattice, lattice_params)

        # --- 3. Num-BBs classification loss -----------------------------------
        pred_num_bbs = self.fc_num_bbs(z)
        loss_num_bbs = F.cross_entropy(pred_num_bbs, num_atoms)

        # --- 4. Sample diffusion timestep per node ----------------------------
        t = paddle.randint(1, self.num_diffusion_steps + 1, [total_nodes])

        # --- 5. Noise the BB-type one-hot via VP diffusion --------------------
        bb_onehot = F.one_hot(
            bb_types.astype("int64"), num_classes=self.num_bb_types
        ).astype("float32")
        noisy_h, eps_h = self.vp_diffusion.forward(bb_onehot, t)

        # --- 6. Noise the fractional coordinates via VP diffusion --------------
        noisy_x, eps_x = self.vp_diffusion.forward(frac_coords, t)

        # --- 7. Build time embedding per node ---------------------------------
        t_normalised = t.astype("float32") / self.num_diffusion_steps
        time_emb = self.time_embedding(t_normalised)  # [N, 256]

        # --- 8. Expand latent to per-node -------------------------------------
        z_per_node = z[batch_idx]  # [N, latent_dim]

        # --- 9. Denoiser input ------------------------------------------------
        denoiser_in = paddle.concat(
            [noisy_x, noisy_h, time_emb, z_per_node], axis=-1
        )
        pred_eps_x, pred_eps_h = self.denoiser(denoiser_in)

        # --- 10. Reconstruction losses ----------------------------------------
        loss_coord = F.mse_loss(pred_eps_x, eps_x)
        loss_type = F.mse_loss(pred_eps_h, eps_h)

        # --- 11. KL divergence ------------------------------------------------
        loss_kl = -0.5 * paddle.mean(
            1.0 + log_var - mu.pow(2) - log_var.exp()
        )

        # --- 12. Total loss ---------------------------------------------------
        loss = loss_coord + loss_type + self.kl_weight * loss_kl + loss_lattice + loss_num_bbs

        return {
            "loss_dict": {
                "loss": loss,
                "loss_coord": loss_coord,
                "loss_type": loss_type,
                "loss_kl": loss_kl,
                "loss_lattice": loss_lattice,
                "loss_num_bbs": loss_num_bbs,
            }
        }

    # ------------------------------------------------------------------
    # Public forward (PM convention)
    # ------------------------------------------------------------------

    def forward(self, batch, **kwargs):
        """PaddleMaterials-compatible forward.

        Returns:
            dict with ``"loss_dict"`` containing all loss terms.
        """
        return self._forward(batch)

    # ------------------------------------------------------------------
    # Inference / sampling
    # ------------------------------------------------------------------

    @paddle.no_grad()
    def sample(self, z, num_atoms_per_graph, num_steps=None):
        """Generate MOF structures from a latent vector *z*.

        This is a simplified DDPM reverse loop that iteratively denoises
        building-block types and fractional coordinates.

        Args:
            z:                   ``[B, latent_dim]`` — sampled latent.
            num_atoms_per_graph: ``[B]`` — how many BBs per structure.
            num_steps:           override for diffusion steps.

        Returns:
            dict with ``pred_coords``, ``pred_types``, ``pred_lattice``.
        """
        if num_steps is None:
            num_steps = self.num_diffusion_steps

        batch_size = z.shape[0]
        total_nodes = int(num_atoms_per_graph.sum().item())

        # Build batch index
        batch_idx = paddle.repeat_interleave(
            paddle.arange(batch_size), num_atoms_per_graph
        )

        # Start from pure noise
        ht = paddle.randn([total_nodes, self.num_bb_types])
        xt = paddle.randn([total_nodes, 3])

        z_per_node = z[batch_idx]

        for step in range(num_steps, 0, -1):
            t = paddle.full([total_nodes], step, dtype="int64")
            t_norm = t.astype("float32") / self.num_diffusion_steps
            time_emb = self.time_embedding(t_norm)

            denoiser_in = paddle.concat([xt, ht, time_emb, z_per_node], axis=-1)
            pred_eps_x, pred_eps_h = self.denoiser(denoiser_in)

            # VP reverse for types
            ht = self.vp_diffusion.reverse(ht, pred_eps_h, t)

            # VP reverse for coordinates
            xt = self.vp_diffusion.reverse(xt, pred_eps_x, t)

        pred_lattice = self.fc_lattice(z)

        return {
            "pred_coords": xt,
            "pred_types": ht.argmax(axis=-1),
            "pred_lattice": pred_lattice,
        }
