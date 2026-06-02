from __future__ import annotations

import paddle

from ..paddle_utils import *

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Unified Radial MLP: Computes all layers' radial functions in a single
batched operation.

Instead of running N separate RadialMLP forward passes:
    for layer in layers:
        radial_out = layer.so2_conv_1.rad_func(x_edge)  # Sequential

We run one batched first layer, then each tail:
    all_radial_outs = unified_radial_mlp(x_edge)  # list of [E, out]
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .radial import RadialMLP
__all__ = ["UnifiedRadialMLP", "create_unified_radial_mlp"]
_EXPECTED_NET_STRUCTURE = (
    paddle.nn.Linear,
    paddle.nn.LayerNorm,
    paddle.nn.SiLU,
    paddle.nn.Linear,
    paddle.nn.LayerNorm,
    paddle.nn.SiLU,
    paddle.nn.Linear,
)


def _validate_radial_mlp(
    mlp: RadialMLP, idx: int, reference: (RadialMLP | None)
) -> None:
    """
    Validate a single RadialMLP has expected structure and matches reference.

    Args:
        mlp: The RadialMLP to validate.
        idx: Index in the list (for error messages).
        reference: First RadialMLP to compare dimensions against (None for first).
    """
    if len(mlp.net) != 7:
        raise ValueError(f"RadialMLP[{idx}]: expected 7 layers, got {len(mlp.net)}")
    for j, expected_type in enumerate(_EXPECTED_NET_STRUCTURE):
        if not isinstance(mlp.net[j], expected_type):
            raise TypeError(
                f"RadialMLP[{idx}].net[{j}]: expected {expected_type.__name__}, got {type(mlp.net[j]).__name__}"
            )
    if reference is not None:
        for j in (0, 3, 6):
            if mlp.net[j].in_features != reference.net[j].in_features:
                raise ValueError(
                    f"RadialMLP[{idx}].net[{j}]: in_features mismatch ({mlp.net[j].in_features} vs {reference.net[j].in_features})"
                )
            if mlp.net[j].out_features != reference.net[j].out_features:
                raise ValueError(
                    f"RadialMLP[{idx}].net[{j}]: out_features mismatch ({mlp.net[j].out_features} vs {reference.net[j].out_features})"
                )


class UnifiedRadialMLP(paddle.nn.Module):
    """
    Unified radial MLP that batches the first linear layer across N RadialMLPs.

    The first layer uses concatenated weights for a single GEMM (all N layers
    share the same input). Layers 2+ use stacked weight buffers for fast
    indexed functional calls.
    """

    def __init__(self, radial_mlps: list[RadialMLP]) -> None:
        """
        Initialize from a list of RadialMLP modules.

        Args:
            radial_mlps: List of RadialMLP modules with identical architecture.
        """
        super().__init__()
        assert len(radial_mlps) > 0, "Need at least one RadialMLP"
        for i, mlp in enumerate(radial_mlps):
            _validate_radial_mlp(mlp, i, radial_mlps[0] if i > 0 else None)
        self.num_layers = len(radial_mlps)
        self.hidden_features = radial_mlps[0].net[0].out_features
        self.ln_eps = radial_mlps[0].net[1].eps
        self.register_buffer(
            "W1_cat", paddle.cat([mlp.net[0].weight.data for mlp in radial_mlps], dim=0)
        )
        self.register_buffer(
            "b1_cat", paddle.cat([mlp.net[0].bias.data for mlp in radial_mlps], dim=0)
        )
        self.register_buffer(
            "ln1_weight",
            paddle.stack([mlp.net[1].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln1_bias",
            paddle.stack([mlp.net[1].bias.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc2_weight",
            paddle.stack([mlp.net[3].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc2_bias",
            paddle.stack([mlp.net[3].bias.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln2_weight",
            paddle.stack([mlp.net[4].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln2_bias",
            paddle.stack([mlp.net[4].bias.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc3_weight",
            paddle.stack([mlp.net[6].weight.data for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc3_bias",
            paddle.stack([mlp.net[6].bias.data for mlp in radial_mlps], dim=0),
        )

    def umas_radial_mlp(self, h: paddle.Tensor, i: int) -> paddle.Tensor:
        """Apply layers 2+ (LN -> SiLU -> Linear -> LN -> SiLU -> Linear)."""
        H = self.hidden_features
        h = paddle.nn.functional.layer_norm(
            h, (H,), self.ln1_weight[i], self.ln1_bias[i], self.ln_eps
        )
        h = paddle.nn.functional.silu(h)
        h = paddle.compat.nn.functional.linear(h, self.fc2_weight[i], self.fc2_bias[i])
        h = paddle.nn.functional.layer_norm(
            h, (H,), self.ln2_weight[i], self.ln2_bias[i], self.ln_eps
        )
        h = paddle.nn.functional.silu(h)
        return paddle.compat.nn.functional.linear(
            h, self.fc3_weight[i], self.fc3_bias[i]
        )

    def forward(self, x: paddle.Tensor) -> list[paddle.Tensor]:
        """
        Compute all N radial outputs.

        Args:
            x: Input tensor of shape [E, in_features]

        Returns:
            List of N tensors, each of shape [E, out_features]
        """
        h_all = paddle.compat.nn.functional.linear(x, self.W1_cat, self.b1_cat)
        h_per_layer = h_all.split(self.hidden_features, dim=1)
        return [self.umas_radial_mlp(h_per_layer[i], i) for i in range(self.num_layers)]


def create_unified_radial_mlp(radial_mlps: list) -> UnifiedRadialMLP:
    """
    Factory function to create a UnifiedRadialMLP from a list of RadialMLPs.

    Args:
        radial_mlps: List of RadialMLP modules

    Returns:
        UnifiedRadialMLP instance with shared first layer weights
    """
    return UnifiedRadialMLP(radial_mlps)
