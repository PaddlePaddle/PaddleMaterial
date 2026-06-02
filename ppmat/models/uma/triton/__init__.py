from __future__ import annotations

"""Paddle-compatible fallback wrappers for UMA fast GPU path."""

from .node_to_edge_wigner_permute import (
    NodeToEdgeWignerPermuteFunction as UMASFastGPUNodeToEdgeWignerPermute,
)
from .permute_wigner_inv_edge_to_node import (
    PermuteWignerInvEdgeToNodeFunction as UMASFastGPUPermuteWignerInvEdgeToNode,
)

__all__ = [
    "UMASFastGPUNodeToEdgeWignerPermute",
    "UMASFastGPUPermuteWignerInvEdgeToNode",
]
