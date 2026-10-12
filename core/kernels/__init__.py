"""
AXMAN Kernels Package.
Provides custom hardware-accelerated kernels for AXMANet architectures:
- Triton GPU Kernels (FPHA with static arity tile K_BLOCK=16)
- High-performance vectorized reference fallbacks
"""

from axman.core.kernels.triton_fpha import (
    HAS_TRITON,
    K_BLOCK_ARITY,
    fused_port_hypergraph_attention_fast,
    triton_fpha_edge_attention,
)

__all__ = [
    "HAS_TRITON",
    "K_BLOCK_ARITY",
    "fused_port_hypergraph_attention_fast",
    "triton_fpha_edge_attention",
]
