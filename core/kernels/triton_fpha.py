"""
Triton GPU Kernels for AXMANet (FPHA - Fused Port-Hypergraph Flash Attention).
Part of AXMAN Phase 5.1 Implementation.

This module provides high-performance custom kernels with static arity tiling (K_BLOCK = 16)
for hyperedges in Compressed Sparse Row (CSR) format:
- Fused dot-product attention calculation across ports of hyperedges.
- Segmented stabilized softmax per hyperedge tile.
- Attention-weighted value reduction into hyperedge representations.
- Scatter reduction of hyperedge messages into target nodes.

Includes a fully-functional PyTorch reference fallback for environments without
native Triton/CUDA or when running on CPU/unsupported architectures.
"""

import math
from typing import Optional, Tuple

import torch

# Check Triton availability safely
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

# Maximum hyperedge arity according to AXMAN ABI & Currying Specification
K_BLOCK_ARITY = 16


if HAS_TRITON:
    @triton.jit
    def _triton_fpha_hyperedge_kernel(
        # Pointers to Tensors
        q_target_ptr,         # [num_edges, num_heads, head_dim]
        k_premise_ptr,        # [total_sources, num_heads, head_dim]
        v_premise_ptr,        # [total_sources, num_heads, head_dim]
        edge_offsets_ptr,     # [num_edges + 1]
        out_edge_msg_ptr,     # [num_edges, num_heads, head_dim]
        # Strides for Q
        stride_q_e, stride_q_h, stride_q_d,
        # Strides for K
        stride_k_s, stride_k_h, stride_k_d,
        # Strides for V
        stride_v_s, stride_v_h, stride_v_d,
        # Strides for Out
        stride_out_e, stride_out_h, stride_out_d,
        # Dimensions & Constants
        scale,
        num_edges,
        HEAD_DIM: tl.constexpr,
        K_BLOCK: tl.constexpr,
    ):
        """
        Triton kernel computing attention over sources of a hyperedge with static arity tiling (K_BLOCK=16).
        Grid: (num_edges, num_heads)
        """
        edge_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        if edge_idx >= num_edges:
            return

        # Read CSR slice for this hyperedge
        start_idx = tl.load(edge_offsets_ptr + edge_idx)
        end_idx = tl.load(edge_offsets_ptr + edge_idx + 1)
        arity = end_idx - start_idx

        # Static range for premises [0 .. K_BLOCK - 1]
        port_offsets = tl.arange(0, K_BLOCK)
        premise_mask = port_offsets < arity
        d_offsets = tl.arange(0, HEAD_DIM)

        # Pointer to Q for this edge and head: [1, HEAD_DIM]
        q_ptr = q_target_ptr + edge_idx * stride_q_e + head_idx * stride_q_h + d_offsets * stride_q_d
        q_vec = tl.load(q_ptr)  # [HEAD_DIM]

        # Pointers to K and V for the premises of this edge: [K_BLOCK, HEAD_DIM]
        src_indices = start_idx + port_offsets
        # 2D offsets
        k_ptrs = (
            k_premise_ptr
            + src_indices[:, None] * stride_k_s
            + head_idx * stride_k_h
            + d_offsets[None, :] * stride_k_d
        )
        v_ptrs = (
            v_premise_ptr
            + src_indices[:, None] * stride_v_s
            + head_idx * stride_v_h
            + d_offsets[None, :] * stride_v_d
        )

        # Load K with masking for arity < K_BLOCK
        k_tile = tl.load(k_ptrs, mask=premise_mask[:, None], other=0.0)  # [K_BLOCK, HEAD_DIM]

        # Scaled dot-product: Q . K^T
        # q_vec: [HEAD_DIM], k_tile: [K_BLOCK, HEAD_DIM]
        scores = tl.sum(k_tile * q_vec[None, :], axis=1) * scale  # [K_BLOCK]
        scores = tl.where(premise_mask, scores, -1e9)

        # Numerically stable Softmax
        max_score = tl.max(scores, axis=0)
        exp_scores = tl.where(premise_mask, tl.exp(scores - max_score), 0.0)
        sum_exp = tl.sum(exp_scores, axis=0) + 1e-9
        weights = exp_scores / sum_exp  # [K_BLOCK]

        # Weighted sum of values: weights . V
        v_tile = tl.load(v_ptrs, mask=premise_mask[:, None], other=0.0)  # [K_BLOCK, HEAD_DIM]
        out_msg = tl.sum(v_tile * weights[:, None], axis=0)              # [HEAD_DIM]

        # Store output message for this hyperedge and head
        out_ptr = out_edge_msg_ptr + edge_idx * stride_out_e + head_idx * stride_out_h + d_offsets * stride_out_d
        tl.store(out_ptr, out_msg)


def triton_fpha_edge_attention(
    q_targets: torch.Tensor,
    k_premises: torch.Tensor,
    v_premises: torch.Tensor,
    edge_offsets: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Executes Triton FPHA kernel for hyperedge premise aggregation.
    Expects CUDA tensors:
      q_targets: [num_edges, num_heads, head_dim]
      k_premises: [total_sources, num_heads, head_dim]
      v_premises: [total_sources, num_heads, head_dim]
      edge_offsets: [num_edges + 1] (int64 or int32)
    Returns:
      edge_messages: [num_edges, num_heads, head_dim]
    """
    if not HAS_TRITON or not q_targets.is_cuda:
        raise RuntimeError("Triton kernel requires triton package and CUDA tensors.")

    num_edges, num_heads, head_dim = q_targets.shape
    out_edge_msg = torch.empty_like(q_targets)

    grid = (num_edges, num_heads)
    _triton_fpha_hyperedge_kernel[grid](
        q_targets,
        k_premises,
        v_premises,
        edge_offsets.to(torch.int64),
        out_edge_msg,
        q_targets.stride(0), q_targets.stride(1), q_targets.stride(2),
        k_premises.stride(0), k_premises.stride(1), k_premises.stride(2),
        v_premises.stride(0), v_premises.stride(1), v_premises.stride(2),
        out_edge_msg.stride(0), out_edge_msg.stride(1), out_edge_msg.stride(2),
        scale,
        num_edges,
        HEAD_DIM=head_dim,
        K_BLOCK=K_BLOCK_ARITY,
    )
    return out_edge_msg


def fused_port_hypergraph_attention_fast(
    q_targets: torch.Tensor,
    k_premises: torch.Tensor,
    v_premises: torch.Tensor,
    edge_offsets: torch.Tensor,
    edge_targets: torch.Tensor,
    num_nodes: int,
    scale: Optional[float] = None,
    prefer_triton: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified high-performance FPHA attention pass.
    Dispatches to Triton GPU kernel when running on CUDA and Triton is available;
    otherwise falls back to a vectorized static-tiled or scatter-reduced implementation.

    Args:
        q_targets: [num_edges, num_heads, head_dim] Queries at hyperedge targets.
        k_premises: [total_sources, num_heads, head_dim] Keys of port-enriched sources.
        v_premises: [total_sources, num_heads, head_dim] Values of port-enriched sources.
        edge_offsets: [num_edges + 1] CSR offsets into edge sources.
        edge_targets: [num_edges] Target node indices.
        num_nodes: Total count of nodes in the graph.
        scale: Optional scaling factor (defaults to 1.0 / sqrt(head_dim)).
        prefer_triton: If True, uses Triton kernel on CUDA when available.

    Returns:
        Tuple of:
          - node_aggregated: [num_nodes, num_heads * head_dim] messages per target node.
          - edge_messages: [num_edges, num_heads * head_dim] aggregated representation per hyperedge.
    """
    num_edges, num_heads, head_dim = q_targets.shape
    d_model = num_heads * head_dim
    device = q_targets.device
    dtype = q_targets.dtype

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if num_edges == 0:
        return (
            torch.zeros(num_nodes, d_model, device=device, dtype=dtype),
            torch.zeros(0, d_model, device=device, dtype=dtype),
        )

    # 1. Compute Hyperedge Messages (Triton or Vectorized PyTorch)
    use_triton = prefer_triton and HAS_TRITON and q_targets.is_cuda
    if use_triton:
        edge_msg_3d = triton_fpha_edge_attention(
            q_targets=q_targets,
            k_premises=k_premises,
            v_premises=v_premises,
            edge_offsets=edge_offsets,
            scale=scale,
        )
    else:
        # Optimized Reference Implementation with static arity handling
        edge_counts = edge_offsets[1:] - edge_offsets[:-1]
        edge_ids = torch.repeat_interleave(
            torch.arange(num_edges, device=device, dtype=torch.long),
            edge_counts.long(),
        )
        total_sources = k_premises.shape[0]

        if total_sources == 0:
            return (
                torch.zeros(num_nodes, d_model, device=device, dtype=dtype),
                torch.zeros(num_edges, d_model, device=device, dtype=dtype),
            )

        # q for each premise from its hyperedge target
        q_expanded = q_targets[edge_ids]  # [total_sources, num_heads, head_dim]
        scores = (q_expanded * k_premises).sum(dim=-1) * scale  # [total_sources, num_heads]

        max_scores = torch.full((num_edges, num_heads), -1e9, device=device, dtype=dtype)
        max_scores = max_scores.scatter_reduce(
            0,
            edge_ids.unsqueeze(-1).expand(-1, num_heads),
            scores,
            reduce="amax",
            include_self=True,
        )

        scores_centered = scores - max_scores[edge_ids]
        exp_scores = torch.exp(scores_centered)

        sum_exp = torch.zeros(num_edges, num_heads, device=device, dtype=dtype)
        sum_exp = sum_exp.scatter_reduce(
            0,
            edge_ids.unsqueeze(-1).expand(-1, num_heads),
            exp_scores,
            reduce="sum",
            include_self=False,
        )

        weights = exp_scores / (sum_exp[edge_ids] + 1e-9)
        weighted_v = weights.unsqueeze(-1) * v_premises  # [total_sources, num_heads, head_dim]

        edge_msg_3d = torch.zeros(num_edges, num_heads, head_dim, device=device, dtype=dtype)
        edge_msg_3d = edge_msg_3d.scatter_reduce(
            0,
            edge_ids.view(-1, 1, 1).expand(-1, num_heads, head_dim),
            weighted_v,
            reduce="sum",
            include_self=False,
        )

    edge_messages = edge_msg_3d.view(num_edges, d_model)

    # 2. Scatter reduction into target nodes
    node_aggregated = torch.zeros(num_nodes, d_model, device=device, dtype=dtype)
    node_aggregated = node_aggregated.scatter_reduce(
        0,
        edge_targets.long().unsqueeze(-1).expand(-1, d_model),
        edge_messages,
        reduce="sum",
        include_self=False,
    )

    return node_aggregated, edge_messages
