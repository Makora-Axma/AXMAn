"""
Unit and integration tests for FPHA kernels (Task 5.1).
Tests correctness of Triton and vectorized fast kernels with static arity tiling.
"""

import math
import sys
from pathlib import Path

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

import pytest
import torch

from axman.core.blocks.fpha import FusedPortHypergraphAttention
from axman.core.kernels.triton_fpha import (
    HAS_TRITON,
    K_BLOCK_ARITY,
    fused_port_hypergraph_attention_fast,
)


def test_triton_module_import_and_constants():
    """Verify kernel module loads, flags are defined, and K_BLOCK matches specification."""
    assert K_BLOCK_ARITY == 16
    assert isinstance(HAS_TRITON, bool)


def test_fast_kernel_numerical_equivalence_with_fpha_reference():
    """
    Verify that fused_port_hypergraph_attention_fast produces identical results
    to FPHA's reference execution.
    """
    torch.manual_seed(42)
    num_nodes = 10
    num_edges = 4
    num_heads = 4
    head_dim = 16
    d_model = num_heads * head_dim
    scale = 1.0 / math.sqrt(head_dim)

    # Edge arities: e0 has 2 sources, e1 has 1 source, e2 has 4 sources, e3 has 3 sources
    arities = [2, 1, 4, 3]
    edge_offsets = torch.tensor([0, 2, 3, 7, 10], dtype=torch.long)
    total_sources = 10

    edge_targets = torch.tensor([0, 1, 0, 5], dtype=torch.long)
    edge_sources = torch.tensor([1, 2, 0, 3, 4, 1, 2, 6, 7, 8], dtype=torch.long)

    # Random Q (at edge targets), K, V
    q_targets = torch.randn(num_edges, num_heads, head_dim)
    k_premises = torch.randn(total_sources, num_heads, head_dim)
    v_premises = torch.randn(total_sources, num_heads, head_dim)

    # 1. Run fast kernel dispatcher
    node_msg_fast, edge_msg_fast = fused_port_hypergraph_attention_fast(
        q_targets=q_targets,
        k_premises=k_premises,
        v_premises=v_premises,
        edge_offsets=edge_offsets,
        edge_targets=edge_targets,
        num_nodes=num_nodes,
        scale=scale,
    )

    # 2. Reference computation matching FPHA logic
    edge_ids = torch.repeat_interleave(
        torch.arange(num_edges),
        torch.tensor(arities),
    )
    scores = (q_targets[edge_ids] * k_premises).sum(dim=-1) * scale

    max_scores = torch.full((num_edges, num_heads), -1e9)
    max_scores = max_scores.scatter_reduce(
        0, edge_ids.unsqueeze(-1).expand(-1, num_heads), scores, reduce="amax", include_self=True
    )
    exp_scores = torch.exp(scores - max_scores[edge_ids])
    sum_exp = torch.zeros(num_edges, num_heads).scatter_reduce(
        0, edge_ids.unsqueeze(-1).expand(-1, num_heads), exp_scores, reduce="sum", include_self=False
    )
    weights = exp_scores / (sum_exp[edge_ids] + 1e-9)
    weighted_v = weights.unsqueeze(-1) * v_premises

    edge_ref = torch.zeros(num_edges, num_heads, head_dim).scatter_reduce(
        0, edge_ids.view(-1, 1, 1).expand(-1, num_heads, head_dim), weighted_v, reduce="sum", include_self=False
    ).view(num_edges, d_model)

    node_ref = torch.zeros(num_nodes, d_model).scatter_reduce(
        0, edge_targets.unsqueeze(-1).expand(-1, d_model), edge_ref, reduce="sum", include_self=False
    )

    # Verify numerical match
    assert torch.allclose(edge_msg_fast, edge_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(node_msg_fast, node_ref, atol=1e-6, rtol=1e-5)


def test_fast_kernel_max_arity_boundary_k16():
    """Verify kernel operates properly with maximum allowed arity k = 16."""
    torch.manual_seed(123)
    num_nodes = 5
    num_edges = 2
    num_heads = 2
    head_dim = 8
    d_model = num_heads * head_dim

    # Hyperedge 0 has arity 16 (max tile boundary), Hyperedge 1 has arity 1
    edge_offsets = torch.tensor([0, 16, 17], dtype=torch.long)
    edge_targets = torch.tensor([2, 4], dtype=torch.long)
    total_sources = 17

    q_targets = torch.randn(num_edges, num_heads, head_dim)
    k_premises = torch.randn(total_sources, num_heads, head_dim)
    v_premises = torch.randn(total_sources, num_heads, head_dim)

    node_msg, edge_msg = fused_port_hypergraph_attention_fast(
        q_targets=q_targets,
        k_premises=k_premises,
        v_premises=v_premises,
        edge_offsets=edge_offsets,
        edge_targets=edge_targets,
        num_nodes=num_nodes,
    )

    assert node_msg.shape == (num_nodes, d_model)
    assert edge_msg.shape == (num_edges, d_model)
    assert not torch.isnan(node_msg).any()
    assert not torch.isnan(edge_msg).any()


def test_fast_kernel_empty_edges():
    """Verify edge cases with zero edges or empty hypergraphs."""
    num_nodes = 4
    num_heads = 2
    head_dim = 8
    d_model = num_heads * head_dim

    q_targets = torch.empty(0, num_heads, head_dim)
    k_premises = torch.empty(0, num_heads, head_dim)
    v_premises = torch.empty(0, num_heads, head_dim)
    edge_offsets = torch.tensor([0], dtype=torch.long)
    edge_targets = torch.empty(0, dtype=torch.long)

    node_msg, edge_msg = fused_port_hypergraph_attention_fast(
        q_targets=q_targets,
        k_premises=k_premises,
        v_premises=v_premises,
        edge_offsets=edge_offsets,
        edge_targets=edge_targets,
        num_nodes=num_nodes,
    )

    assert node_msg.shape == (num_nodes, d_model)
    assert edge_msg.shape == (0, d_model)
    assert torch.all(node_msg == 0.0)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_TRITON, reason="CUDA and Triton required")
def test_triton_cuda_kernel_execution():
    """
    Test native Triton kernel execution on GPU device when available.
    """
    device = torch.device("cuda:0")
    num_nodes = 8
    num_edges = 3
    num_heads = 4
    head_dim = 32

    edge_offsets = torch.tensor([0, 3, 7, 10], device=device, dtype=torch.long)
    edge_targets = torch.tensor([1, 3, 5], device=device, dtype=torch.long)
    total_sources = 10

    q_targets = torch.randn(num_edges, num_heads, head_dim, device=device)
    k_premises = torch.randn(total_sources, num_heads, head_dim, device=device)
    v_premises = torch.randn(total_sources, num_heads, head_dim, device=device)

    # Compute with prefer_triton=True
    node_msg_triton, edge_msg_triton = fused_port_hypergraph_attention_fast(
        q_targets=q_targets,
        k_premises=k_premises,
        v_premises=v_premises,
        edge_offsets=edge_offsets,
        edge_targets=edge_targets,
        num_nodes=num_nodes,
        prefer_triton=True,
    )

    # Compute with prefer_triton=False (fallback reference)
    node_msg_ref, edge_msg_ref = fused_port_hypergraph_attention_fast(
        q_targets=q_targets,
        k_premises=k_premises,
        v_premises=v_premises,
        edge_offsets=edge_offsets,
        edge_targets=edge_targets,
        num_nodes=num_nodes,
        prefer_triton=False,
    )

    assert torch.allclose(edge_msg_triton, edge_msg_ref, atol=1e-4, rtol=1e-3)
    assert torch.allclose(node_msg_triton, node_msg_ref, atol=1e-4, rtol=1e-3)
