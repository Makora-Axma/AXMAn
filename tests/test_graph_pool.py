"""
Unit tests for Task 5.2: Pinned Graph Memory Pools and Continuous Batching.
Tests axman.core.memory.graph_pool module.
"""

import sys
from pathlib import Path

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

import pytest
import numpy as np
import torch

from axman.core.memory import (
    BatchedHypergraph,
    ContinuousGraphBatcher,
    PinnedHostMemoryPool,
    DeviceRingBuffer,
    GraphMemoryPool,
)
from axman.core.blocks.amb_backbone import AXMANetBackbone
from axman.core.config import AxmanConfig


def _create_sample_graph(num_nodes: int, feat_dim: int, edges: list):
    """
    Helper creating a test graph dict:
    edges is a list of tuples: (sources_list, ports_list, target_node)
    """
    node_features = torch.randn(num_nodes, feat_dim)
    edge_sources = []
    edge_ports = []
    edge_targets = []
    edge_offsets = [0]

    for srcs, prts, tgt in edges:
        edge_sources.extend(srcs)
        edge_ports.extend(prts)
        edge_targets.append(tgt)
        edge_offsets.append(len(edge_sources))

    return {
        "node_features": node_features,
        "edge_sources": torch.tensor(edge_sources, dtype=torch.long),
        "edge_ports": torch.tensor(edge_ports, dtype=torch.int32),
        "edge_targets": torch.tensor(edge_targets, dtype=torch.long),
        "edge_offsets": torch.tensor(edge_offsets, dtype=torch.long),
    }


def test_continuous_graph_batcher_topological_invariants():
    """
    Verify continuous batching correctly shifts node IDs and CSR offsets across
    independent graphs of variable topology without corrupting indices.
    """
    feat_dim = 16

    # Graph 0: 3 nodes (0, 1, 2), 2 hyperedges
    # e0: sources [0, 1] -> target 2
    # e1: sources [1] -> target 0
    g0 = _create_sample_graph(
        num_nodes=3,
        feat_dim=feat_dim,
        edges=[
            ([0, 1], [0, 1], 2),
            ([1], [0], 0),
        ],
    )

    # Graph 1: 4 nodes (0, 1, 2, 3), 1 hyperedge
    # e0: sources [2, 3] -> target 1
    g1 = _create_sample_graph(
        num_nodes=4,
        feat_dim=feat_dim,
        edges=[
            ([2, 3], [0, 1], 1),
        ],
    )

    # Batch graphs
    batched = ContinuousGraphBatcher.batch([g0, g1], as_torch=True)

    assert batched.num_graphs == 2
    assert batched.total_nodes == 3 + 4  # 7 nodes
    assert batched.total_edges == 2 + 1  # 3 edges
    assert batched.total_sources == 3 + 2  # 5 sources

    assert batched.node_features.shape == (7, feat_dim)
    assert batched.edge_targets.shape == (3,)
    assert batched.edge_offsets.shape == (4,)

    # Verify G1 targets and sources got offset by 3 (num_nodes of G0)
    # G0: target 2, target 0
    # G1: target 1 + 3 = 4
    assert batched.edge_targets.tolist() == [2, 0, 4]

    # Sources:
    # G0: [0, 1], [1]
    # G1: [2 + 3, 3 + 3] = [5, 6]
    assert batched.edge_sources.tolist() == [0, 1, 1, 5, 6]

    # CSR offsets:
    # G0: [0, 2, 3]
    # G1: [0 + 3, 2 + 3] = [3, 5]
    # Unified CSR offsets: [0, 2, 3, 5]
    assert batched.edge_offsets.tolist() == [0, 2, 3, 5]

    # Verify sub-graph boundary pointers
    assert batched.graph_node_ptr.tolist() == [0, 3, 7]
    assert batched.graph_edge_ptr.tolist() == [0, 2, 3]
    assert batched.graph_source_ptr.tolist() == [0, 3, 5]


def test_batched_hypergraph_unbatching():
    """
    Verify unbatch_node_features and unbatch_edge_embeddings reconstruct original sub-graph slices.
    """
    feat_dim = 8
    g0 = _create_sample_graph(num_nodes=2, feat_dim=feat_dim, edges=[([0], [0], 1)])
    g1 = _create_sample_graph(num_nodes=3, feat_dim=feat_dim, edges=[([1, 2], [0, 1], 0)])

    batched = ContinuousGraphBatcher.batch([g0, g1], as_torch=True)

    dummy_node_out = torch.arange(5 * 8, dtype=torch.float32).view(5, 8)
    dummy_edge_out = torch.arange(2 * 4, dtype=torch.float32).view(2, 4)

    node_slices = batched.unbatch_node_features(dummy_node_out)
    edge_slices = batched.unbatch_edge_embeddings(dummy_edge_out)

    assert len(node_slices) == 2
    assert node_slices[0].shape == (2, 8)
    assert node_slices[1].shape == (3, 8)
    assert torch.all(node_slices[0] == dummy_node_out[:2])
    assert torch.all(node_slices[1] == dummy_node_out[2:])

    assert len(edge_slices) == 2
    assert edge_slices[0].shape == (1, 4)
    assert edge_slices[1].shape == (1, 4)


def test_pinned_memory_pool_copy():
    """Verify pre-allocated pinned memory pool safely copies batched graphs into fixed buffers."""
    pool = PinnedHostMemoryPool(max_nodes=100, max_edges=50, max_sources=100, feat_dim=16)

    g0 = _create_sample_graph(num_nodes=5, feat_dim=16, edges=[([0, 1], [0, 1], 2)])
    batched = ContinuousGraphBatcher.batch([g0], as_torch=True)

    pooled_batch = pool.copy_batched_graph(batched)

    assert pooled_batch.total_nodes == 5
    assert pooled_batch.node_features.shape == (5, 16)
    assert torch.allclose(pooled_batch.node_features, g0["node_features"])


def test_device_ring_buffer_slot_rotation():
    """Verify ring buffer rotates across cyclic slots without reallocations."""
    ring = DeviceRingBuffer(num_slots=2, max_nodes=50, max_edges=20, max_sources=50, feat_dim=8, device="cpu")

    g0 = _create_sample_graph(num_nodes=4, feat_dim=8, edges=[([0], [0], 1)])
    batched = ContinuousGraphBatcher.batch([g0], as_torch=True)

    assert ring.current_slot == 0
    slot0 = ring.acquire_slot(batched)
    assert ring.current_slot == 1
    slot1 = ring.acquire_slot(batched)
    assert ring.current_slot == 0
    slot2 = ring.acquire_slot(batched)
    assert ring.current_slot == 1

    # Verify slots point to pre-allocated buffers
    assert slot0.node_features.data_ptr() == ring.slots[0]["node_features"].data_ptr()
    assert slot1.node_features.data_ptr() == ring.slots[1]["node_features"].data_ptr()
    assert slot2.node_features.data_ptr() == ring.slots[0]["node_features"].data_ptr()


def test_graph_memory_pool_end_to_end_backbone_forward():
    """
    End-to-end integration: batch multiple variable-topology graphs through GraphMemoryPool
    and run forward pass through AXMANetBackbone.
    """
    cfg = AxmanConfig.small(d_model=64, num_heads=4, port_dim=32, num_layers=2)
    backbone = AXMANetBackbone(cfg)
    backbone.eval()

    pool = GraphMemoryPool(max_nodes=50, max_edges=30, max_sources=60, feat_dim=64, device="cpu")

    g0 = _create_sample_graph(num_nodes=4, feat_dim=64, edges=[([0, 1], [0, 1], 2), ([2], [0], 3)])
    g1 = _create_sample_graph(num_nodes=6, feat_dim=64, edges=[([1, 2, 3], [0, 1, 2], 4), ([4, 5], [0, 1], 0)])

    # Full batching through pool
    batch = pool.prepare_batch([g0, g1])

    with torch.no_grad():
        out = backbone(
            node_features=batch.node_features,
            edge_targets=batch.edge_targets,
            edge_offsets=batch.edge_offsets,
            edge_sources=batch.edge_sources,
            edge_ports=batch.edge_ports,
        )

    assert out.node_embeddings.shape == (10, 64)
    assert out.edge_embeddings.shape == (4, 64)

    # Unbatch back to individual graph results
    node_res = batch.unbatch_node_features(out.node_embeddings)
    edge_res = batch.unbatch_edge_embeddings(out.edge_embeddings)

    assert len(node_res) == 2
    assert node_res[0].shape == (4, 64)
    assert node_res[1].shape == (6, 64)
    assert len(edge_res) == 2
    assert edge_res[0].shape == (2, 64)
    assert edge_res[1].shape == (2, 64)
