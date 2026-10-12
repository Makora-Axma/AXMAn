"""
Unit tests for AXMANet Phase 1.2: Fused Port-Hypergraph Flash Attention (FPHA).
Tests axman.core.blocks.fpha module.
"""

import sys
import unittest
from pathlib import Path

import torch

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

from axman.core.blocks.fpha import FusedPortHypergraphAttention


class TestFusedPortHypergraphAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.num_heads = 4
        self.port_dim = 32

        self.fpha = FusedPortHypergraphAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            port_dim=self.port_dim,
            dropout=0.0,
            bias=True,
        )

    def _create_sample_hypergraph(self):
        """
        Creates a synthetic port-hypergraph:
        Nodes: 6 nodes (0..5)
        Hyperedges: 3 hyperedges (e0, e1, e2)
          e0: (0, 1, 2) -> target 3 (arity 3)
          e1: (3, 4) -> target 5 (arity 2)
          e2: (0) -> target 5 (arity 1, multiple edges pointing to node 5)
        """
        num_nodes = 6
        node_embeddings = torch.randn(num_nodes, self.d_model, requires_grad=True)

        edge_targets = torch.tensor([3, 5, 5], dtype=torch.long)
        edge_offsets = torch.tensor([0, 3, 5, 6], dtype=torch.long)
        edge_sources = torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long)
        edge_ports = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long)

        return node_embeddings, edge_targets, edge_offsets, edge_sources, edge_ports

    def test_forward_dimensions(self):
        nodes, targets, offsets, sources, ports = self._create_sample_hypergraph()

        node_updates, edge_embeds, attn_weights = self.fpha(
            node_embeddings=nodes,
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
            return_weights=True,
        )

        self.assertEqual(node_updates.shape, (6, self.d_model))
        self.assertEqual(edge_embeds.shape, (3, self.d_model))
        self.assertEqual(attn_weights.shape, (6, self.num_heads))
        self.assertFalse(torch.isnan(node_updates).any())
        self.assertFalse(torch.isnan(edge_embeds).any())

    def test_attention_weights_partition_of_unity(self):
        """
        Attention weights for premises of each hyperedge must sum to 1.0 per head.
        """
        nodes, targets, offsets, sources, ports = self._create_sample_hypergraph()

        _, _, attn_weights = self.fpha(
            node_embeddings=nodes,
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
            return_weights=True,
        )

        # e0 has premises 0, 1, 2 (offsets 0..3)
        e0_sum = attn_weights[0:3].sum(dim=0)
        self.assertTrue(torch.allclose(e0_sum, torch.ones(self.num_heads), atol=1e-5))

        # e1 has premises 3, 4 (offsets 3..5)
        e1_sum = attn_weights[3:5].sum(dim=0)
        self.assertTrue(torch.allclose(e1_sum, torch.ones(self.num_heads), atol=1e-5))

        # e2 has premise 5 (offsets 5..6, arity 1)
        e2_sum = attn_weights[5:6].sum(dim=0)
        self.assertTrue(torch.allclose(e2_sum, torch.ones(self.num_heads), atol=1e-5))

    def test_isolated_nodes_remain_zero(self):
        """
        Nodes with no incoming hyperedges (nodes 0, 1, 2, 4) must have zero updates before projection bias.
        """
        fpha_no_bias = FusedPortHypergraphAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            port_dim=self.port_dim,
            bias=False,
        )
        nodes, targets, offsets, sources, ports = self._create_sample_hypergraph()
        node_updates, _, _ = fpha_no_bias(
            node_embeddings=nodes,
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
        )

        # Nodes 0, 1, 2 are not targets of any hyperedge
        self.assertTrue(torch.allclose(node_updates[0], torch.zeros(self.d_model)))
        self.assertTrue(torch.allclose(node_updates[1], torch.zeros(self.d_model)))
        self.assertTrue(torch.allclose(node_updates[2], torch.zeros(self.d_model)))

        # Node 3 and Node 5 are targets, must have non-zero updates
        self.assertFalse(torch.allclose(node_updates[3], torch.zeros(self.d_model)))
        self.assertFalse(torch.allclose(node_updates[5], torch.zeros(self.d_model)))

    def test_port_sensitivity(self):
        """
        Swapping ports must change the attention output (port-equivariance / non-commutativity).
        """
        nodes, targets, offsets, sources, _ = self._create_sample_hypergraph()

        # Config A: ports [0, 1, 2, ...]
        ports_a = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long)
        out_a, _, _ = self.fpha(nodes, targets, offsets, sources, ports_a)

        # Config B: reversed ports in e0: [2, 1, 0, ...]
        ports_b = torch.tensor([2, 1, 0, 0, 1, 0], dtype=torch.long)
        out_b, _, _ = self.fpha(nodes, targets, offsets, sources, ports_b)

        diff = (out_a - out_b).abs().sum().item()
        self.assertGreater(diff, 1e-3, "Attention must be sensitive to port permutation!")

    def test_gradient_propagation(self):
        nodes, targets, offsets, sources, ports = self._create_sample_hypergraph()

        node_updates, edge_embeds, _ = self.fpha(
            node_embeddings=nodes,
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
        )

        loss = node_updates.sum() + edge_embeds.sum()
        loss.backward()

        self.assertIsNotNone(nodes.grad)
        self.assertIsNotNone(self.fpha.q_proj.weight.grad)
        self.assertIsNotNone(self.fpha.k_proj.weight.grad)
        self.assertIsNotNone(self.fpha.v_proj.weight.grad)
        self.assertIsNotNone(self.fpha.premise_fusion.weight.grad)
        self.assertIsNotNone(self.fpha.out_proj.weight.grad)
        self.assertIsNotNone(self.fpha.edge_proj.weight.grad)

    def test_empty_graph_graceful_handling(self):
        nodes = torch.randn(4, self.d_model)
        targets = torch.empty(0, dtype=torch.long)
        offsets = torch.tensor([0], dtype=torch.long)
        sources = torch.empty(0, dtype=torch.long)
        ports = torch.empty(0, dtype=torch.long)

        node_updates, edge_embeds, weights = self.fpha(
            nodes, targets, offsets, sources, ports
        )

        self.assertEqual(node_updates.shape, (4, self.d_model))
        self.assertEqual(edge_embeds.shape, (0, self.d_model))
        self.assertIsNone(weights)
        self.assertTrue(torch.allclose(node_updates, torch.zeros_like(node_updates)))

    def test_head_divisibility_validation(self):
        with self.assertRaises(ValueError):
            FusedPortHypergraphAttention(d_model=65, num_heads=8)


if __name__ == "__main__":
    unittest.main()
