"""
Unit tests for AXMANet Phase 1.5: AXMA Morphism Block (AMB) and Multi-Layer Backbone.
Tests axman.core.blocks.amb_backbone module.
"""

import sys
import unittest
from pathlib import Path

import torch

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

from axman.core.config import AxmanConfig
from axman.core.blocks.amb_backbone import (
    AxmaMorphismBlock,
    AXMANetBackbone,
    BackboneOutput,
)
from axman.core.encoders.type_embedding import TypePrimitiveKind


class TestAxmaMorphismBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = AxmanConfig(
            d_model=64,
            num_layers=1,
            num_heads=4,
            port_dim=32,
            dropout=0.0,
            layer_scale_init=0.1,
        )
        self.block = AxmaMorphismBlock(self.config)

    def test_single_block_forward(self):
        num_nodes = 5
        node_emb = torch.randn(num_nodes, self.config.d_model)

        targets = torch.tensor([2, 4], dtype=torch.long)
        offsets = torch.tensor([0, 2, 4], dtype=torch.long)
        sources = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        ports = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        out_nodes, out_edges, attn = self.block(
            node_embeddings=node_emb,
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
            return_weights=True,
        )

        self.assertEqual(out_nodes.shape, (num_nodes, self.config.d_model))
        self.assertEqual(out_edges.shape, (2, self.config.d_model))
        self.assertIsNotNone(attn)
        self.assertEqual(attn.shape, (4, self.config.num_heads))
        self.assertFalse(torch.isnan(out_nodes).any())
        self.assertFalse(torch.isnan(out_edges).any())


class TestAXMANetBackbone(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = AxmanConfig(
            d_model=64,
            num_layers=3,
            num_heads=4,
            port_dim=32,
            in_features=16,
            dropout=0.0,
        )
        self.backbone = AXMANetBackbone(self.config)

    def _sample_graph(self):
        num_nodes = 7
        kind_ids = torch.tensor([
            TypePrimitiveKind.SORT,
            TypePrimitiveKind.CONST,
            TypePrimitiveKind.APP,
            TypePrimitiveKind.LAMBDA,
            TypePrimitiveKind.FORALL,
            TypePrimitiveKind.BVAR,
            TypePrimitiveKind.MVAR,
        ], dtype=torch.long)
        node_feats = torch.randn(num_nodes, 16)

        # 3 hyperedges:
        # e0: (0, 1) -> 2
        # e1: (2, 3, 4) -> 5
        # e2: (5) -> 6
        targets = torch.tensor([2, 5, 6], dtype=torch.long)
        offsets = torch.tensor([0, 2, 5, 6], dtype=torch.long)
        sources = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        ports = torch.tensor([0, 1, 0, 1, 2, 0], dtype=torch.long)

        return kind_ids, node_feats, targets, offsets, sources, ports

    def test_backbone_forward_shapes(self):
        kinds, feats, targets, offsets, sources, ports = self._sample_graph()

        output = self.backbone(
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
            kind_ids=kinds,
            node_features=feats,
        )

        self.assertIsInstance(output, BackboneOutput)
        self.assertEqual(output.node_embeddings.shape, (7, self.config.d_model))
        self.assertEqual(output.edge_embeddings.shape, (3, self.config.d_model))
        self.assertFalse(torch.isnan(output.node_embeddings).any())
        self.assertFalse(torch.isnan(output.edge_embeddings).any())

    def test_backbone_intermediate_outputs(self):
        kinds, feats, targets, offsets, sources, ports = self._sample_graph()

        output = self.backbone(
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
            kind_ids=kinds,
            node_features=feats,
            output_hidden_states=True,
            output_attentions=True,
        )

        # L + 1 hidden states (embedding + 3 layers)
        self.assertIsNotNone(output.all_hidden_states)
        self.assertEqual(len(output.all_hidden_states), self.config.num_layers + 1)

        # L attention weight tensors
        self.assertIsNotNone(output.all_attentions)
        self.assertEqual(len(output.all_attentions), self.config.num_layers)

    def test_high_arity_hyperedge(self):
        """
        Verify backbone handles arity up to max static tile limit (k = 16).
        """
        num_nodes = 20
        kinds = torch.randint(0, 10, (num_nodes,))

        # Single hyperedge with arity 16: (0..15) -> 16
        targets = torch.tensor([16], dtype=torch.long)
        offsets = torch.tensor([0, 16], dtype=torch.long)
        sources = torch.arange(16, dtype=torch.long)
        ports = torch.arange(16, dtype=torch.long)

        out = self.backbone(
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
            kind_ids=kinds,
        )

        self.assertEqual(out.node_embeddings.shape, (num_nodes, self.config.d_model))
        self.assertEqual(out.edge_embeddings.shape, (1, self.config.d_model))

    def test_end_to_end_gradient_propagation(self):
        kinds, feats, targets, offsets, sources, ports = self._sample_graph()
        feats.requires_grad = True

        output = self.backbone(
            edge_targets=targets,
            edge_offsets=offsets,
            edge_sources=sources,
            edge_ports=ports,
            kind_ids=kinds,
            node_features=feats,
        )

        loss = output.node_embeddings.sum() + output.edge_embeddings.sum()
        loss.backward()

        self.assertIsNotNone(feats.grad)
        # Check gradient in first layer and last layer
        first_layer = self.backbone.layers[0]
        last_layer = self.backbone.layers[-1]
        self.assertIsNotNone(first_layer.fpha.q_proj.weight.grad)
        self.assertIsNotNone(last_layer.fpha.q_proj.weight.grad)
        self.assertIsNotNone(self.backbone.final_norm.weight.grad)

    def test_config_presets(self):
        small_cfg = AxmanConfig.small()
        self.assertEqual(small_cfg.num_layers, 6)
        self.assertEqual(small_cfg.d_model, 256)

        base_cfg = AxmanConfig.base()
        self.assertEqual(base_cfg.num_layers, 12)
        self.assertEqual(base_cfg.d_model, 512)

        large_cfg = AxmanConfig.large()
        self.assertEqual(large_cfg.num_layers, 24)
        self.assertEqual(large_cfg.d_model, 1024)


if __name__ == "__main__":
    unittest.main()
