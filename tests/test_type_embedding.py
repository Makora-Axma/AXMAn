"""
Unit tests for AXMANet Phase 1.1: Type Theory and Port Embedders.
Tests axman.core.encoders.type_embedding module.
"""

import sys
import unittest
from pathlib import Path

import torch

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

from axman.core.encoders.type_embedding import (
    TypePrimitiveKind,
    TypePrimitiveEmbedding,
    DeBruijnPortEmbedding,
    NodeAndPortEncoder,
    RMSNorm,
)


class TestTypePrimitiveEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 128
        self.num_kinds = 32
        self.in_features = 64

    def test_kind_embedding_only(self):
        model = TypePrimitiveEmbedding(d_model=self.d_model, num_kinds=self.num_kinds)
        kinds = torch.tensor([
            TypePrimitiveKind.SORT,
            TypePrimitiveKind.BVAR,
            TypePrimitiveKind.LAMBDA,
            TypePrimitiveKind.FORALL,
        ], dtype=torch.long)

        out = model(kind_ids=kinds)
        self.assertEqual(out.shape, (4, self.d_model))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_continuous_features_only(self):
        model = TypePrimitiveEmbedding(
            d_model=self.d_model,
            in_features=self.in_features,
        )
        node_feats = torch.randn(10, self.in_features)
        out = model(node_features=node_feats)
        self.assertEqual(out.shape, (10, self.d_model))
        self.assertFalse(torch.isnan(out).any())

    def test_combined_features_and_kinds(self):
        model = TypePrimitiveEmbedding(
            d_model=self.d_model,
            num_kinds=self.num_kinds,
            in_features=self.in_features,
        )
        num_nodes = 8
        kinds = torch.randint(0, self.num_kinds, (num_nodes,))
        feats = torch.randn(num_nodes, self.in_features)

        out = model(kind_ids=kinds, node_features=feats)
        self.assertEqual(out.shape, (num_nodes, self.d_model))

        # Check gradient flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.kind_embedding.weight.grad)
        self.assertIsNotNone(model.feature_proj.weight.grad)

    def test_clamping_out_of_bounds_kinds(self):
        model = TypePrimitiveEmbedding(d_model=self.d_model, num_kinds=self.num_kinds)
        # Pass kinds outside valid range
        kinds = torch.tensor([-5, 0, 10, 999], dtype=torch.long)
        out = model(kind_ids=kinds)
        self.assertEqual(out.shape, (4, self.d_model))
        self.assertFalse(torch.isnan(out).any())


class TestDeBruijnPortEmbedding(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.port_dim = 64
        self.ports = torch.tensor([0, 1, 2, 3, 7, 15, 20], dtype=torch.long)

    def test_learned_mode(self):
        emb_module = DeBruijnPortEmbedding(port_dim=self.port_dim, mode="learned")
        out = emb_module(self.ports)
        self.assertEqual(out.shape, (len(self.ports), self.port_dim))
        # Ensure different ports yield distinct representations
        diff = (out[0] - out[1]).abs().sum().item()
        self.assertGreater(diff, 1e-4)

    def test_sinusoidal_mode(self):
        emb_module = DeBruijnPortEmbedding(port_dim=self.port_dim, mode="sinusoidal")
        out = emb_module(self.ports)
        self.assertEqual(out.shape, (len(self.ports), self.port_dim))
        self.assertFalse(torch.isnan(out).any())
        # Check orthogonality/distinctness of harmonics
        cos_sim = torch.cosine_similarity(out[0:1], out[1:2])
        self.assertLess(cos_sim.item(), 0.999)

    def test_hybrid_mode(self):
        emb_module = DeBruijnPortEmbedding(port_dim=self.port_dim, mode="hybrid")
        out = emb_module(self.ports)
        self.assertEqual(out.shape, (len(self.ports), self.port_dim))

        # Test gradient propagation
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(emb_module.learned_embed.weight.grad)
        self.assertIsNotNone(emb_module.sin_proj.weight.grad)

    def test_parity_odd_dimensions(self):
        emb_odd = DeBruijnPortEmbedding(port_dim=33, mode="sinusoidal")
        out_odd = emb_odd(self.ports)
        self.assertEqual(out_odd.shape, (len(self.ports), 33))


class TestNodeAndPortEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 128
        self.port_dim = 64
        self.num_nodes = 5
        self.num_sources = 6

        self.encoder = NodeAndPortEncoder(
            d_model=self.d_model,
            port_dim=self.port_dim,
            num_kinds=32,
            in_features=32,
            port_mode="hybrid",
        )

    def test_joint_forward(self):
        kinds = torch.tensor([0, 1, 5, 6, 7], dtype=torch.long)
        feats = torch.randn(self.num_nodes, 32)
        ports = torch.tensor([0, 1, 0, 1, 2, 0], dtype=torch.long)

        node_emb, port_emb = self.encoder(
            kind_ids=kinds,
            node_features=feats,
            edge_ports=ports,
        )

        self.assertEqual(node_emb.shape, (self.num_nodes, self.d_model))
        self.assertEqual(port_emb.shape, (self.num_sources, self.port_dim))

    def test_enrich_sources_modes(self):
        kinds = torch.randint(0, 16, (self.num_nodes,))
        node_emb, _ = self.encoder(kind_ids=kinds)

        sources = torch.tensor([0, 1, 1, 2, 3, 4], dtype=torch.long)
        ports = torch.tensor([0, 1, 0, 1, 2, 0], dtype=torch.long)

        # 1. concat_proj mode: projects to d_model
        enriched_proj = self.encoder.enrich_sources(node_emb, sources, ports, mode="concat_proj")
        self.assertEqual(enriched_proj.shape, (self.num_sources, self.d_model))

        # 2. add mode: projects port to d_model and adds
        enriched_add = self.encoder.enrich_sources(node_emb, sources, ports, mode="add")
        self.assertEqual(enriched_add.shape, (self.num_sources, self.d_model))

        # 3. concat mode: raw concatenation [d_model + port_dim]
        enriched_concat = self.encoder.enrich_sources(node_emb, sources, ports, mode="concat")
        self.assertEqual(enriched_concat.shape, (self.num_sources, self.d_model + self.port_dim))

    def test_gradient_backprop(self):
        kinds = torch.randint(0, 16, (self.num_nodes,))
        node_emb, _ = self.encoder(kind_ids=kinds)
        sources = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        ports = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        enriched = self.encoder.enrich_sources(node_emb, sources, ports, mode="concat_proj")
        loss = enriched.pow(2).mean()
        loss.backward()

        self.assertIsNotNone(self.encoder.node_encoder.kind_embedding.weight.grad)
        self.assertIsNotNone(self.encoder.port_encoder.learned_embed.weight.grad)
        self.assertIsNotNone(self.encoder.fuse_proj.weight.grad)


class TestRMSNorm(unittest.TestCase):
    def test_rmsnorm_properties(self):
        dim = 64
        norm = RMSNorm(dim)
        x = torch.randn(4, 10, dim) * 5.0 + 3.0
        y = norm(x)

        self.assertEqual(y.shape, x.shape)
        # Check that RMS is approximately 1.0 along the last dimension
        rms = torch.sqrt(torch.mean(y.pow(2), dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
