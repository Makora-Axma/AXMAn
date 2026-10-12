"""
Unit tests for AXMANet Phase 1.3: Categorical Morphism Projection (CMP).
Tests axman.core.blocks.cmp module.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

from axman.core.blocks.cmp import CategoricalMorphismProjection


class TestCategoricalMorphismProjection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.rank = 128
        self.cmp = CategoricalMorphismProjection(
            d_model=self.d_model,
            rank=self.rank,
            use_spectral_norm=True,
            dropout=0.0,
            gate_init=0.0,
        )

    def test_unary_forward_shape(self):
        x = torch.randn(10, self.d_model)
        out = self.cmp(x)
        self.assertEqual(out.shape, (10, self.d_model))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_unary_multidimensional_batch(self):
        x = torch.randn(2, 5, 8, self.d_model)
        out = self.cmp(x)
        self.assertEqual(out.shape, (2, 5, 8, self.d_model))

    def test_compose_morphisms_shape(self):
        f = torch.randn(16, self.d_model)
        g = torch.randn(16, self.d_model)
        gf = self.cmp.compose(f, g)
        self.assertEqual(gf.shape, (16, self.d_model))
        self.assertFalse(torch.isnan(gf).any())

    def test_identity_morphism_preservation(self):
        """
        When initialized with low gate (identity residual dominating),
        composition with identity morphism preserves the input morphism direction.
        """
        cmp_id = CategoricalMorphismProjection(
            d_model=self.d_model,
            rank=self.rank,
            use_spectral_norm=True,
            gate_init=-6.0,  # sigmoid(-6) ~ 0.0025 -> base_residual 0.5*(f+g) dominates
        )
        f = torch.randn(8, self.d_model)
        # In base residual, compose(f, f) = 0.5*(f + f) = f
        ff = cmp_id.compose(f, f)
        cos_sim = F.cosine_similarity(f, ff, dim=-1)
        self.assertTrue(torch.all(cos_sim > 0.99))

    def test_approximate_associativity(self):
        """
        Test that composition exhibits algebraic stability:
        (h o g) o f vs h o (g o f) maintains high alignment.
        """
        cmp_strict = CategoricalMorphismProjection(
            d_model=self.d_model,
            rank=self.rank,
            use_spectral_norm=True,
            gate_init=-2.0,
        )
        f = torch.randn(4, self.d_model)
        g = torch.randn(4, self.d_model)
        h = torch.randn(4, self.d_model)

        # Left-associated: (h o g) o f
        hg = cmp_strict.compose(g, h)
        left = cmp_strict.compose(f, hg)

        # Right-associated: h o (g o f)
        gf = cmp_strict.compose(f, g)
        right = cmp_strict.compose(gf, h)

        cos_sim = F.cosine_similarity(left, right, dim=-1)
        # Cosine similarity between compositions should be positive and strongly correlated
        self.assertTrue(torch.all(cos_sim > 0.75))

    def test_spectral_norm_active(self):
        """
        Verify that spectral normalization is actively attached to projection weights.
        """
        # In PyTorch parametrizations, parametrizations.weight exists when spectral_norm is applied
        self.assertTrue(hasattr(self.cmp.proj_a, "parametrizations"))
        self.assertTrue(hasattr(self.cmp.proj_b, "parametrizations"))
        self.assertTrue(hasattr(self.cmp.proj_out, "parametrizations"))
        self.assertTrue(hasattr(self.cmp.linear_comp, "parametrizations"))
        self.assertTrue(hasattr(self.cmp.unary_linear, "parametrizations"))

    def test_gradient_backprop(self):
        f = torch.randn(5, self.d_model, requires_grad=True)
        g = torch.randn(5, self.d_model, requires_grad=True)

        gf = self.cmp.compose(f, g)
        unary = self.cmp(f)
        loss = gf.sum() + unary.sum()
        loss.backward()

        self.assertIsNotNone(f.grad)
        self.assertIsNotNone(g.grad)
        self.assertIsNotNone(self.cmp.gate.grad)
        self.assertIsNotNone(self.cmp.proj_a.parametrizations.weight.original.grad)
        self.assertIsNotNone(self.cmp.proj_out.parametrizations.weight.original.grad)

    def test_invalid_dimension_raises(self):
        bad_x = torch.randn(5, self.d_model + 1)
        with self.assertRaises(ValueError):
            self.cmp(bad_x)

        with self.assertRaises(ValueError):
            self.cmp.compose(torch.randn(5, self.d_model), bad_x)


if __name__ == "__main__":
    unittest.main()
