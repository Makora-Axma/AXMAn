"""
Unit tests for AXMANet Phase 1.4: SwiGLU Hyper-FeedForward and Pre-RMSNorm.
Tests axman.core.blocks.ffn module.
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

from axman.core.blocks.ffn import SwiGLUFeedForward, HyperFeedForwardBlock
from axman.core.encoders.type_embedding import RMSNorm


class TestSwiGLUFeedForward(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.ffn = SwiGLUFeedForward(d_model=self.d_model, dropout=0.0)

    def test_forward_dimensions(self):
        x = torch.randn(8, self.d_model)
        out = self.ffn(x)
        self.assertEqual(out.shape, (8, self.d_model))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_multidimensional_batch(self):
        x = torch.randn(2, 4, 10, self.d_model)
        out = self.ffn(x)
        self.assertEqual(out.shape, (2, 4, 10, self.d_model))

    def test_swiglu_arithmetic_precision(self):
        """
        Verify exact arithmetic matches the mathematical formula:
        SwiGLU(x) = (silu(x W_gate) * (x W_up)) W_down
        """
        x = torch.randn(4, self.d_model)

        # Module execution
        mod_out = self.ffn(x)

        # Manual manual tensor computation
        gate = F.silu(F.linear(x, self.ffn.w_gate.weight, self.ffn.w_gate.bias))
        up = F.linear(x, self.ffn.w_up.weight, self.ffn.w_up.bias)
        manual_out = F.linear(gate * up, self.ffn.w_down.weight, self.ffn.w_down.bias)

        diff = (mod_out - manual_out).abs().max().item()
        self.assertLess(diff, 1e-6, "SwiGLU output does not match theoretical formula!")

    def test_intermediate_dimension_rounding(self):
        # 64 * 8/3 = 170.66 -> rounded to multiple of 64 is 192
        self.assertEqual(self.ffn.intermediate_dim, 192)

        # Custom intermediate dim
        custom_ffn = SwiGLUFeedForward(d_model=64, intermediate_dim=256)
        self.assertEqual(custom_ffn.intermediate_dim, 256)

    def test_gradient_propagation(self):
        x = torch.randn(5, self.d_model, requires_grad=True)
        out = self.ffn(x)
        loss = out.pow(2).sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.ffn.w_gate.weight.grad)
        self.assertIsNotNone(self.ffn.w_up.weight.grad)
        self.assertIsNotNone(self.ffn.w_down.weight.grad)


class TestHyperFeedForwardBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 64
        self.block = HyperFeedForwardBlock(d_model=self.d_model, dropout=0.0)

    def test_block_forward_and_residual(self):
        x = torch.randn(12, self.d_model)
        out = self.block(x)
        self.assertEqual(out.shape, (12, self.d_model))

        # Output must be non-trivially different from input due to FFN transformation
        self.assertFalse(torch.allclose(out, x))

    def test_zero_input_preservation(self):
        """
        Zero input passed through RMSNorm with eps does not blow up or produce NaN.
        """
        zero_x = torch.zeros(5, self.d_model)
        out = self.block(zero_x)
        self.assertEqual(out.shape, (5, self.d_model))
        self.assertFalse(torch.isnan(out).any())

    def test_gradient_flow_through_residual_and_norm(self):
        x = torch.randn(6, self.d_model, requires_grad=True)
        out = self.block(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.block.norm.weight.grad)
        self.assertIsNotNone(self.block.ffn.w_gate.weight.grad)


if __name__ == "__main__":
    unittest.main()
