"""
Unit and integration tests for Task 5.3: Model Quantization (INT8 PTQ, FP8 E4M3, and BF16).
Tests axman.core.quantization module.
"""

import sys
from pathlib import Path

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

import pytest
import torch
import torch.nn as nn

from axman.core.config import AxmanConfig
from axman.core.blocks.amb_backbone import AXMANetBackbone
from axman.core.quantization import (
    PrecisionType,
    QuantizedLinear,
    ActivationCalibrator,
    quantize_to_int8_symmetric,
    dequantize_int8,
    quantize_to_fp8_e4m3,
    dequantize_fp8,
    quantize_model,
    calculate_model_weight_bytes,
)


def test_int8_symmetric_quantization_numerical_precision():
    """
    Verify symmetric INT8 quantization and dequantization preserves distribution
    with expected low quantization error.
    """
    torch.manual_seed(42)
    # Tensor with standard normal distribution
    t = torch.randn(64, 128)
    q_int8, scale = quantize_to_int8_symmetric(t, per_channel=True)

    assert q_int8.dtype == torch.int8
    assert q_int8.shape == (64, 128)
    assert scale.shape == (64, 1)

    # All values in [-128, 127]
    assert q_int8.min().item() >= -128
    assert q_int8.max().item() <= 127

    # Dequantize
    t_recon = dequantize_int8(q_int8, scale)
    # Cosine similarity between original and reconstructed should be extremely high (> 0.999)
    cos_sim = torch.cosine_similarity(t.view(-1), t_recon.view(-1), dim=0)
    assert cos_sim.item() > 0.999

    # Max absolute error bounded by scale
    abs_err = (t - t_recon).abs()
    assert abs_err.max().item() <= scale.max().item() + 1e-4


def test_fp8_e4m3_quantization_and_range():
    """
    Verify FP8 E4M3 scaling, range clamping [-448, 448], and reconstruction.
    """
    torch.manual_seed(123)
    t = torch.randn(32, 64) * 10.0
    q_fp8, scale = quantize_to_fp8_e4m3(t)

    t_recon = dequantize_fp8(q_fp8, scale)
    cos_sim = torch.cosine_similarity(t.view(-1), t_recon.view(-1), dim=0)
    assert cos_sim.item() > 0.995


def test_quantized_linear_layer_forward():
    """
    Verify QuantizedLinear layer converts from nn.Linear and executes forward GEMM.
    """
    torch.manual_seed(77)
    linear = nn.Linear(32, 16, bias=True)
    x = torch.randn(4, 32)

    # 1. FP32 reference forward
    out_fp32 = linear(x)

    # 2. INT8 QuantizedLinear forward
    q_linear_int8 = QuantizedLinear.from_float(linear, precision=PrecisionType.INT8)
    out_int8 = q_linear_int8(x)

    assert out_int8.shape == (4, 16)
    # Quantized linear output should be very close to reference
    cos_sim = torch.cosine_similarity(out_fp32.view(-1), out_int8.view(-1), dim=0)
    assert cos_sim.item() > 0.995

    # 3. FP8 QuantizedLinear forward
    q_linear_fp8 = QuantizedLinear.from_float(linear, precision=PrecisionType.FP8_E4M3)
    out_fp8 = q_linear_fp8(x)
    assert out_fp8.shape == (4, 16)
    cos_sim_fp8 = torch.cosine_similarity(out_fp32.view(-1), out_fp8.view(-1), dim=0)
    assert cos_sim_fp8.item() > 0.99


def test_activation_calibrator():
    """
    Verify ActivationCalibrator accumulates activation ranges and computes proper PTQ scales.
    """
    calibrator = ActivationCalibrator()

    t1 = torch.tensor([-5.0, 10.0, 2.0])
    t2 = torch.tensor([-12.7, 3.0, 8.0])

    calibrator.observe("layer1_act", t1)
    calibrator.observe("layer1_act", t2)

    scale_int8 = calibrator.compute_scale("layer1_act", PrecisionType.INT8)
    # Abs max is 12.7 -> scale = 12.7 / 127 = 0.1
    assert pytest.approx(scale_int8, rel=1e-3) == 0.1

    scale_fp8 = calibrator.compute_scale("layer1_act", PrecisionType.FP8_E4M3)
    # Abs max is 12.7 -> scale = 12.7 / 448
    assert pytest.approx(scale_fp8, rel=1e-3) == (12.7 / 448.0)


def test_quantize_model_memory_reduction_and_backbone_forward():
    """
    Verify quantize_model recursively quantizes AXMANetBackbone, achieves ~4x weight memory reduction
    for linear weights in INT8, and runs forward pass successfully.
    """
    cfg = AxmanConfig.small(d_model=64, num_heads=4, port_dim=32, num_layers=2)
    model = AXMANetBackbone(cfg)
    model.eval()

    orig_bytes = calculate_model_weight_bytes(model)

    # 1. Quantize model to INT8
    q_model_int8 = quantize_model(model, precision=PrecisionType.INT8, inplace=False)
    q_bytes_int8 = calculate_model_weight_bytes(q_model_int8)

    # Substantial memory reduction (linear layer weights are 1 byte instead of 4 bytes)
    assert q_bytes_int8 < orig_bytes
    reduction_pct = (1.0 - q_bytes_int8 / orig_bytes) * 100.0
    assert reduction_pct > 40.0  # At least 40% reduction for the whole composite model including embeddings

    # 2. Forward pass with quantized backbone
    num_nodes = 5
    num_edges = 2
    node_features = torch.randn(num_nodes, 64)
    edge_targets = torch.tensor([1, 3], dtype=torch.long)
    edge_offsets = torch.tensor([0, 2, 4], dtype=torch.long)
    edge_sources = torch.tensor([0, 2, 1, 4], dtype=torch.long)
    edge_ports = torch.tensor([0, 1, 0, 1], dtype=torch.int32)

    with torch.no_grad():
        out_orig = model(
            node_features=node_features,
            edge_targets=edge_targets,
            edge_offsets=edge_offsets,
            edge_sources=edge_sources,
            edge_ports=edge_ports,
        )
        out_q = q_model_int8(
            node_features=node_features,
            edge_targets=edge_targets,
            edge_offsets=edge_offsets,
            edge_sources=edge_sources,
            edge_ports=edge_ports,
        )

    assert out_q.node_embeddings.shape == out_orig.node_embeddings.shape
    assert out_q.edge_embeddings.shape == out_orig.edge_embeddings.shape
    # Check that output representations remain strongly correlated
    cos_nodes = torch.cosine_similarity(
        out_orig.node_embeddings.view(-1),
        out_q.node_embeddings.view(-1),
        dim=0,
    )
    assert cos_nodes.item() > 0.95


def test_quantize_model_bf16_conversion():
    """Verify quantize_model with BF16 precision converts all parameters to bfloat16."""
    cfg = AxmanConfig.small(d_model=32, num_heads=2, port_dim=16, num_layers=1)
    model = AXMANetBackbone(cfg)
    bf16_model = quantize_model(model, precision=PrecisionType.BF16)

    for p in bf16_model.parameters():
        assert p.dtype == torch.bfloat16
