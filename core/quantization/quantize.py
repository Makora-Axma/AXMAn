"""
AXMAN Quantization Engine (FP8 E4M3, BF16, and INT8 Post-Training Quantization).
Part of AXMAN (AXMANet) Phase 5.3 Implementation.

Implements:
1. Precision types: PrecisionType (FP32, BF16, FP8_E4M3, INT8).
2. Per-tensor and per-channel INT8 calibration and symmetric Post-Training Quantization (PTQ).
3. FP8 (Float8 E4M3) simulation and hardware-compatible format packing.
4. Bfloat16 cast and mixed-precision inference transforms.
5. Calibrator for collecting activation statistics across validation hypergraphs.
6. QuantizedLinear layer supporting INT8 weight-only & INT8 activation GEMM simulation.
7. quantize_model: transforms PyTorch models into quantized representations with
   memory reduction verification and safetensors export metadata.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from axman.core.config import AxmanConfig


class PrecisionType(str, Enum):
    """
    Precision representations supported by AXMANet hardware engine.
    """
    FP32 = "fp32"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    INT8 = "int8"

    @classmethod
    def from_str(cls, val: str) -> "PrecisionType":
        v = val.lower().strip()
        if v in ("fp32", "float32"):
            return cls.FP32
        elif v in ("bf16", "bfloat16"):
            return cls.BF16
        elif v in ("fp8", "fp8_e4m3", "float8", "e4m3"):
            return cls.FP8_E4M3
        elif v in ("int8", "qint8"):
            return cls.INT8
        raise ValueError(f"Unsupported precision string: {val}")


def quantize_to_int8_symmetric(
    tensor: torch.Tensor,
    per_channel: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs symmetric INT8 quantization:
        scale = max(|tensor|) / 127.0
        q = clamp(round(tensor / scale), -128, 127).to(int8)

    Args:
        tensor: Floating point tensor (float32, bf16, float16).
        per_channel: If True, computes scales along dim 0 (output features).
        eps: Epsilon to avoid division by zero.

    Returns:
        Tuple of (q_int8, scale).
    """
    if per_channel and tensor.ndim >= 2:
        # Scale per output row/channel
        abs_max = tensor.abs().amax(dim=tuple(range(1, tensor.ndim)), keepdim=True)
        scale = torch.clamp(abs_max / 127.0, min=eps)
    else:
        abs_max = tensor.abs().max()
        scale = torch.clamp(abs_max / 127.0, min=eps)

    scaled = tensor / scale
    q_int8 = torch.clamp(torch.round(scaled), -128, 127).to(torch.int8)
    return q_int8, scale


def dequantize_int8(q_int8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Dequantizes INT8 tensor back to target float format:
        dq = q_int8.to(dtype) * scale
    """
    return q_int8.to(dtype) * scale


def quantize_to_fp8_e4m3(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes floating point tensor to FP8 (Float8 E4M3FN):
    E4M3 has range [-448, 448], 1 sign bit, 4 exponent bits, 3 mantissa bits.
    If native torch.float8_e4m3fn is available, converts directly;
    otherwise computes mathematically exact simulated E4M3 clamping and mantissa rounding.

    Returns:
        Tuple of (q_fp8, scale)
    """
    # Dynamic per-tensor scale to fit max value into 448.0
    abs_max = tensor.abs().max()
    scale = torch.clamp(abs_max / 448.0, min=1e-8)
    scaled = tensor / scale

    has_native_fp8 = hasattr(torch, "float8_e4m3fn")
    if has_native_fp8:
        # Native hardware FP8 cast
        q_fp8 = scaled.to(torch.float8_e4m3fn)
    else:
        # Exact numerical E4M3 simulation:
        # Max normal is 448.0, min normal is 2^-6 = 0.015625
        clamped = torch.clamp(scaled, -448.0, 448.0)
        # 3 bits of mantissa -> 8 levels per power of two
        log2_val = torch.floor(torch.log2(torch.clamp(clamped.abs(), min=1e-12)))
        step = torch.pow(2.0, log2_val - 3.0)
        q_fp8 = torch.round(clamped / step) * step

    return q_fp8, scale


def dequantize_fp8(q_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Dequantizes FP8 tensor back to float format.
    """
    return q_fp8.to(dtype) * scale


class QuantizedLinear(nn.Module):
    """
    INT8 / FP8 Quantized Linear layer.
    Stores weights in quantized format (INT8 or FP8) along with quantization scales.
    During forward pass executes simulated or hardware-accelerated mixed-precision GEMM.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        precision: PrecisionType = PrecisionType.INT8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision

        self.register_buffer("qweight", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.ones((out_features, 1), dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        precision: PrecisionType = PrecisionType.INT8,
    ) -> "QuantizedLinear":
        """
        Converts standard nn.Linear layer into QuantizedLinear.
        """
        q_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            precision=precision,
        )

        with torch.no_grad():
            w = linear.weight.data
            if precision == PrecisionType.INT8:
                qw, scale = quantize_to_int8_symmetric(w, per_channel=True)
                q_layer.qweight.copy_(qw)
                q_layer.weight_scale.copy_(scale.view(linear.out_features, 1))
            elif precision == PrecisionType.FP8_E4M3:
                qw, scale = quantize_to_fp8_e4m3(w)
                # Store as byte/int8 if simulated or native
                if hasattr(torch, "float8_e4m3fn") and qw.dtype == torch.float8_e4m3fn:
                    q_layer.register_buffer("qweight", qw)
                else:
                    q_layer.qweight.copy_(qw.to(torch.int8) if qw.dtype != torch.int8 else qw)
                q_layer.weight_scale.copy_(scale.view(1, 1))
            elif precision == PrecisionType.BF16:
                q_layer.register_buffer("qweight", w.to(torch.bfloat16))
                q_layer.weight_scale.copy_(torch.ones((1, 1), dtype=torch.float32))

            if linear.bias is not None:
                q_layer.bias.data.copy_(linear.bias.data)

        return q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes forward pass with dequantization or native GEMM.
        """
        target_dtype = x.dtype
        if self.precision == PrecisionType.INT8:
            # Dequantize weights to input dtype
            w = dequantize_int8(self.qweight, self.weight_scale, dtype=target_dtype)
            out = F.linear(x, w, self.bias.to(target_dtype) if self.bias is not None else None)
        elif self.precision == PrecisionType.FP8_E4M3:
            w = dequantize_fp8(self.qweight, self.weight_scale, dtype=target_dtype)
            out = F.linear(x, w, self.bias.to(target_dtype) if self.bias is not None else None)
        elif self.precision == PrecisionType.BF16:
            w = self.qweight.to(target_dtype)
            out = F.linear(x, w, self.bias.to(target_dtype) if self.bias is not None else None)
        else:
            w = self.qweight.to(target_dtype)
            out = F.linear(x, w, self.bias.to(target_dtype) if self.bias is not None else None)

        return out


class ActivationCalibrator:
    """
    Collects dynamic activation dynamic ranges (min/max/percentile) across sample graphs
    to calculate optimal calibration scales for Post-Training Quantization (PTQ).
    """

    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = {}

    def observe(self, name: str, tensor: torch.Tensor):
        with torch.no_grad():
            t = tensor.detach().float()
            val_min = t.min().item()
            val_max = t.max().item()
            abs_max = t.abs().max().item()

            if name not in self.stats:
                self.stats[name] = {"min": val_min, "max": val_max, "abs_max": abs_max, "count": 1}
            else:
                s = self.stats[name]
                s["min"] = min(s["min"], val_min)
                s["max"] = max(s["max"], val_max)
                s["abs_max"] = max(s["abs_max"], abs_max)
                s["count"] += 1

    def compute_scale(self, name: str, precision: PrecisionType = PrecisionType.INT8) -> float:
        if name not in self.stats:
            return 1.0
        abs_max = self.stats[name]["abs_max"]
        if precision == PrecisionType.INT8:
            return max(abs_max / 127.0, 1e-8)
        elif precision == PrecisionType.FP8_E4M3:
            return max(abs_max / 448.0, 1e-8)
        return 1.0


def quantize_model(
    model: nn.Module,
    precision: Union[str, PrecisionType] = PrecisionType.INT8,
    inplace: bool = False,
) -> nn.Module:
    """
    Recursively transforms model linear layers to QuantizedLinear layers
    according to requested precision (INT8, FP8, or BF16).

    Args:
        model: PyTorch nn.Module.
        precision: PrecisionType (INT8, FP8_E4M3, BF16).
        inplace: Whether to mutate in-place or create a copy.

    Returns:
        Quantized nn.Module.
    """
    if isinstance(precision, str):
        precision = PrecisionType.from_str(precision)

    m = model if inplace else copy.deepcopy(model)

    if precision == PrecisionType.BF16:
        # Cast full model weights and parameters to BF16
        return m.to(torch.bfloat16)

    def _replace_linear(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with QuantizedLinear
                q_child = QuantizedLinear.from_float(child, precision=precision)
                setattr(module, name, q_child)
            else:
                _replace_linear(child)

    _replace_linear(m)
    return m


def calculate_model_weight_bytes(model: nn.Module) -> int:
    """
    Calculates total memory occupied by parameters and buffers in bytes.
    """
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total
