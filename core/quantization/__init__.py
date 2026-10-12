"""
AXMAN Quantization Package.
Provides FP8 (E4M3), BF16, and INT8 Post-Training Quantization (PTQ) utilities.
"""

from axman.core.quantization.quantize import (
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

__all__ = [
    "PrecisionType",
    "QuantizedLinear",
    "ActivationCalibrator",
    "quantize_to_int8_symmetric",
    "dequantize_int8",
    "quantize_to_fp8_e4m3",
    "dequantize_fp8",
    "quantize_model",
    "calculate_model_weight_bytes",
]
