"""
Safetensors serialization and export utilities for AXMANet (AXMA-N).
Part of AXMAN Phase 3.3 Implementation.

Exports complete PyTorch AXMANet models into HuggingFace .safetensors format
with metadata describing model configuration (d_model, num_layers, num_heads, etc.).
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import safetensors.torch
import torch
import torch.nn as nn

from axman.core.config import AxmanConfig


def export_model_to_safetensors(
    model: nn.Module,
    config: AxmanConfig,
    output_path: Union[str, Path],
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Exports AXMANet model state dict into a standardized .safetensors archive.

    Args:
        model: PyTorch nn.Module or backbone/heads composite model.
        config: AxmanConfig instance detailing architecture hyperparameters.
        output_path: Target path for the .safetensors file.
        extra_metadata: Optional dictionary of additional string key-value metadata.

    Returns:
        Path to the saved safetensors file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Prepare configuration metadata
    metadata = {
        "format": "axmanet-safetensors",
        "version": "1.0.0",
        "d_model": str(config.d_model),
        "num_layers": str(config.num_layers),
        "num_heads": str(config.num_heads),
        "port_dim": str(config.port_dim),
        "num_kinds": str(config.num_kinds),
        "in_features": str(config.in_features if config.in_features is not None else config.d_model),
        "max_ports": str(config.max_ports),
        "port_mode": str(config.port_mode),
        "eps": str(config.eps),
        "precision": str(getattr(config, "precision", "fp32")),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    # 2. Extract contiguous float32 state dict tensors
    raw_state_dict = model.state_dict()
    clean_state_dict: Dict[str, torch.Tensor] = {}

    for k, v in raw_state_dict.items():
        # Ensure contiguous on CPU
        clean_state_dict[k] = v.detach().cpu().contiguous()

    # 3. Save via safetensors
    safetensors.torch.save_file(clean_state_dict, str(out_file), metadata=metadata)
    return out_file


def load_safetensors_metadata(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Fast extraction of header metadata from safetensors file without loading weights.
    """
    path = Path(file_path)
    with open(path, "rb") as f:
        hdr_len_bytes = f.read(8)
        if len(hdr_len_bytes) < 8:
            raise ValueError(f"Corrupted safetensors file: {file_path}")
        hdr_len = int.from_bytes(hdr_len_bytes, byteorder="little")
        hdr_json = f.read(hdr_len).decode("utf-8")
        header = json.loads(hdr_json)
        return header.get("__metadata__", {})
