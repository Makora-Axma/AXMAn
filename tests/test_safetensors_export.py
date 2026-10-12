"""
Tests for Task 3.3: Safetensors export and metadata inspection.
"""

import json
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

from axman.core.config import AxmanConfig
from axman.core.blocks.amb_backbone import AXMANetBackbone
from axman.core.runtime.safetensors_export import (
    export_model_to_safetensors,
    load_safetensors_metadata,
)


class SimpleAXMANet(nn.Module):
    def __init__(self, config: AxmanConfig):
        super().__init__()
        self.config = config
        self.backbone = AXMANetBackbone(config)


def test_export_and_metadata_roundtrip(tmp_path: Path):
    cfg = AxmanConfig.small()
    cfg.in_features = 64
    model = SimpleAXMANet(cfg)
    model.eval()

    target_path = tmp_path / "model_export.safetensors"
    exported_path = export_model_to_safetensors(
        model=model,
        config=cfg,
        output_path=target_path,
        extra_metadata={"custom_tag": "test_verification"},
    )

    assert exported_path.exists()
    assert exported_path.stat().st_size > 0

    meta = load_safetensors_metadata(exported_path)
    assert meta["format"] == "axmanet-safetensors"
    assert meta["version"] == "1.0.0"
    assert meta["d_model"] == "256"
    assert meta["num_layers"] == "6"
    assert meta["num_heads"] == "8"
    assert meta["custom_tag"] == "test_verification"


def test_safetensors_c_loader_compatibility(tmp_path: Path):
    cfg = AxmanConfig.small()
    cfg.in_features = 64
    model = SimpleAXMANet(cfg)

    target_path = tmp_path / "model_export.safetensors"
    export_model_to_safetensors(model=model, config=cfg, output_path=target_path)

    # Verify tensor payload can be read as torch tensors
    import safetensors.torch
    tensors = safetensors.torch.load_file(str(target_path))
    assert len(tensors) > 0
    # Every tensor should be float32 and contiguous
    for k, t in tensors.items():
        assert t.dtype == torch.float32
        assert t.is_contiguous()
