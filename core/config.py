"""
Configuration dataclasses for AXMANet (AXMA-N).
Part of AXMAN (AXMANet) Phase 1.5 Implementation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AxmanConfig:
    """
    Model configuration parameters for AXMANet core and backbone.
    """
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    port_dim: int = 64
    num_kinds: int = 32
    in_features: Optional[int] = None
    intermediate_dim: Optional[int] = None
    dropout: float = 0.0
    max_ports: int = 64
    port_mode: str = "hybrid"
    use_spectral_norm: bool = True
    layer_scale_init: Optional[float] = 1e-2
    eps: float = 1e-6
    precision: str = "fp32"

    @classmethod
    def small(cls, **kwargs) -> "AxmanConfig":
        """
        AXMAN-Small configuration preset (6 layers, d_model=256, 8 heads).
        Ideal for CPU inference and fast tactic search.
        """
        cfg = cls(
            d_model=256,
            num_layers=6,
            num_heads=8,
            port_dim=64,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def base(cls, **kwargs) -> "AxmanConfig":
        """
        AXMAN-Base configuration preset (12 layers, d_model=512, 16 heads).
        Balanced architecture for large Mathlib formal environments.
        """
        cfg = cls(
            d_model=512,
            num_layers=12,
            num_heads=16,
            port_dim=128,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def large(cls, **kwargs) -> "AxmanConfig":
        """
        AXMAN-Large configuration preset (24 layers, d_model=1024, 16 heads).
        High-capacity model for complex Putnam/IMO reasoning and lemma abduction.
        """
        cfg = cls(
            d_model=1024,
            num_layers=24,
            num_heads=16,
            port_dim=256,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg
