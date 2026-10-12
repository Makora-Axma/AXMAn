"""
AXMANet Python Package (axman).
"""

from .model import (
    AxmanNet,
    Hypergraph,
    HypergraphBuilder,
    ModelConfig,
    InferenceOutput,
    AxmanError,
    version,
)

__version__ = version()

__all__ = [
    "AxmanNet",
    "Hypergraph",
    "HypergraphBuilder",
    "ModelConfig",
    "InferenceOutput",
    "AxmanError",
    "version",
]
