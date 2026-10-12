"""
AXMAN (AXMANet) Core Neural Network Library
Axiomatic eXploration & Morphism Architecture Network
"""

from axman.core.config import AxmanConfig
from axman.core.blocks.amb_backbone import AXMANetBackbone, AxmaMorphismBlock, BackboneOutput
from axman.core.blocks.fpha import FusedPortHypergraphAttention
from axman.core.blocks.cmp import CategoricalMorphismProjection
from axman.core.blocks.ffn import SwiGLUFeedForward, HyperFeedForwardBlock
from axman.core.encoders.type_embedding import (
    TypePrimitiveKind,
    TypePrimitiveEmbedding,
    DeBruijnPortEmbedding,
    NodeAndPortEncoder,
    RMSNorm,
)

__version__ = "1.0.0"

__all__ = [
    "AxmanConfig",
    "AXMANetBackbone",
    "AxmaMorphismBlock",
    "BackboneOutput",
    "FusedPortHypergraphAttention",
    "CategoricalMorphismProjection",
    "SwiGLUFeedForward",
    "HyperFeedForwardBlock",
    "TypePrimitiveKind",
    "TypePrimitiveEmbedding",
    "DeBruijnPortEmbedding",
    "NodeAndPortEncoder",
    "RMSNorm",
    "MorphismPolicyHead",
    "PolicyOutput",
    "TacticConstructorKind",
    "DistanceToClosureValueHead",
    "MultiHeadAttentionPooling",
    "ValueOutput",
    "LatentInvariantDiffusionHead",
    "FlowMatchingLossOutput",
    "SinusoidalTimeEmbedding",
    "get_separate_optimizer_param_groups",
]

from axman.core.heads import (
    MorphismPolicyHead,
    PolicyOutput,
    TacticConstructorKind,
    DistanceToClosureValueHead,
    MultiHeadAttentionPooling,
    ValueOutput,
    LatentInvariantDiffusionHead,
    FlowMatchingLossOutput,
    SinusoidalTimeEmbedding,
    get_separate_optimizer_param_groups,
)
from axman.core.kernels import (
    HAS_TRITON,
    K_BLOCK_ARITY,
    fused_port_hypergraph_attention_fast,
    triton_fpha_edge_attention,
)
from axman.core.memory import (
    BatchedHypergraph,
    ContinuousGraphBatcher,
    PinnedHostMemoryPool,
    DeviceRingBuffer,
    GraphMemoryPool,
)
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

__all__ += [
    "HAS_TRITON",
    "K_BLOCK_ARITY",
    "fused_port_hypergraph_attention_fast",
    "triton_fpha_edge_attention",
    "BatchedHypergraph",
    "ContinuousGraphBatcher",
    "PinnedHostMemoryPool",
    "DeviceRingBuffer",
    "GraphMemoryPool",
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

from axman.core.data import (
    HypergraphSample,
    BatchedTargets,
    TopologicalAugmentor,
    HypergraphDataset,
    collate_hypergraph_samples,
    create_hypergraph_dataloader,
    save_hypergraphs_to_binary_dump,
    save_hypergraphs_to_arrow,
    BinaryDumpReader,
    ArrowDatasetReader,
)

__all__ += [
    "HypergraphSample",
    "BatchedTargets",
    "TopologicalAugmentor",
    "HypergraphDataset",
    "collate_hypergraph_samples",
    "create_hypergraph_dataloader",
    "save_hypergraphs_to_binary_dump",
    "save_hypergraphs_to_arrow",
    "BinaryDumpReader",
    "ArrowDatasetReader",
]

from axman.core.training import (
    LossWeights,
    GradNormLossBalancer,
    PCGrad,
    AXMANetModel,
    ModelOutput,
    TaskLosses,
    TrainingConfig,
    EpochMetrics,
    AXMANetTrainer,
)

__all__ += [
    "LossWeights",
    "GradNormLossBalancer",
    "PCGrad",
    "AXMANetModel",
    "ModelOutput",
    "TaskLosses",
    "TrainingConfig",
    "EpochMetrics",
    "AXMANetTrainer",
]



