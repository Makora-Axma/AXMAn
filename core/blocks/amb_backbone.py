"""
AXMA Morphism Block (AMB) and Multi-Layer AXMANetBackbone.
Part of AXMAN (AXMANet) Phase 1.5 Implementation.

Combines:
1. Pre-RMSNorm
2. Fused Port-Hypergraph Flash Attention (FPHA)
3. Categorical Morphism Projection (CMP)
4. SwiGLU Hyper-FeedForward Network
with LayerScale and Residual connections into a deep hierarchical feature extractor.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from axman.core.config import AxmanConfig
from axman.core.encoders.type_embedding import NodeAndPortEncoder, RMSNorm
from axman.core.blocks.fpha import FusedPortHypergraphAttention
from axman.core.blocks.cmp import CategoricalMorphismProjection
from axman.core.blocks.ffn import HyperFeedForwardBlock


@dataclass
class BackboneOutput:
    """
    Container for AXMANetBackbone outputs.
    """
    node_embeddings: torch.Tensor             # [num_nodes, d_model]
    edge_embeddings: torch.Tensor             # [num_hyperedges, d_model]
    all_hidden_states: Optional[List[torch.Tensor]] = None
    all_attentions: Optional[List[torch.Tensor]] = None


class AxmaMorphismBlock(nn.Module):
    """
    Individual AXMA Morphism Block (AMB).

    Internal layer pipeline:
    1. Pre-RMSNorm -> FPHA Attention -> LayerScale / Residual
    2. Pre-RMSNorm -> Categorical Morphism Projection (CMP) -> LayerScale / Residual
    3. Pre-RMSNorm -> SwiGLU Feed-Forward -> LayerScale / Residual
    """

    def __init__(self, config: AxmanConfig):
        super().__init__()
        self.d_model = config.d_model

        # 1. Attention sub-layer with Pre-Norm
        self.norm_attn = RMSNorm(config.d_model, eps=config.eps)
        self.fpha = FusedPortHypergraphAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            port_dim=config.port_dim,
            dropout=config.dropout,
        )

        # 2. Categorical Morphism Projection sub-layer with Pre-Norm
        self.norm_cmp = RMSNorm(config.d_model, eps=config.eps)
        self.cmp = CategoricalMorphismProjection(
            d_model=config.d_model,
            use_spectral_norm=config.use_spectral_norm,
            dropout=config.dropout,
        )

        # 3. SwiGLU Hyper-FeedForward sub-layer (internally pre-normed)
        self.ffn_block = HyperFeedForwardBlock(
            d_model=config.d_model,
            intermediate_dim=config.intermediate_dim,
            dropout=config.dropout,
            eps=config.eps,
        )

        # LayerScale parameters for deep residual stabilization
        if config.layer_scale_init is not None and config.layer_scale_init > 0.0:
            self.ls_attn = nn.Parameter(torch.full((config.d_model,), config.layer_scale_init))
            self.ls_cmp = nn.Parameter(torch.full((config.d_model,), config.layer_scale_init))
        else:
            self.ls_attn = None
            self.ls_cmp = None

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_targets: torch.Tensor,
        edge_offsets: torch.Tensor,
        edge_sources: torch.Tensor,
        edge_ports: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward computation of AMB block.

        Args:
            node_embeddings: [num_nodes, d_model]
            edge_targets: [num_hyperedges]
            edge_offsets: [num_hyperedges + 1]
            edge_sources: [total_sources]
            edge_ports: [total_sources]
            return_weights: whether to return attention weights

        Returns:
            Tuple of:
            - updated_node_embeddings: [num_nodes, d_model]
            - edge_embeddings: [num_hyperedges, d_model]
            - attn_weights: Optional [total_sources, num_heads]
        """
        # Step 1: Pre-RMSNorm and FPHA Port-Hypergraph Attention
        normed_nodes = self.norm_attn(node_embeddings)
        node_updates, edge_embeddings, attn_weights = self.fpha(
            node_embeddings=normed_nodes,
            edge_targets=edge_targets,
            edge_offsets=edge_offsets,
            edge_sources=edge_sources,
            edge_ports=edge_ports,
            return_weights=return_weights,
        )

        # Apply LayerScale or direct residual
        if self.ls_attn is not None:
            node_updates = self.ls_attn * node_updates
        h1 = node_embeddings + self.dropout(node_updates)

        # Step 2: Categorical Morphism Projection (CMP)
        normed_h1 = self.norm_cmp(h1)
        cmp_out = self.cmp(normed_h1)
        # Identity residual: diff from pre-normed representation
        cmp_delta = cmp_out - normed_h1
        if self.ls_cmp is not None:
            cmp_delta = self.ls_cmp * cmp_delta
        h2 = h1 + self.dropout(cmp_delta)

        # Step 3: SwiGLU Feed-Forward Network
        h3 = self.ffn_block(h2)

        return h3, edge_embeddings, attn_weights


class AXMANetBackbone(nn.Module):
    """
    Hierarchical multi-layer AXMANet feature extractor.

    Accepts raw port-hypergraphs with AST primitive kinds and continuous node features,
    encodes type and port primitives, and processes the graph through L sequential
    AXMA Morphism Blocks (AMB).

    Returns deeply enriched representations for both nodes and morphisms.
    """

    def __init__(self, config: Optional[AxmanConfig] = None):
        super().__init__()
        self.config = config if config is not None else AxmanConfig()

        # 1. Type and Port Embedding Encoder
        self.encoder = NodeAndPortEncoder(
            d_model=self.config.d_model,
            port_dim=self.config.port_dim,
            num_kinds=self.config.num_kinds,
            in_features=self.config.in_features,
            max_ports=self.config.max_ports,
            port_mode=self.config.port_mode,
            dropout=self.config.dropout,
        )

        # 2. Sequential AMB Layers
        self.layers = nn.ModuleList([
            AxmaMorphismBlock(self.config) for _ in range(self.config.num_layers)
        ])

        # 3. Final Pre-Head Normalization
        self.final_norm = RMSNorm(self.config.d_model, eps=self.config.eps)

    def forward(
        self,
        edge_targets: torch.Tensor,
        edge_offsets: torch.Tensor,
        edge_sources: torch.Tensor,
        edge_ports: torch.Tensor,
        kind_ids: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> BackboneOutput:
        """
        Forward computation of AXMANetBackbone.

        Args:
            edge_targets: [num_hyperedges] indices of target nodes.
            edge_offsets: [num_hyperedges + 1] CSR offsets into edge_sources.
            edge_sources: [total_sources] flat indices of source premise nodes.
            edge_ports: [total_sources] positional port indices.
            kind_ids: Optional [num_nodes] categorical AST node kinds.
            node_features: Optional [num_nodes, in_features] continuous feature matrix.
            output_hidden_states: whether to collect layer representations.
            output_attentions: whether to collect attention weights.

        Returns:
            BackboneOutput with node_embeddings [num_nodes, d_model] and
            edge_embeddings [num_hyperedges, d_model].
        """
        # 1. Initial Type & Feature Encoding
        h_nodes, _ = self.encoder(
            kind_ids=kind_ids,
            node_features=node_features,
            edge_ports=None,  # ports are encoded dynamically inside FPHA
        )

        all_hidden_states = [h_nodes] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        last_edge_embeddings = None

        # 2. Sequential pass through AMB layers
        for layer in self.layers:
            h_nodes, edge_emb, attn_w = layer(
                node_embeddings=h_nodes,
                edge_targets=edge_targets,
                edge_offsets=edge_offsets,
                edge_sources=edge_sources,
                edge_ports=edge_ports,
                return_weights=output_attentions,
            )
            last_edge_embeddings = edge_emb

            if output_hidden_states:
                all_hidden_states.append(h_nodes)
            if output_attentions:
                all_attentions.append(attn_w)

        # 3. Final Root Normalization
        h_nodes = self.final_norm(h_nodes)

        # Fallback if graph had no edges
        if last_edge_embeddings is None:
            num_edges = edge_targets.shape[0]
            last_edge_embeddings = torch.zeros(
                num_edges, self.config.d_model, device=h_nodes.device, dtype=h_nodes.dtype
            )

        return BackboneOutput(
            node_embeddings=h_nodes,
            edge_embeddings=last_edge_embeddings,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
        )
