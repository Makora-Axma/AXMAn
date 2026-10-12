"""
Type Theory Primitive Encoders and de Bruijn Port Embeddings for AXMANet.
Part of AXMAN (AXMANet) Phase 1.1 Implementation.

This module provides continuous tensor representations for:
1. Type theory primitives (Sorts, BVars, FVars, MVars, Apps, Lambdas, Pis, etc.)
2. Positional ports and de Bruijn binder scopes for hyperedges (k <= 16).
"""

import math
from enum import IntEnum
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypePrimitiveKind(IntEnum):
    """
    Categorical node kinds representing Calculus of Inductive Constructions (CIC)
    and Lean 4 abstract syntax tree primitives.
    """
    SORT = 0          # Type universe / Prop (Sort u)
    BVAR = 1          # de Bruijn bound variable
    FVAR = 2          # Free variable / local hypothesis
    MVAR = 3          # Metavariable / open goal / goal hole
    CONST = 4         # Global constant / definition / theorem
    APP = 5           # Function application
    LAMBDA = 6        # Lambda abstraction (\x: A => t)
    FORALL = 7        # Dependent Pi-type / universal quantifier (\Pi x: A, B)
    LET = 8           # Local let-binding (let x : A := v in b)
    LITERAL = 9       # Literal value (Nat, String, etc.)
    MDATA = 10        # Metadata / macro annotation
    PROJ = 11         # Structure field projection
    AXIOM = 12        # Foundational axiom
    INDUCTIVE = 13    # Inductive datatype declaration
    CONSTRUCTOR = 14  # Inductive data constructor
    RECURSOR = 15     # Eliminator / recursor (e.g. Nat.rec)
    TYPE_CLASS = 16   # Typeclass instance or class predicate
    CUSTOM = 17       # User-defined / domain-specific term


class TypePrimitiveEmbedding(nn.Module):
    """
    Embeds discrete categorical type kinds and continuous node feature vectors
    into a unified continuous latent space of dimension d_model.

    Supports:
    - Trainable embedding matrix for AST primitive kinds.
    - Projection of continuous input features (e.g. from C-ABI node_features).
    - Optional normalization (RMSNorm / LayerNorm) and residual fusion.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_kinds: int = 32,
        in_features: Optional[int] = None,
        dropout: float = 0.0,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_kinds = num_kinds
        self.in_features = in_features
        self.use_rmsnorm = use_rmsnorm

        # Discrete kind embedding matrix
        self.kind_embedding = nn.Embedding(
            num_embeddings=num_kinds,
            embedding_dim=d_model,
        )
        nn.init.normal_(self.kind_embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        # Continuous feature projection (if continuous features are provided)
        if in_features is not None and in_features > 0:
            self.feature_proj = nn.Linear(in_features, d_model, bias=True)
            nn.init.xavier_uniform_(self.feature_proj.weight)
            nn.init.zeros_(self.feature_proj.bias)
        else:
            self.feature_proj = None

        # Normalization layer
        if use_rmsnorm:
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        kind_ids: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute continuous node embeddings.

        Args:
            kind_ids: Tensor of categorical node kinds [num_nodes], dtype int64.
            node_features: Continuous feature matrix [num_nodes, in_features], dtype float32.

        Returns:
            Continuous node representations [num_nodes, d_model].
        """
        if kind_ids is None and node_features is None:
            raise ValueError("Either kind_ids or node_features must be provided to TypePrimitiveEmbedding.")

        h = None

        if kind_ids is not None:
            # Clamp kind IDs to valid range to guard against corrupted inputs
            clamped_kinds = torch.clamp(kind_ids, min=0, max=self.num_kinds - 1)
            h_kind = self.kind_embedding(clamped_kinds)
            h = h_kind

        if node_features is not None:
            if self.feature_proj is None:
                # If continuous features provided without preconfigured in_features,
                # dynamically project or verify dimension
                if node_features.shape[-1] == self.d_model:
                    h_feat = node_features
                else:
                    raise ValueError(
                        f"node_features feature dimension {node_features.shape[-1]} "
                        f"does not match d_model {self.d_model}, and in_features was not configured."
                    )
            else:
                h_feat = self.feature_proj(node_features)

            if h is None:
                h = h_feat
            else:
                h = h + h_feat

        h = self.norm(h)
        h = self.dropout(h)
        return h


class DeBruijnPortEmbedding(nn.Module):
    """
    Positional and binding-depth port encoder for hyperedges.

    In AXMAN Port-Hypergraphs G = (V, E, P), each hyperedge e = (u_1, ..., u_k) -> v
    indexes source premises through positional ports p in {0, ..., k-1}.
    Ports convey:
    - Argument position in function signatures (non-commutative parameter order).
    - de Bruijn binder depth / quantifier nesting level.

    Supports:
    - 'learned': Trainable embedding table.
    - 'sinusoidal': Deterministic harmonic encoding invariant to sequence scaling.
    - 'hybrid': Deterministic sinusoidal basis + learned linear projection & embedding.
    """

    def __init__(
        self,
        port_dim: int = 64,
        max_ports: int = 64,
        mode: str = "hybrid",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.port_dim = port_dim
        self.max_ports = max_ports
        self.mode = mode.lower()

        if self.mode not in ("learned", "sinusoidal", "hybrid"):
            raise ValueError(f"Unknown port embedding mode: {mode}. Must be 'learned', 'sinusoidal', or 'hybrid'.")

        if self.mode in ("learned", "hybrid"):
            self.learned_embed = nn.Embedding(max_ports, port_dim)
            nn.init.normal_(self.learned_embed.weight, mean=0.0, std=1.0 / math.sqrt(port_dim))
        else:
            self.learned_embed = None

        if self.mode in ("sinusoidal", "hybrid"):
            # Sinusoidal projection layer to align frequencies
            self.sin_proj = nn.Linear(port_dim, port_dim, bias=False)
            nn.init.orthogonal_(self.sin_proj.weight)
        else:
            self.sin_proj = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _generate_sinusoidal(self, ports: torch.Tensor) -> torch.Tensor:
        """
        Generate harmonic positional embeddings for arbitrary integer port indices.
        PE(p, 2i)   = sin(p / 10000^(2i / d))
        PE(p, 2i+1) = cos(p / 10000^(2i / d))
        """
        ports_flat = ports.float().unsqueeze(-1)  # [N, 1]
        half_dim = self.port_dim // 2
        denom = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=ports.device, dtype=torch.float32) * -denom)
        args = ports_flat * freqs.unsqueeze(0)  # [N, half_dim]

        sin_part = torch.sin(args)
        cos_part = torch.cos(args)

        if self.port_dim % 2 == 1:
            # Handle odd port_dim by appending extra zero column
            zeros = torch.zeros_like(ports_flat)
            pe = torch.cat([sin_part, cos_part, zeros], dim=-1)
        else:
            pe = torch.cat([sin_part, cos_part], dim=-1)

        return pe

    def forward(self, ports: torch.Tensor) -> torch.Tensor:
        """
        Encode positional port indices into continuous port vectors.

        Args:
            ports: 1D Tensor of port indices [num_edge_sources], dtype int32 or int64.

        Returns:
            Continuous port representations [num_edge_sources, port_dim].
        """
        if ports.dim() == 0:
            ports = ports.unsqueeze(0)

        device = ports.device

        if self.mode == "learned":
            clamped = torch.clamp(ports.long(), min=0, max=self.max_ports - 1)
            emb = self.learned_embed(clamped)
        elif self.mode == "sinusoidal":
            pe = self._generate_sinusoidal(ports)
            emb = self.sin_proj(pe) if self.sin_proj is not None else pe
        elif self.mode == "hybrid":
            clamped = torch.clamp(ports.long(), min=0, max=self.max_ports - 1)
            learned_part = self.learned_embed(clamped)
            sin_part = self._generate_sinusoidal(ports)
            emb = learned_part + self.sin_proj(sin_part)

        return self.dropout(emb)


class NodeAndPortEncoder(nn.Module):
    """
    Combined feature encoder for AXMAN Port-Hypergraphs.

    Processes both:
    1. Node attributes & AST primitive types (TypePrimitiveEmbedding)
    2. Edge port positions & de Bruijn binders (DeBruijnPortEmbedding)

    Provides utilities to enrich premise source nodes with their respective port
    representations, preparing the [node_feature || port_embedding] representations
    for downstream attention layers (FPHA, CMP, AMB).
    """

    def __init__(
        self,
        d_model: int = 256,
        port_dim: int = 64,
        num_kinds: int = 32,
        in_features: Optional[int] = None,
        max_ports: int = 64,
        port_mode: str = "hybrid",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.port_dim = port_dim

        self.node_encoder = TypePrimitiveEmbedding(
            d_model=d_model,
            num_kinds=num_kinds,
            in_features=in_features,
            dropout=dropout,
        )

        self.port_encoder = DeBruijnPortEmbedding(
            port_dim=port_dim,
            max_ports=max_ports,
            mode=port_mode,
            dropout=dropout,
        )

        # Port-to-model projection for residual addition or combination
        if port_dim != d_model:
            self.port_to_model = nn.Linear(port_dim, d_model, bias=False)
            nn.init.xavier_uniform_(self.port_to_model.weight)
        else:
            self.port_to_model = nn.Identity()

        # Fused projection for concatenation mode [d_model + port_dim] -> d_model
        self.fuse_proj = nn.Linear(d_model + port_dim, d_model, bias=True)
        nn.init.xavier_uniform_(self.fuse_proj.weight)
        nn.init.zeros_(self.fuse_proj.bias)

    def forward(
        self,
        kind_ids: Optional[torch.Tensor] = None,
        node_features: Optional[torch.Tensor] = None,
        edge_ports: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode node features and port indices.

        Args:
            kind_ids: [num_nodes] categorical kind IDs.
            node_features: [num_nodes, in_features] continuous node features.
            edge_ports: Optional [total_sources] port indices.

        Returns:
            Tuple of:
            - node_embeddings: [num_nodes, d_model]
            - port_embeddings: [total_sources, port_dim] (or None if edge_ports is None)
        """
        node_emb = self.node_encoder(kind_ids=kind_ids, node_features=node_features)

        port_emb = None
        if edge_ports is not None:
            port_emb = self.port_encoder(edge_ports)

        return node_emb, port_emb

    def enrich_sources(
        self,
        node_embeddings: torch.Tensor,
        edge_sources: torch.Tensor,
        edge_ports: torch.Tensor,
        mode: str = "concat_proj",
    ) -> torch.Tensor:
        """
        Combine source node embeddings with their corresponding port embeddings.

        Args:
            node_embeddings: [num_nodes, d_model] node representation table.
            edge_sources: [total_sources] index tensor pointing into node_embeddings.
            edge_ports: [total_sources] positional port index tensor.
            mode: 'concat_proj' (concatenate and linear project to d_model),
                  'add' (project port to d_model and add to source node),
                  'concat' (raw concatenation [total_sources, d_model + port_dim]).

        Returns:
            Enriched source premise representations [total_sources, d_out].
        """
        # Gather source node features for each premise
        src_nodes = node_embeddings[edge_sources.long()]  # [total_sources, d_model]
        port_emb = self.port_encoder(edge_ports)          # [total_sources, port_dim]

        if mode == "concat_proj":
            concat = torch.cat([src_nodes, port_emb], dim=-1)
            return self.fuse_proj(concat)
        elif mode == "add":
            port_proj = self.port_to_model(port_emb)
            return src_nodes + port_proj
        elif mode == "concat":
            return torch.cat([src_nodes, port_emb], dim=-1)
        else:
            raise ValueError(f"Unsupported enrichment mode: {mode}. Choose 'concat_proj', 'add', or 'concat'.")


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Provides computationally efficient normalization without mean-centering overhead.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight
