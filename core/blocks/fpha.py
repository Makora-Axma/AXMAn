"""
Fused Port-Hypergraph Flash Attention (FPHA) for AXMANet.
Part of AXMAN (AXMANet) Phase 1.2 Implementation.

This module computes higher-order multi-head hypergraph attention between
target nodes v and hyperedges e = (u_1, ..., u_k) -> v with positional ports p:

    Attn(v, e) = Softmax( (Q v) * (K [u || Emb(p)])^T / sqrt(d_head) ) * V [u || Emb(p)]

Implements vectorized segmented Softmax and scatter aggregation in pure PyTorch.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from axman.core.encoders.type_embedding import DeBruijnPortEmbedding


class FusedPortHypergraphAttention(nn.Module):
    """
    Fused Port-Hypergraph Flash Attention (FPHA) layer.

    Aggregates multi-source premise sets into hyperedges according to their
    de Bruijn positional ports and binding depths, and routes morphism messages
    to the target nodes.

    Architecture:
    1. Query projection from target nodes: Q(v) in R^{d_model}
    2. Key & Value projections from port-enriched source premises:
       Premise_rep = W_fuse([u || Emb_port(p)])
       K(premise), V(premise) in R^{d_model}
    3. Multi-Head Scaled Dot-Product Attention segmented per hyperedge:
       scores(j, i) = Q(v_j) . K(u_{j, i}) / sqrt(d_k)
       alpha(j, i)  = Softmax_i(scores(j, i))
       m_{e_j}      = sum_{i=1}^{k_j} alpha(j, i) * V(u_{j, i})
    4. Target node message aggregation:
       m_v = sum_{e_j: target(e_j) = v} m_{e_j}
    5. Output projection and gating.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        port_dim: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        port_encoder: Optional[DeBruijnPortEmbedding] = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.port_dim = port_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Positional Port Encoder
        if port_encoder is not None:
            self.port_encoder = port_encoder
        else:
            self.port_encoder = DeBruijnPortEmbedding(
                port_dim=port_dim,
                max_ports=64,
                mode="hybrid",
                dropout=dropout,
            )

        # Source premise fusion: [d_model + port_dim] -> d_model
        self.premise_fusion = nn.Linear(d_model + port_dim, d_model, bias=bias)

        # Multi-Head Linear Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Hyperedge morphism projection (maps aggregated message to hyperedge embedding)
        self.edge_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.proj_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.premise_fusion.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        if self.premise_fusion.bias is not None:
            nn.init.zeros_(self.premise_fusion.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        if self.edge_proj.bias is not None:
            nn.init.zeros_(self.edge_proj.bias)

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
        Forward computation for Fused Port-Hypergraph Attention.

        Args:
            node_embeddings: [num_nodes, d_model] continuous representations of nodes.
            edge_targets: [num_hyperedges] indices of target nodes in [0, num_nodes - 1].
            edge_offsets: [num_hyperedges + 1] CSR offsets into edge_sources.
            edge_sources: [total_sources] flat indices of source premise nodes.
            edge_ports: [total_sources] positional port indices in [0, k_max - 1].
            return_weights: Whether to return premise attention weights [total_sources, num_heads].

        Returns:
            Tuple of:
            - node_updates: [num_nodes, d_model] aggregated messages for each node.
            - edge_embeddings: [num_hyperedges, d_model] aggregated representation of each hyperedge.
            - attention_weights: Optional [total_sources, num_heads] attention weights.
        """
        num_nodes = node_embeddings.shape[0]
        num_edges = edge_targets.shape[0]
        device = node_embeddings.device
        dtype = node_embeddings.dtype

        # Fast path for empty graph / empty edges
        if num_edges == 0 or edge_sources.numel() == 0:
            zero_nodes = torch.zeros(num_nodes, self.d_model, device=device, dtype=dtype)
            zero_edges = torch.zeros(num_edges, self.d_model, device=device, dtype=dtype)
            return zero_nodes, zero_edges, None

        total_sources = edge_sources.numel()

        # 1. Compute arities and edge_id for each source premise
        edge_counts = edge_offsets[1:] - edge_offsets[:-1]  # [num_edges]
        edge_ids = torch.repeat_interleave(
            torch.arange(num_edges, device=device, dtype=torch.long),
            edge_counts.long(),
        )  # [total_sources]

        # 2. Enrich source premises with positional port embeddings
        src_nodes = node_embeddings[edge_sources.long()]  # [total_sources, d_model]
        port_emb = self.port_encoder(edge_ports)          # [total_sources, port_dim]
        premise_cat = torch.cat([src_nodes, port_emb], dim=-1)  # [total_sources, d_model + port_dim]
        premise_rep = self.premise_fusion(premise_cat)          # [total_sources, d_model]

        # 3. Compute Multi-Head Q, K, V
        # Targets for each premise: target_node = edge_targets[edge_id]
        premise_targets = edge_targets[edge_ids].long()  # [total_sources]
        target_node_reps = node_embeddings[premise_targets]  # [total_sources, d_model]

        q = self.q_proj(target_node_reps).view(total_sources, self.num_heads, self.head_dim)
        k = self.k_proj(premise_rep).view(total_sources, self.num_heads, self.head_dim)
        v = self.v_proj(premise_rep).view(total_sources, self.num_heads, self.head_dim)

        # 4. Scaled Dot-Product per premise: [total_sources, num_heads]
        scores = (q * k).sum(dim=-1) * self.scale

        # 5. Numerically Stable Segmented Softmax over each hyperedge's premises
        # Find maximum score per hyperedge for stabilization
        max_scores = torch.full((num_edges, self.num_heads), -1e9, device=device, dtype=dtype)
        max_scores = max_scores.scatter_reduce(
            0,
            edge_ids.unsqueeze(-1).expand(-1, self.num_heads),
            scores,
            reduce="amax",
            include_self=True,
        )

        scores_centered = scores - max_scores[edge_ids]
        exp_scores = torch.exp(scores_centered)

        # Denominator: sum of exps per hyperedge
        sum_exp = torch.zeros(num_edges, self.num_heads, device=device, dtype=dtype)
        sum_exp = sum_exp.scatter_reduce(
            0,
            edge_ids.unsqueeze(-1).expand(-1, self.num_heads),
            exp_scores,
            reduce="sum",
            include_self=False,
        )

        attn_weights = exp_scores / (sum_exp[edge_ids] + 1e-9)
        attn_weights = self.attn_dropout(attn_weights)  # [total_sources, num_heads]

        # 6. Weight values and aggregate into hyperedge messages
        # [total_sources, num_heads, head_dim]
        weighted_v = attn_weights.unsqueeze(-1) * v

        # Pool premises per hyperedge: sum over premises
        edge_messages = torch.zeros(num_edges, self.num_heads, self.head_dim, device=device, dtype=dtype)
        edge_messages = edge_messages.scatter_reduce(
            0,
            edge_ids.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim),
            weighted_v,
            reduce="sum",
            include_self=False,
        )

        # Flatten multi-head dimension: [num_edges, d_model]
        edge_messages_flat = edge_messages.view(num_edges, self.d_model)
        edge_embeddings = self.edge_proj(edge_messages_flat)

        # 7. Aggregate hyperedge messages into target nodes
        # Each hyperedge targets edge_targets[j]
        node_aggregated = torch.zeros(num_nodes, self.d_model, device=device, dtype=dtype)
        node_aggregated = node_aggregated.scatter_reduce(
            0,
            edge_targets.long().unsqueeze(-1).expand(-1, self.d_model),
            edge_messages_flat,
            reduce="sum",
            include_self=False,
        )

        node_updates = self.proj_dropout(self.out_proj(node_aggregated))

        weights_to_return = attn_weights if return_weights else None
        return node_updates, edge_embeddings, weights_to_return
