"""
Distance-to-Closure Value Head for AXMANet (AXMA-N).
Part of AXMAN (AXMANet) Phase 2.2 Implementation.

Predicts:
1. Scalar probability of proof closure V in [0, 1] (Sigmoid output).
2. Topological distance-to-closure in steps to Q.E.D. (continuous/ReLU or integer step prediction).
3. Employs Multi-Head Attention Pooling across goal/graph nodes with learned global query.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from axman.core.config import AxmanConfig
from axman.core.encoders.type_embedding import RMSNorm


@dataclass
class ValueOutput:
    """
    Container for Distance-to-Closure Value Head outputs.
    """
    value_prob: torch.Tensor          # Probability of proof closure V in [0, 1], shape [batch_size, 1] or [1]
    distance_to_qed: torch.Tensor     # Predicted topological steps to closure, shape [batch_size, 1] or [1]
    graph_latent: torch.Tensor        # Pooled global graph latent state h_graph [batch_size, d_model]
    pooling_weights: Optional[torch.Tensor] = None  # Attention weights over nodes [batch_size, num_heads, 1, num_nodes]


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling over sets of node embeddings.
    Aggregates variable-size node sets into a single graph context vector h_graph.

    Uses a learned global query token:
    Query: Q = q_global (learned parameter) [1, num_heads, 1, head_dim]
    Keys/Values: K, V = Linear_k(nodes), Linear_v(nodes)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.norm = RMSNorm(d_model, eps=eps)

        # Learned global query token
        self.global_query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.global_query, std=0.02)

        # Projections for keys and values
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_out = nn.Linear(d_model, d_model, bias=False)

        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pool nodes into a single vector.

        Args:
            node_embeddings: [num_nodes, d_model] or [batch_size, num_nodes, d_model].
            node_mask: Optional boolean mask where True = keep, False = ignore.
                       Shape [num_nodes] or [batch_size, num_nodes].
            return_weights: Whether to return attention weights.

        Returns:
            Tuple of:
            - pooled: [batch_size, d_model]
            - attn_weights: Optional [batch_size, num_heads, 1, num_nodes]
        """
        is_unbatched = False
        if node_embeddings.dim() == 2:
            is_unbatched = True
            node_embeddings = node_embeddings.unsqueeze(0)  # [1, num_nodes, d_model]

        batch_size, num_nodes, _ = node_embeddings.shape

        normed_nodes = self.norm(node_embeddings)

        # Expand global query across batch
        q = self.global_query.expand(batch_size, -1, -1)  # [batch_size, 1, d_model]
        k = self.proj_k(normed_nodes)                      # [batch_size, num_nodes, d_model]
        v = self.proj_v(normed_nodes)                      # [batch_size, num_nodes, d_model]

        # Reshape for multi-head: [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product scores: [batch_size, num_heads, 1, num_nodes]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if node_mask is not None:
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)
            if node_mask.dim() == 2:
                # Shape [batch_size, 1, 1, num_nodes]
                mask_expanded = node_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask_expanded = node_mask
            scores = scores.masked_fill(~mask_expanded, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: [batch_size, num_heads, 1, head_dim]
        context = torch.matmul(attn_weights, v)
        # Permute back: [batch_size, 1, num_heads * head_dim] -> [batch_size, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, self.d_model)
        pooled = self.proj_out(context)

        if is_unbatched:
            pooled = pooled.squeeze(0)
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(0)

        weights_to_return = attn_weights if return_weights else None
        return pooled, weights_to_return


class DistanceToClosureValueHead(nn.Module):
    """
    Distance-to-Closure Value Head for AXMANet.

    Architecture:
    1. Multi-Head Attention Pooling across all open subgoals / active nodes -> h_graph.
    2. Shared trunk MLP with Pre-RMSNorm and GELU.
    3. Dual Prediction Heads:
       - Value Head: Linear -> Sigmoid -> V in [0, 1] (Probability of proof closure).
       - Distance Head: Linear -> Softplus (or ReLU) -> D >= 0 (Topological steps to Q.E.D.).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        h_dim = hidden_dim if hidden_dim is not None else d_model

        # Global Attention Pooling
        self.pooling = MultiHeadAttentionPooling(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            eps=eps,
        )

        # Trunk Pre-Norm
        self.norm = RMSNorm(d_model, eps=eps)

        # Shared MLP Trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(d_model, h_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(h_dim, h_dim, bias=True),
            nn.GELU(),
        )

        # Head 1: Value probability V in [0, 1]
        self.value_linear = nn.Linear(h_dim, 1, bias=True)

        # Head 2: Topological distance to Q.E.D. in steps (non-negative)
        self.distance_linear = nn.Linear(h_dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.shared_trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.value_linear.weight)
        nn.init.zeros_(self.value_linear.bias)
        nn.init.xavier_uniform_(self.distance_linear.weight)
        # Initialize distance bias to a reasonable prior (e.g. 5 steps to Q.E.D.)
        nn.init.constant_(self.distance_linear.bias, 5.0)

    @classmethod
    def from_config(cls, config: AxmanConfig, **kwargs) -> "DistanceToClosureValueHead":
        """
        Instantiate Value Head from AxmanConfig.
        """
        return cls(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            eps=config.eps,
            **kwargs,
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        return_pooling_weights: bool = False,
    ) -> ValueOutput:
        """
        Forward computation of Distance-to-Closure Value Head.

        Args:
            node_embeddings: [num_nodes, d_model] or [batch_size, num_nodes, d_model].
            node_mask: Optional boolean mask where True = keep, False = ignore.
            return_pooling_weights: Whether to return pooling attention weights.

        Returns:
            ValueOutput containing:
            - value_prob: probability of closure in [0, 1].
            - distance_to_qed: estimated remaining proof steps >= 0.
            - graph_latent: pooled representation h_graph [d_model] or [batch_size, d_model].
            - pooling_weights: optional attention weights.
        """
        # Step 1: Multi-Head Attention Pooling
        h_graph, pooling_weights = self.pooling(
            node_embeddings=node_embeddings,
            node_mask=node_mask,
            return_weights=return_pooling_weights,
        )

        # Step 2: Shared Trunk Representation
        normed_h = self.norm(h_graph)
        trunk_rep = self.shared_trunk(normed_h)

        # Step 3: Dual Predictions
        value_logits = self.value_linear(trunk_rep)
        value_prob = torch.sigmoid(value_logits)

        dist_raw = self.distance_linear(trunk_rep)
        # Softplus ensures smooth, strictly non-negative step predictions
        distance_to_qed = F.softplus(dist_raw)

        return ValueOutput(
            value_prob=value_prob,
            distance_to_qed=distance_to_qed,
            graph_latent=h_graph,
            pooling_weights=pooling_weights,
        )
