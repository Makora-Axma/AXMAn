"""
Morphism Policy Head for AXMANet (AXMA-N).
Part of AXMAN (AXMANet) Phase 2.1 Implementation.

Predicts:
1. Tactic Constructor Category Logits (discrete taxonomy of inference actions).
2. Premise Selection Logits via Pointer Network / Bilinear Scoring against candidate nodes.
3. Supports Action Masking for valid moves and premise filtering.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from axman.core.config import AxmanConfig
from axman.core.encoders.type_embedding import RMSNorm


class TacticConstructorKind(IntEnum):
    """
    Taxonomy of discrete tactical inference actions in formal verification (CIC / Lean 4).
    """
    INTRO = 0          # Introduce hypothesis / binder into context
    EXACT = 1          # Exact match / proof closure by term
    APPLY = 2          # Apply theorem or function backward
    REWRITE = 3        # Equality substitution / rewrite rule
    SIMP = 4           # Simplification / confluence reduction
    CASES = 5          # Case analysis on inductive type
    INDUCTION = 6      # Structural or well-founded induction
    CONTRADICTION = 7  # Ex falso quodlibet / contradiction
    SPLIT = 8          # Split conjunction / constructor decomposition
    RING = 9           # Ring/field algebraic solver
    LINARITH = 10      # Linear integer/real arithmetic
    NORM_NUM = 11      # Numerical normalizer
    ASSUMPTION = 12    # Look up matching hypothesis in local context
    CONSTRUCTOR = 13   # Apply inductive constructor directly
    REFLEXIVITY = 14   # Definitional equality / refl
    TRIVIAL = 15       # Trivial solver
    CUSTOM = 16        # User-defined tactic or composite rule


@dataclass
class PolicyOutput:
    """
    Container for Morphism Policy Head outputs.
    """
    action_logits: torch.Tensor          # [num_goals, num_actions]
    premise_logits: torch.Tensor         # [num_goals, num_candidates]
    selected_premise_indices: Optional[torch.Tensor] = None


class MorphismPolicyHead(nn.Module):
    """
    Two-stream Morphism Policy Head:
    1. Tactic Action Classifier: MLP predicting logits over discrete tactic actions.
    2. Premise Pointing Network: Scaled dot-product / bilinear attention mechanism
       scoring candidate premises (local hypotheses and library lemmas) for the selected goal.

    Features:
    - Pre-RMSNorm on goal and premise embeddings.
    - Configurable bilinear or scaled dot-product projection for premise scoring.
    - Support for action masking (ignoring invalid tactics).
    - Support for candidate premise masking (ignoring invalid/inaccessible hypotheses).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_actions: int = 32,
        action_hidden_dim: Optional[int] = None,
        scoring_mode: str = "bilinear",
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.scoring_mode = scoring_mode.lower()

        if self.scoring_mode not in ("bilinear", "dot_product"):
            raise ValueError(
                f"Unknown scoring_mode '{scoring_mode}'. Must be 'bilinear' or 'dot_product'."
            )

        # Pre-normalization layers
        self.norm_goal = RMSNorm(d_model, eps=eps)
        self.norm_premise = RMSNorm(d_model, eps=eps)

        # Stream 1: Tactic action classifier MLP
        act_hidden = action_hidden_dim if action_hidden_dim is not None else d_model
        self.action_mlp = nn.Sequential(
            nn.Linear(d_model, act_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(act_hidden, num_actions, bias=True),
        )

        # Stream 2: Premise Pointing Network
        # Goal query projection & Candidate key projection
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

        if self.scoring_mode == "bilinear":
            self.bilinear_weight = nn.Parameter(torch.empty(d_model, d_model))
            nn.init.orthogonal_(self.bilinear_weight)
        else:
            self.bilinear_weight = None

        self.scale = 1.0 / (d_model ** 0.5)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @classmethod
    def from_config(cls, config: AxmanConfig, num_actions: int = 32, **kwargs) -> "MorphismPolicyHead":
        """
        Instantiate Policy Head directly from AxmanConfig.
        """
        return cls(
            d_model=config.d_model,
            num_actions=num_actions,
            dropout=config.dropout,
            eps=config.eps,
            **kwargs,
        )

    def forward(
        self,
        goal_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        """
        Forward computation of Morphism Policy Head.

        Args:
            goal_embeddings: [num_goals, d_model] representation of target goal node(s).
                             Can also be [batch_size, num_goals, d_model] or 1D [d_model].
            candidate_embeddings: [num_candidates, d_model] candidate premise representations.
                                  Can also be [batch_size, num_candidates, d_model].
            action_mask: Optional boolean tensor of valid actions. True = valid, False = masked out.
                         Shape: [num_goals, num_actions] or [num_actions].
            candidate_mask: Optional boolean tensor of valid candidates. True = valid, False = masked out.
                            Shape: [num_goals, num_candidates] or [num_candidates].

        Returns:
            PolicyOutput containing:
            - action_logits: [num_goals, num_actions]
            - premise_logits: [num_goals, num_candidates]
        """
        squeeze_goal = False
        if goal_embeddings.dim() == 1:
            goal_embeddings = goal_embeddings.unsqueeze(0)  # [1, d_model]
            squeeze_goal = True

        squeeze_candidates = False
        if candidate_embeddings.dim() == 1:
            candidate_embeddings = candidate_embeddings.unsqueeze(0)  # [1, d_model]
            squeeze_candidates = True

        # Pre-RMSNorm
        norm_goal = self.norm_goal(goal_embeddings)
        norm_candidates = self.norm_premise(candidate_embeddings)

        # 1. Tactic Action Classification Stream
        action_logits = self.action_mlp(norm_goal)  # [num_goals, num_actions]

        if action_mask is not None:
            # Mask out invalid actions with large negative value (-1e9)
            if action_mask.dtype != torch.bool:
                action_mask = action_mask.bool()
            action_logits = action_logits.masked_fill(~action_mask, -1e9)

        # 2. Premise Pointing Stream
        q = self.query_proj(norm_goal)          # [num_goals, d_model]
        k = self.key_proj(norm_candidates)      # [num_candidates, d_model]

        if self.scoring_mode == "bilinear":
            # q @ W @ k^T
            # q: [num_goals, d_model], bilinear_weight: [d_model, d_model]
            q_trans = torch.matmul(q, self.bilinear_weight)  # [num_goals, d_model]
            premise_logits = torch.matmul(q_trans, k.transpose(-1, -2)) * self.scale
        else:
            # Scaled Dot-Product: q @ k^T
            premise_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if candidate_mask is not None:
            if candidate_mask.dtype != torch.bool:
                candidate_mask = candidate_mask.bool()
            if candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.unsqueeze(0)
            premise_logits = premise_logits.masked_fill(~candidate_mask, -1e9)

        if squeeze_goal and not squeeze_candidates:
            # goal was 1D [d_model], return [num_candidates] for premise and [num_actions] for action
            pass

        return PolicyOutput(
            action_logits=action_logits,
            premise_logits=premise_logits,
        )
