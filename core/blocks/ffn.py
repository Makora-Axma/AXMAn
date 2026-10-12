"""
SwiGLU Hyper-FeedForward Network and Pre-RMSNorm for AXMANet.
Part of AXMAN (AXMANet) Phase 1.4 Implementation.

Implements the gated non-linear transformation layer:
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
with Pre-RMSNorm and residual connection to mitigate gradient vanishing/explosion
and combat over-smoothing in deep hypergraph stacks.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from axman.core.encoders.type_embedding import RMSNorm


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) Feed-Forward Network.

    Mathematical formulation (Shazeer, 2020):
        SwiGLU(x) = (Swish(x W_gate) * x W_up) W_down
    where Swish(z) = SiLU(z) = z * sigmoid(z).

    Default intermediate dimension follows LLaMA/Mistral convention:
    intermediate_dim = 2 * floor(4 * d_model / 3) rounded to multiple of 256.
    """

    def __init__(
        self,
        d_model: int = 256,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model

        if intermediate_dim is None:
            # Standard 8/3 expansion rounded to multiple of 64 or 256
            raw_dim = int(8 * d_model / 3)
            # Round up to nearest multiple of 64
            self.intermediate_dim = ((raw_dim + 63) // 64) * 64
        else:
            self.intermediate_dim = intermediate_dim

        # Gated linear projections
        self.w_gate = nn.Linear(d_model, self.intermediate_dim, bias=bias)
        self.w_up = nn.Linear(d_model, self.intermediate_dim, bias=bias)
        self.w_down = nn.Linear(self.intermediate_dim, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization with scaling for gated networks
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_up.weight)
        nn.init.xavier_uniform_(self.w_down.weight)
        if self.w_gate.bias is not None:
            nn.init.zeros_(self.w_gate.bias)
        if self.w_up.bias is not None:
            nn.init.zeros_(self.w_up.bias)
        if self.w_down.bias is not None:
            nn.init.zeros_(self.w_down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward computation for SwiGLU.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Transformed tensor [..., d_model]
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        activated = gate * up
        down = self.w_down(activated)
        return self.dropout(down)


class HyperFeedForwardBlock(nn.Module):
    """
    Hyper-FeedForward Block with Pre-RMSNorm and residual addition:
        out = x + Dropout(SwiGLU(RMSNorm(x)))

    Ensures gradient stability in deep hypergraph stacks and prevents
    latent over-smoothing.
    """

    def __init__(
        self,
        d_model: int = 256,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLUFeedForward(
            d_model=d_model,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Pre-RMSNorm and Residual connection.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Output tensor [..., d_model]
        """
        normed = self.norm(x)
        ffn_out = self.ffn(normed)
        return x + self.dropout(ffn_out)
