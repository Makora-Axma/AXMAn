"""
Latent Invariant Flow Matching Diffusion Head for AXMANet (AXMA-N).
Part of AXMAN (AXMANet) Phase 2.3 Implementation.

Implements:
1. Continuous Normalizing Flows / Flow Matching (Optimal Transport CFM) for lemma abduction.
2. Linear probability path: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1 (or standard (1 - t) x_0 + t x_1).
   Target vector field: u_t(x_t | x_1, x_0) = x_1 - (1 - sigma_min) * x_0.
3. Conditioned vector field regressor v_theta(x_t, t, context) using sinusoidal time embeddings,
   adaptive RMSNorm / FiLM modulation, and ResNet/MLP blocks.
4. ODE solvers: Euler, Midpoint (RK2), and Runge-Kutta 4 (RK4) for fast 5-10 step sampling.
5. Utilities for training with separate parameter groups and GradNorm/loss balancing.
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from axman.core.config import AxmanConfig
from axman.core.encoders.type_embedding import RMSNorm


@dataclass
class FlowMatchingLossOutput:
    """
    Container for Flow Matching training loss and diagnostics.
    """
    loss: torch.Tensor             # Mean squared error between predicted and target vector field
    v_pred: torch.Tensor           # Predicted velocity vector field
    v_target: torch.Tensor         # Target velocity vector field (x_1 - x_0)
    x_t: torch.Tensor              # Interpolated state at time t
    t: torch.Tensor                # Sampled time step in [0, 1]


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal harmonic time embedding for continuous time t in [0, 1].
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        denom = math.log(10000.0) / max(self.half_dim - 1, 1)
        self.register_buffer("freqs", torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * -denom))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embed continuous time t into [batch_size, dim].

        Args:
            t: 1D tensor [batch_size] with values in [0, 1].

        Returns:
            Embeddings of shape [batch_size, dim].
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_flat = t.float().unsqueeze(-1)  # [batch_size, 1]
        args = t_flat * self.freqs.unsqueeze(0)  # [batch_size, half_dim]
        sin_part = torch.sin(args)
        cos_part = torch.cos(args)

        if self.dim % 2 == 1:
            zeros = torch.zeros_like(t_flat)
            pe = torch.cat([sin_part, cos_part, zeros], dim=-1)
        else:
            pe = torch.cat([sin_part, cos_part], dim=-1)
        return pe


class FlowVectorFieldBlock(nn.Module):
    """
    Residual block with AdaRMSNorm (Adaptive RMSNorm with FiLM scale and shift)
    conditioned on time and graph context embeddings.
    """

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        self.cond_proj = nn.Linear(cond_dim, dim * 2, bias=True)
        self.linear1 = nn.Linear(dim, dim, bias=True)
        self.linear2 = nn.Linear(dim, dim, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Zero initialization for residual scaling
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Current latent state [batch_size, dim].
            cond: Combined conditioning vector (time + context) [batch_size, cond_dim].

        Returns:
            Residual output [batch_size, dim].
        """
        h = self.norm(x)
        # FiLM modulation: scale (gamma) and shift (beta)
        film = self.cond_proj(cond)
        gamma, beta = torch.chunk(film, 2, dim=-1)
        h = h * (1.0 + gamma) + beta

        h = self.act(h)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h


class LatentInvariantDiffusionHead(nn.Module):
    """
    Flow Matching Generative Head for synthesizing latent lemma invariants.

    Supports:
    - Optimal Transport Conditional Flow Matching (OT-CFM):
      x_t = (1 - t) * x_0 + t * x_1, where x_0 ~ N(0, I), x_1 is target invariant.
      Target velocity: v_t = x_1 - x_0.
    - Continuous time parameterization t in [0, 1].
    - Conditioning on graph context vector h_graph (from backbone or value pooling).
    - Fast ODE integration solvers for inference: Euler, Midpoint, RK4.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        context_dim: int = 256,
        time_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.0,
        sigma_min: float = 1e-4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.time_dim = time_dim
        self.sigma_min = sigma_min

        # Time encoder
        self.time_embed = SinusoidalTimeEmbedding(dim=time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Context adapter
        self.context_norm = RMSNorm(context_dim, eps=eps)
        cond_dim = time_dim + context_dim

        # Input projection for state x_t
        self.in_proj = nn.Linear(latent_dim, latent_dim)

        # Flow ResNet blocks
        self.blocks = nn.ModuleList([
            FlowVectorFieldBlock(
                dim=latent_dim,
                cond_dim=cond_dim,
                dropout=dropout,
                eps=eps,
            )
            for _ in range(num_blocks)
        ])

        # Final output projection for predicted vector field v_theta
        self.final_norm = RMSNorm(latent_dim, eps=eps)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @classmethod
    def from_config(cls, config: AxmanConfig, **kwargs) -> "LatentInvariantDiffusionHead":
        """
        Instantiate Latent Invariant Diffusion Head from AxmanConfig.
        """
        return cls(
            latent_dim=config.d_model,
            context_dim=config.d_model,
            dropout=config.dropout,
            eps=config.eps,
            **kwargs,
        )

    def forward_vector_field(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the instantaneous vector field v_theta(x_t, t, context) = dx_t / dt.

        Args:
            x_t: Latent state at time t, shape [batch_size, latent_dim] or [latent_dim].
            t: Continuous time in [0, 1], shape [batch_size] or scalar float/tensor.
            context: Conditioning context vector, shape [batch_size, context_dim] or [context_dim].

        Returns:
            Predicted velocity vector field [batch_size, latent_dim] (or unbatched).
        """
        is_unbatched = False
        if x_t.dim() == 1:
            is_unbatched = True
            x_t = x_t.unsqueeze(0)

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=x_t.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0)

        if t.shape[0] != x_t.shape[0]:
            t = t.expand(x_t.shape[0])

        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.shape[0] != x_t.shape[0]:
            context = context.expand(x_t.shape[0], -1)

        # Time embedding
        t_emb = self.time_embed(t)
        t_feat = self.time_mlp(t_emb)

        # Context conditioning
        c_normed = self.context_norm(context)
        cond = torch.cat([t_feat, c_normed], dim=-1)  # [batch_size, cond_dim]

        # Vector field prediction
        h = self.in_proj(x_t)
        for block in self.blocks:
            h = block(h, cond)

        v_pred = self.out_proj(self.final_norm(h))

        if is_unbatched:
            v_pred = v_pred.squeeze(0)

        return v_pred

    def compute_loss(
        self,
        target_invariants: torch.Tensor,
        context: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> FlowMatchingLossOutput:
        """
        Compute Optimal Transport Conditional Flow Matching (OT-CFM) loss.

        Args:
            target_invariants: Real target lemma representations x_1 [batch_size, latent_dim].
            context: Context representations [batch_size, context_dim].
            noise: Optional base standard normal noise x_0 ~ N(0, I).
            t: Optional sampled times in [0, 1].

        Returns:
            FlowMatchingLossOutput with mean-squared-error loss and trajectory diagnostics.
        """
        batch_size = target_invariants.shape[0]
        device = target_invariants.device
        dtype = target_invariants.dtype

        # Base noise x_0 ~ N(0, I)
        if noise is None:
            noise = torch.randn_like(target_invariants)
        x_0 = noise
        x_1 = target_invariants

        # Sample uniform time t in [0, 1]
        if t is None:
            t = torch.rand(batch_size, device=device, dtype=dtype)

        t_expanded = t.view(batch_size, 1)

        # OT Flow trajectory interpolation:
        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        alpha_t = 1.0 - (1.0 - self.sigma_min) * t_expanded
        beta_t = t_expanded
        x_t = alpha_t * x_0 + beta_t * x_1

        # Target vector field: u_t(x_t | x_1, x_0) = x_1 - (1 - sigma_min) * x_0
        v_target = x_1 - (1.0 - self.sigma_min) * x_0

        # Predict vector field
        v_pred = self.forward_vector_field(x_t, t, context)

        # Flow Matching regression loss (MSE)
        loss = F.mse_loss(v_pred, v_target)

        return FlowMatchingLossOutput(
            loss=loss,
            v_pred=v_pred,
            v_target=v_target,
            x_t=x_t,
            t=t,
        )

    @torch.no_grad()
    def sample_latent_invariant(
        self,
        context: torch.Tensor,
        num_steps: int = 10,
        solver: str = "midpoint",
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample continuous lemma invariant representation from base Gaussian prior
        by integrating the learned vector field along [0, 1] via ODE solvers.

        Args:
            context: Conditioning context vector [batch_size, context_dim] or [context_dim].
            num_steps: Number of integration steps (recommended 5-10 for OT flow).
            solver: ODE integration method: 'euler', 'midpoint' (RK2), or 'rk4'.
            noise: Optional initial noise vector x_0.

        Returns:
            Generated latent invariant tensor x_1 [batch_size, latent_dim] or [latent_dim].
        """
        is_unbatched = False
        if context.dim() == 1:
            is_unbatched = True
            context = context.unsqueeze(0)

        batch_size = context.shape[0]
        device = context.device
        dtype = context.dtype

        # Initial sample x_0 ~ N(0, I)
        if noise is None:
            x_t = torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)
        else:
            x_t = noise if noise.dim() == 2 else noise.unsqueeze(0)

        # Time discretization steps along t in [0, 1]
        dt = 1.0 / float(num_steps)
        solver_name = solver.lower()

        for step in range(num_steps):
            t_curr = step * dt
            t_tensor = torch.full((batch_size,), t_curr, device=device, dtype=dtype)

            if solver_name == "euler":
                v = self.forward_vector_field(x_t, t_tensor, context)
                x_t = x_t + dt * v

            elif solver_name == "midpoint":
                # RK2 Midpoint:
                # k1 = v(x_t, t)
                # k2 = v(x_t + 0.5 * dt * k1, t + 0.5 * dt)
                # x_{t+dt} = x_t + dt * k2
                v1 = self.forward_vector_field(x_t, t_tensor, context)
                t_mid = torch.full((batch_size,), t_curr + 0.5 * dt, device=device, dtype=dtype)
                x_mid = x_t + 0.5 * dt * v1
                v2 = self.forward_vector_field(x_mid, t_mid, context)
                x_t = x_t + dt * v2

            elif solver_name == "rk4":
                # Classical 4th-order Runge-Kutta
                t_mid = torch.full((batch_size,), t_curr + 0.5 * dt, device=device, dtype=dtype)
                t_next = torch.full((batch_size,), t_curr + dt, device=device, dtype=dtype)

                k1 = self.forward_vector_field(x_t, t_tensor, context)
                k2 = self.forward_vector_field(x_t + 0.5 * dt * k1, t_mid, context)
                k3 = self.forward_vector_field(x_t + 0.5 * dt * k2, t_mid, context)
                k4 = self.forward_vector_field(x_t + dt * k3, t_next, context)

                x_t = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            else:
                raise ValueError(
                    f"Unknown solver '{solver}'. Supported solvers: 'euler', 'midpoint', 'rk4'."
                )

        if is_unbatched:
            x_t = x_t.squeeze(0)

        return x_t


def get_separate_optimizer_param_groups(
    backbone: nn.Module,
    policy_head: nn.Module,
    value_head: nn.Module,
    diffusion_head: nn.Module,
    lr_backbone: float = 1e-4,
    lr_policy: float = 1e-4,
    lr_value: float = 1e-4,
    lr_flow: float = 2e-4,
    weight_decay: float = 1e-2,
) -> List[Dict[str, Union[List[nn.Parameter], float, str]]]:
    """
    Construct separate parameter groups for AdamW optimizer to prevent gradient conflicts
    between discrete policy/value tasks and generative continuous flow matching.

    Returns:
        List of parameter group dictionaries compatible with torch.optim.Optimizer.
    """
    param_groups = [
        {
            "name": "backbone",
            "params": list(backbone.parameters()),
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        },
        {
            "name": "policy_head",
            "params": list(policy_head.parameters()),
            "lr": lr_policy,
            "weight_decay": weight_decay,
        },
        {
            "name": "value_head",
            "params": list(value_head.parameters()),
            "lr": lr_value,
            "weight_decay": weight_decay,
        },
        {
            "name": "flow_head",
            "params": list(diffusion_head.parameters()),
            "lr": lr_flow,
            "weight_decay": weight_decay,
        },
    ]
    return param_groups
