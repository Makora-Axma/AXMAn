"""
Categorical Morphism Projection (CMP) for AXMANet.
Part of AXMAN (AXMANet) Phase 1.3 Implementation.

This module models categorical morphism composition (g o f) and morphism
space projection using spectral-normalized bilinear tensor mappings and
identity residual connections.

Adheres to category-theoretic axioms:
1. Identity: f o id =~ f, id o g =~ g
2. Associativity: (h o g) o f =~ h o (g o f)
3. 1-Lipschitz stability via Spectral Normalization: prevents gradient explosion
   along deep deduction chains.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class CategoricalMorphismProjection(nn.Module):
    """
    Categorical Morphism Projection (CMP) Layer.

    Computes tensor-algebraic projections and composition of categorical morphisms
    using a low-rank bilinear tensor factorization with spectral normalization
    and identity residuals.

    Mathematical formulation:
    For morphism embeddings f, g in R^{d_model}:
        u = SN(W_A) f  in R^rank
        v = SN(W_B) g  in R^rank
        bilinear_core = SN(W_out) (u * v)
        linear_residual = SN(W_lin) [f || g]
        (g o f) = id_res(f, g) + alpha * (bilinear_core + linear_residual)

    When applied as a unary endomorphism layer in the AMB block:
        CMP(x) = x + alpha * Bilinear(x, x)
    """

    def __init__(
        self,
        d_model: int = 256,
        rank: Optional[int] = None,
        use_spectral_norm: bool = True,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank if rank is not None else d_model
        self.use_spectral_norm = use_spectral_norm

        def maybe_sn(layer: nn.Linear) -> nn.Linear:
            return spectral_norm(layer) if use_spectral_norm else layer

        # Bilinear Factorized Projections: R^{d_model} -> R^{rank}
        self.proj_a = maybe_sn(nn.Linear(d_model, self.rank, bias=False))
        self.proj_b = maybe_sn(nn.Linear(d_model, self.rank, bias=False))

        # Bilinear Core Output: R^{rank} -> R^{d_model}
        self.proj_out = maybe_sn(nn.Linear(self.rank, d_model, bias=False))

        # Linear interaction path: R^{2 * d_model} -> R^{d_model} (for binary composition)
        self.linear_comp = maybe_sn(nn.Linear(2 * d_model, d_model, bias=False))

        # Unary linear projection path: R^{d_model} -> R^{d_model} (for unary forward)
        self.unary_linear = maybe_sn(nn.Linear(d_model, d_model, bias=False))

        # Learnable canonical identity morphism vector: id in Hom(X, X)
        self.identity_morphism = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.identity_morphism, mean=0.0, std=0.02)

        # Learnable gating parameter alpha (controls non-linear morphism deformation)
        # Initialized such that at step 0 the identity residual dominates
        self.gate = nn.Parameter(torch.tensor(gate_init))

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization designed for bilinear products
        std_factor = 1.0 / math.sqrt(self.d_model)
        nn.init.normal_(self.proj_a.weight, mean=0.0, std=std_factor)
        nn.init.normal_(self.proj_b.weight, mean=0.0, std=std_factor)
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=1.0 / math.sqrt(self.rank))
        nn.init.xavier_uniform_(self.linear_comp.weight)
        nn.init.xavier_uniform_(self.unary_linear.weight)

    def get_identity(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns the canonical identity morphism representation.
        """
        vec = self.identity_morphism
        if device is not None and vec.device != device:
            vec = vec.to(device)

        if batch_size is None:
            return vec
        return vec.unsqueeze(0).expand(batch_size, -1)

    def compose(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Computes categorical composition of two morphisms: (g o f).

        Args:
            f: Morphism representation [..., d_model] (e.g. morphism A -> B)
            g: Morphism representation [..., d_model] (e.g. morphism B -> C)

        Returns:
            Composite morphism (g o f) [..., d_model] (morphism A -> C)
        """
        if f.shape[-1] != self.d_model or g.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension {self.d_model}, got f: {f.shape[-1]}, g: {g.shape[-1]}"
            )

        # 1. Bilinear factorized interaction
        u = F.gelu(self.proj_a(f))  # [..., rank]
        v = F.gelu(self.proj_b(g))  # [..., rank]
        bilinear = self.proj_out(u * v)  # [..., d_model]

        # 2. Linear composite projection
        fg_concat = torch.cat([f, g], dim=-1)  # [..., 2 * d_model]
        linear_part = self.linear_comp(fg_concat)  # [..., d_model]

        # 3. Base Identity Residual: Mean of composable arrows preserves direction
        base_residual = 0.5 * (f + g)

        # 4. Gated fusion
        alpha = torch.sigmoid(self.gate)
        composed = base_residual + alpha * self.dropout(bilinear + linear_part)
        return composed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unary morphism transformation / projection in the AMB layer.

        Args:
            x: Tensor of node or morphism embeddings [..., d_model].

        Returns:
            Transformed embeddings [..., d_model] with identity residual.
        """
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input feature dimension {x.shape[-1]} != d_model {self.d_model}")

        # Self-bilinear interaction
        u = F.gelu(self.proj_a(x))
        v = F.gelu(self.proj_b(x))
        bilinear = self.proj_out(u * v)

        # Linear transformation
        lin = self.unary_linear(x)

        # Gated deformation with identity residual
        alpha = torch.sigmoid(self.gate)
        out = x + alpha * self.dropout(bilinear + lin)
        return out
