"""
Multi-Task Loss Balancing and Gradient Conflict Mitigation for AXMANet.
Part of AXMAN (AXMANet) Phase 6.2 Implementation.

This module provides:
1. GradNormLossBalancer: Dynamically balances multi-task loss weights according to
   gradient norms and relative training rates to prevent task domination.
2. PCGrad: Projecting Conflicting Gradients optimizer wrapper that projects conflicting
   gradients onto normal planes to eliminate destructive interference between discrete
   policy/value heads and continuous flow matching.
3. StaticLossBalancer: Baseline fixed weighting with cosine schedule for diffusion.
"""

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Container for multi-task loss weights."""
    policy: float = 1.0
    premise: float = 1.0
    value: float = 0.5
    flow: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "policy": self.policy,
            "premise": self.premise,
            "value": self.value,
            "flow": self.flow,
        }


class GradNormLossBalancer(nn.Module):
    """
    GradNorm (Chen et al., 2018): Gradient Normalization for Adaptive Loss Balancing.

    Dynamically tunes task weights w_i(t) to balance gradient magnitudes and
    equalize the pace of learning across tasks:
    1. Computes gradient norms G_i(t) with respect to shared backbone parameters.
    2. Computes relative training rates r_i(t) = (L_i(t) / L_i(0)) / (average relative loss).
    3. Minimizes L1 loss between G_i(t) and target gradient norm G_target = G_avg * (r_i(t))^alpha.
    4. Renormalizes weights so that sum(w_i) = num_tasks.
    """

    def __init__(
        self,
        task_names: Sequence[str] = ("policy", "premise", "value", "flow"),
        alpha: float = 0.16,
        lr_weights: float = 0.025,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.task_names = list(task_names)
        self.num_tasks = len(self.task_names)
        self.alpha = alpha

        # Learnable task weights initialized to 1.0
        init_weights = torch.ones(self.num_tasks, dtype=torch.float32, device=device)
        self.weights = nn.Parameter(init_weights)

        self.optimizer = torch.optim.Adam([self.weights], lr=lr_weights)

        # Baseline loss records L_i(0)
        self.register_buffer("initial_losses", torch.zeros(self.num_tasks, dtype=torch.float32, device=device))
        self.has_initial_losses = False

    def get_weights_dict(self) -> Dict[str, float]:
        """Return current task weights as a dictionary."""
        with torch.no_grad():
            w = self.weights.detach().cpu().numpy()
            return {name: float(w[i]) for i, name in enumerate(self.task_names)}

    def compute_weighted_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute scalar total loss as sum(w_i * L_i).
        """
        loss_tensor = torch.stack([losses[name] for name in self.task_names])

        if not self.has_initial_losses:
            with torch.no_grad():
                self.initial_losses.copy_(loss_tensor.detach().clamp(min=1e-8))
                self.has_initial_losses = True

        # Normalized weights ensuring sum(w_i) = num_tasks (detached for model backprop)
        norm_weights = (self.weights / (self.weights.sum() / self.num_tasks)).detach()
        weighted_loss = (norm_weights * loss_tensor).sum()
        return weighted_loss

    def step(
        self,
        losses: Dict[str, torch.Tensor],
        shared_parameters: Union[nn.Parameter, Sequence[nn.Parameter]],
    ) -> Dict[str, float]:
        """
        Perform a GradNorm weight update step.

        Args:
            losses: Dictionary of unweighted task losses (policy, premise, value, flow).
            shared_parameters: Shared parameter tensor(s) to compute gradient norms against
                               (e.g. final layer weights of backbone).

        Returns:
            Dictionary of updated loss weights.
        """
        if isinstance(shared_parameters, (list, tuple)):
            shared_param = shared_parameters[-1]
        else:
            shared_param = shared_parameters

        loss_tensor = torch.stack([losses[name] for name in self.task_names])

        if not self.has_initial_losses:
            with torch.no_grad():
                self.initial_losses = self.initial_losses.to(loss_tensor.device)
                self.initial_losses.copy_(loss_tensor.detach().clamp(min=1e-8))
                self.has_initial_losses = True
            return self.get_weights_dict()

        if self.initial_losses.device != loss_tensor.device:
            self.initial_losses = self.initial_losses.to(loss_tensor.device)

        # 1. Compute task gradient norms G_i(t) = w_i * ||grad_W L_i||_2
        norms = []
        for i in range(self.num_tasks):
            # Compute gradient of task loss w.r.t shared parameters with retain_graph=True
            if loss_tensor[i].requires_grad and loss_tensor[i].grad_fn is not None:
                grads = torch.autograd.grad(
                    loss_tensor[i],
                    shared_param,
                    retain_graph=True,
                    allow_unused=True,
                )
                if grads[0] is not None:
                    g_norm = torch.norm(grads[0].detach(), p=2)
                else:
                    g_norm = torch.tensor(0.0, device=self.weights.device)
            else:
                g_norm = torch.tensor(0.0, device=self.weights.device)

            # G_i(t) = w_i * ||grad_W L_i||
            norms.append(self.weights[i] * g_norm)

        norms_tensor = torch.stack(norms)
        mean_norm = norms_tensor.mean().detach()

        # 2. Compute relative inverse training rates r_i(t)
        loss_ratios = (loss_tensor.detach() / self.initial_losses.clamp(min=1e-8))
        mean_ratio = loss_ratios.mean()
        inverse_rates = loss_ratios / mean_ratio.clamp(min=1e-8)

        # 3. Target norms
        target_norms = (mean_norm * (inverse_rates ** self.alpha)).detach()

        # 4. GradNorm loss on weights
        gradnorm_loss = F.l1_loss(norms_tensor, target_norms)

        # 5. Optimize weights
        self.optimizer.zero_grad()
        gradnorm_loss.backward()
        self.optimizer.step()

        # 6. Renormalize weights
        with torch.no_grad():
            self.weights.clamp_(min=1e-3)
            scale = self.num_tasks / self.weights.sum()
            self.weights.mul_(scale)

        return self.get_weights_dict()


class PCGrad:
    """
    Projecting Conflicting Gradients (PCGrad, Yu et al., 2020).

    When multi-task gradients conflict (cosine similarity < 0), PCGrad projects
    each gradient onto the normal plane of conflicting gradients, preventing
    destructive interference during backpropagation.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.optimizer.step()

    def pcgrad_backward(self, task_losses: Sequence[torch.Tensor]):
        """
        Compute projected conflicting gradients across multiple task losses.

        Args:
            task_losses: Sequence of scalar loss tensors for each task.
        """
        num_tasks = len(task_losses)
        if num_tasks <= 1:
            task_losses[0].backward()
            return

        # Retrieve shared trainable parameters
        parameters = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    parameters.append(p)

        # Compute task gradients for each task independently
        task_grads: List[List[torch.Tensor]] = []
        for i, loss in enumerate(task_losses):
            retain = (i < num_tasks - 1)
            if loss.requires_grad and loss.grad_fn is not None:
                loss.backward(retain_graph=retain)
                grads_i = []
                for p in parameters:
                    if p.grad is not None:
                        grads_i.append(p.grad.clone())
                        p.grad.zero_()
                    else:
                        grads_i.append(torch.zeros_like(p))
            else:
                grads_i = [torch.zeros_like(p) for p in parameters]
            task_grads.append(grads_i)

        # Project conflicting gradients
        projected_grads = [[g.clone() for g in grads] for grads in task_grads]

        for i in range(num_tasks):
            indices = list(range(num_tasks))
            indices.remove(i)
            random.shuffle(indices)

            for j in indices:
                # Dot product between gradient vectors
                dot = sum(
                    torch.sum(projected_grads[i][k] * task_grads[j][k])
                    for k in range(len(parameters))
                )
                if dot < 0.0:
                    norm_sq = sum(
                        torch.sum(task_grads[j][k] * task_grads[j][k])
                        for k in range(len(parameters))
                    )
                    scale = dot / (norm_sq + 1e-8)
                    for k in range(len(parameters)):
                        projected_grads[i][k] -= scale * task_grads[j][k]

        # Combine projected gradients into parameter gradients
        for k, p in enumerate(parameters):
            accum_grad = sum(projected_grads[i][k] for i in range(num_tasks))
            p.grad = accum_grad
