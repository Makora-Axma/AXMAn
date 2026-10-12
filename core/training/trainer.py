"""
Autonomous Multi-Task Trainer for AXMANet (AXMA-N).
Part of AXMAN (AXMANet) Phase 6.2 Implementation.

This module provides AXMANetTrainer:
- Decoupled AdamW parameter groups.
- Cosine Annealing with Warmup scheduler.
- Dynamic loss balancing via GradNorm or PCGrad.
- Automatic Mixed Precision (AMP FP16/BF16).
- Evaluation, checkpointing, and Safetensors export.
"""

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from axman.core.config import AxmanConfig
from axman.core.training.model import AXMANetModel, TaskLosses
from axman.core.training.loss_balancer import GradNormLossBalancer, PCGrad
from axman.core.heads.diffusion_head import get_separate_optimizer_param_groups
from axman.core.runtime.safetensors_export import export_model_to_safetensors


@dataclass
class TrainingConfig:
    """Hyperparameter configuration for AXMANet training."""
    epochs: int = 10
    batch_size: int = 16
    lr_backbone: float = 1e-4
    lr_policy: float = 1e-4
    lr_value: float = 1e-4
    lr_flow: float = 2e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 100
    balancer_type: str = "gradnorm"          # "gradnorm", "pcgrad", "static"
    gradnorm_alpha: float = 0.16
    gradnorm_lr: float = 0.025
    grad_clip_norm: float = 1.0
    mixed_precision: str = "no"              # "no", "fp16", "bf16"
    device: str = "cpu"
    eval_every: int = 1
    save_every: int = 1
    output_dir: str = "checkpoints"


@dataclass
class EpochMetrics:
    """Metrics recorded during a training/validation epoch."""
    epoch: int
    train_total_loss: float
    train_policy_loss: float
    train_premise_loss: float
    train_value_loss: float
    train_flow_loss: float
    train_policy_acc: float
    train_premise_top1: float
    val_total_loss: Optional[float] = None
    val_policy_acc: Optional[float] = None
    val_premise_top1: Optional[float] = None
    task_weights: Optional[Dict[str, float]] = None
    elapsed_seconds: float = 0.0


class AXMANetTrainer:
    """
    Autonomous multi-task trainer orchestrating AXMANet pre-training and fine-tuning.
    """

    def __init__(
        self,
        model: AXMANetModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        train_config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_config or TrainingConfig()

        # Device setup
        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.cfg.device)

        self.model.to(self.device)

        # 1. Decoupled AdamW Parameter Groups
        param_groups = get_separate_optimizer_param_groups(
            backbone=self.model.backbone,
            policy_head=self.model.policy_head,
            value_head=self.model.value_head,
            diffusion_head=self.model.diffusion_head,
            lr_backbone=self.cfg.lr_backbone,
            lr_policy=self.cfg.lr_policy,
            lr_value=self.cfg.lr_value,
            lr_flow=self.cfg.lr_flow,
            weight_decay=self.cfg.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        # 2. Learning rate schedule: Cosine Annealing with Linear Warmup
        total_steps = max(len(self.train_loader) * self.cfg.epochs, 1)
        self.total_steps = total_steps
        self.current_step = 0

        # 3. Multi-task Loss Balancer setup
        self.gradnorm: Optional[GradNormLossBalancer] = None
        self.pcgrad: Optional[PCGrad] = None

        if self.cfg.balancer_type.lower() == "gradnorm":
            self.gradnorm = GradNormLossBalancer(
                task_names=("policy", "premise", "value", "flow"),
                alpha=self.cfg.gradnorm_alpha,
                lr_weights=self.cfg.gradnorm_lr,
                device=self.device,
            )
        elif self.cfg.balancer_type.lower() == "pcgrad":
            self.pcgrad = PCGrad(self.optimizer)

        # 4. Mixed precision setup
        self.amp_enabled = self.cfg.mixed_precision.lower() in ("fp16", "bf16") and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if self.cfg.mixed_precision.lower() == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp_enabled and self.amp_dtype == torch.float16))

        # Checkpoint directory
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[EpochMetrics] = []

    def _adjust_learning_rate(self, step: int):
        """Linear warmup followed by cosine annealing schedule."""
        if step < self.cfg.warmup_steps:
            alpha = float(step) / max(float(self.cfg.warmup_steps), 1.0)
        else:
            progress = min(max(float(step - self.cfg.warmup_steps) / max(float(self.total_steps - self.cfg.warmup_steps), 1.0), 0.0), 1.0)
            alpha = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Apply multiplier to base lr of each param group
        base_lrs = [self.cfg.lr_backbone, self.cfg.lr_policy, self.cfg.lr_value, self.cfg.lr_flow]
        for i, group in enumerate(self.optimizer.param_groups):
            group["lr"] = base_lrs[i] * max(alpha, 1e-2)

    def train_epoch(self, epoch: int) -> EpochMetrics:
        """Run a single training epoch."""
        self.model.train()
        start_time = time.time()

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        premise_loss_accum = 0.0
        value_loss_accum = 0.0
        flow_loss_accum = 0.0
        policy_acc_accum = 0.0
        premise_top1_accum = 0.0
        num_batches = len(self.train_loader)

        for b_idx, (b_graph, b_targets) in enumerate(self.train_loader):
            self.current_step += 1
            self._adjust_learning_rate(self.current_step)

            # Move batch data to device
            b_graph = b_graph.to(self.device)
            b_targets = b_targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass & loss computation under optional AMP
            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                task_losses: TaskLosses = self.model.compute_losses(b_graph, b_targets)

            # Optimization step based on balancer type
            if self.gradnorm is not None:
                # Dynamic GradNorm balancing
                loss_dict = task_losses.to_dict()
                shared_params = list(self.model.backbone.parameters())[-1]
                self.gradnorm.step(loss_dict, shared_params)
                weighted_loss = self.gradnorm.compute_weighted_loss(loss_dict)

                if self.amp_enabled and self.scaler.is_enabled():
                    self.scaler.scale(weighted_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    weighted_loss.backward()
                    if self.cfg.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                    self.optimizer.step()

            elif self.pcgrad is not None:
                # PCGrad conflict-free backpropagation
                task_list = [
                    task_losses.policy_loss,
                    task_losses.premise_loss,
                    task_losses.value_loss,
                    task_losses.flow_loss,
                ]
                self.pcgrad.pcgrad_backward(task_list)
                if self.cfg.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.pcgrad.step()

            else:
                # Static weighting
                total_loss = (
                    task_losses.policy_loss
                    + task_losses.premise_loss
                    + 0.5 * task_losses.value_loss
                    + task_losses.flow_loss
                )
                if self.amp_enabled and self.scaler.is_enabled():
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    if self.cfg.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                    self.optimizer.step()

            # Accumulate statistics
            total_loss_accum += task_losses.total_loss.item()
            policy_loss_accum += task_losses.policy_loss.item()
            premise_loss_accum += task_losses.premise_loss.item()
            value_loss_accum += task_losses.value_loss.item()
            flow_loss_accum += task_losses.flow_loss.item()
            policy_acc_accum += task_losses.metrics["policy_acc"]
            premise_top1_accum += task_losses.metrics["premise_top1"]

        elapsed = time.time() - start_time
        task_weights = self.gradnorm.get_weights_dict() if self.gradnorm else None

        metrics = EpochMetrics(
            epoch=epoch,
            train_total_loss=total_loss_accum / max(num_batches, 1),
            train_policy_loss=policy_loss_accum / max(num_batches, 1),
            train_premise_loss=premise_loss_accum / max(num_batches, 1),
            train_value_loss=value_loss_accum / max(num_batches, 1),
            train_flow_loss=flow_loss_accum / max(num_batches, 1),
            train_policy_acc=policy_acc_accum / max(num_batches, 1),
            train_premise_top1=premise_top1_accum / max(num_batches, 1),
            task_weights=task_weights,
            elapsed_seconds=elapsed,
        )

        return metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run validation evaluation on val_loader."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        policy_acc = 0.0
        premise_top1 = 0.0
        num_batches = len(self.val_loader)

        for b_graph, b_targets in self.val_loader:
            b_graph = b_graph.to(self.device)
            b_targets = b_targets.to(self.device)

            losses = self.model.compute_losses(b_graph, b_targets)
            total_loss += losses.total_loss.item()
            policy_acc += losses.metrics["policy_acc"]
            premise_top1 += losses.metrics["premise_top1"]

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_policy_acc": policy_acc / max(num_batches, 1),
            "val_premise_top1": premise_top1 / max(num_batches, 1),
        }

    def save_checkpoint(
        self,
        epoch: int,
        tag: str = "latest",
        export_safetensors_format: bool = True,
    ) -> Path:
        """Save training checkpoint and export safetensors archive."""
        ckpt_path = self.output_dir / f"axman_model_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.config,
                "training_config": self.cfg,
            },
            ckpt_path,
        )

        if export_safetensors_format:
            st_path = self.output_dir / f"axman_model_{tag}.safetensors"
            export_model_to_safetensors(
                model=self.model,
                config=self.model.config,
                output_path=st_path,
                extra_metadata={"epoch": str(epoch), "tag": tag},
            )

        return ckpt_path

    def train(self) -> List[EpochMetrics]:
        """Execute complete training loop over configured epochs."""
        best_val_loss = float("inf")

        for epoch in range(1, self.cfg.epochs + 1):
            metrics = self.train_epoch(epoch)

            if self.val_loader is not None and (epoch % self.cfg.eval_every == 0 or epoch == self.cfg.epochs):
                val_res = self.evaluate()
                metrics.val_total_loss = val_res.get("val_loss")
                metrics.val_policy_acc = val_res.get("val_policy_acc")
                metrics.val_premise_top1 = val_res.get("val_premise_top1")

                if metrics.val_total_loss is not None and metrics.val_total_loss < best_val_loss:
                    best_val_loss = metrics.val_total_loss
                    self.save_checkpoint(epoch, tag="best")

            if epoch % self.cfg.save_every == 0 or epoch == self.cfg.epochs:
                self.save_checkpoint(epoch, tag="latest")

            self.history.append(metrics)

        # Final export
        self.save_checkpoint(self.cfg.epochs, tag="final")
        return self.history
