"""
Complete Composite AXMANet Model Architecture for Training and Inference.
Part of AXMAN (AXMANet) Phase 6.2 Implementation.

This module unifies:
1. AXMANetBackbone (Topological port attention & categorical morphism blocks)
2. MorphismPolicyHead (Action classification & Premise Selection Pointing Network)
3. DistanceToClosureValueHead (Attention Pooling, Closure probability, Steps-to-QED)
4. LatentInvariantDiffusionHead (Flow Matching OT-CFM lemma abduction)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from axman.core.config import AxmanConfig
from axman.core.blocks.amb_backbone import AXMANetBackbone, BackboneOutput
from axman.core.heads.policy_head import MorphismPolicyHead, PolicyOutput, TacticConstructorKind
from axman.core.heads.value_head import DistanceToClosureValueHead, ValueOutput
from axman.core.heads.diffusion_head import LatentInvariantDiffusionHead, FlowMatchingLossOutput
from axman.core.memory.graph_pool import BatchedHypergraph
from axman.core.data.dataset import BatchedTargets


@dataclass
class ModelOutput:
    """Container for full AXMANet multi-head inference predictions."""
    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    action_logits: torch.Tensor                    # [num_graphs, num_actions]
    closure_probabilities: torch.Tensor            # [num_graphs, 1]
    topological_distances: torch.Tensor            # [num_graphs, 1]
    pooled_context: torch.Tensor                   # [num_graphs, d_model]
    subgraph_premise_logits: List[torch.Tensor]    # list of [1, N_i] premise scores
    sampled_lemma_invariants: Optional[torch.Tensor] = None


@dataclass
class TaskLosses:
    """Container for unweighted task losses and metrics."""
    policy_loss: torch.Tensor
    premise_loss: torch.Tensor
    value_loss: torch.Tensor
    flow_loss: torch.Tensor
    total_loss: torch.Tensor
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "policy": self.policy_loss,
            "premise": self.premise_loss,
            "value": self.value_loss,
            "flow": self.flow_loss,
        }


class AXMANetModel(nn.Module):
    """
    Unified AXMANet end-to-end multi-task neural network.
    """

    def __init__(self, config: AxmanConfig):
        super().__init__()
        self.config = config

        # 1. Core Topological Backbone
        self.backbone = AXMANetBackbone(config)

        # 2. Model Heads
        self.policy_head = MorphismPolicyHead(
            d_model=config.d_model,
            num_actions=len(TacticConstructorKind),
            dropout=config.dropout,
            eps=config.eps,
        )

        self.value_head = DistanceToClosureValueHead(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            eps=config.eps,
        )

        self.diffusion_head = LatentInvariantDiffusionHead(
            latent_dim=config.d_model,
            context_dim=config.d_model,
            dropout=config.dropout,
            eps=config.eps,
        )

    @classmethod
    def from_preset(cls, preset: str = "small", **kwargs) -> "AXMANetModel":
        if preset == "small":
            cfg = AxmanConfig.small(**kwargs)
        elif preset == "base":
            cfg = AxmanConfig.base(**kwargs)
        elif preset == "large":
            cfg = AxmanConfig.large(**kwargs)
        else:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: small, base, large.")
        return cls(cfg)

    def forward(
        self,
        batched_graph: BatchedHypergraph,
        open_goal_mask: Optional[torch.Tensor] = None,
        sample_lemmas: bool = False,
        diffusion_steps: int = 5,
    ) -> ModelOutput:
        """
        Execute forward pass across backbone and all heads.
        """
        # Step 1: Backbone forward
        bb_out: BackboneOutput = self.backbone(
            node_features=batched_graph.node_features,
            edge_sources=batched_graph.edge_sources,
            edge_ports=batched_graph.edge_ports,
            edge_targets=batched_graph.edge_targets,
            edge_offsets=batched_graph.edge_offsets,
        )
        node_emb = bb_out.node_embeddings
        edge_emb = bb_out.edge_embeddings

        # Step 2: Unbatch node embeddings per sub-graph
        slices = batched_graph.unbatch_node_features(node_emb)
        num_graphs = batched_graph.num_graphs

        action_logits_list = []
        premise_logits_list = []
        val_probs = []
        val_dists = []
        val_latents = []

        # Step 3: Run Policy and Value heads for each sub-graph
        for i, s_nodes in enumerate(slices):
            # Extract slice of open goal mask for sub-graph if present
            sub_mask = None
            if open_goal_mask is not None:
                node_ptr = batched_graph.graph_node_ptr
                if isinstance(node_ptr, torch.Tensor):
                    start, end = node_ptr[i].item(), node_ptr[i + 1].item()
                else:
                    start, end = node_ptr[i], node_ptr[i + 1]
                sub_mask = open_goal_mask[start:end]

            # Goal embedding: mean over open goal mask if present, otherwise first node
            if sub_mask is not None and torch.any(sub_mask):
                goal_emb = s_nodes[sub_mask].mean(dim=0, keepdim=True)
            else:
                goal_emb = s_nodes[0:1]

            # Policy Head: Action logits & Premise Pointing Network
            pol_out: PolicyOutput = self.policy_head(
                goal_embeddings=goal_emb,
                candidate_embeddings=s_nodes,
            )
            action_logits_list.append(pol_out.action_logits)
            premise_logits_list.append(pol_out.premise_logits)

            # Value Head: Attention Pooling across sub-graph
            val_out: ValueOutput = self.value_head(
                node_embeddings=s_nodes,
                node_mask=sub_mask if sub_mask is not None and torch.any(sub_mask) else None,
            )
            val_probs.append(val_out.value_prob)
            val_dists.append(val_out.distance_to_qed)
            val_latents.append(val_out.graph_latent)

        action_logits = torch.cat(action_logits_list, dim=0)
        closure_probs = torch.cat(val_probs, dim=0).unsqueeze(-1) if val_probs[0].dim() == 0 else torch.stack(val_probs, dim=0)
        topological_dists = torch.cat(val_dists, dim=0).unsqueeze(-1) if val_dists[0].dim() == 0 else torch.stack(val_dists, dim=0)
        pooled_context = torch.stack(val_latents, dim=0) if val_latents[0].dim() == 1 else torch.cat(val_latents, dim=0)

        # Step 4: Optional lemma abduction sampling via ODE solver
        sampled_lemmas = None
        if sample_lemmas:
            sampled_lemmas = self.diffusion_head.sample_latent_invariant(
                context=pooled_context,
                num_steps=diffusion_steps,
                solver="euler",
            )

        return ModelOutput(
            node_embeddings=node_emb,
            edge_embeddings=edge_emb,
            action_logits=action_logits,
            closure_probabilities=closure_probs,
            topological_distances=topological_dists,
            pooled_context=pooled_context,
            subgraph_premise_logits=premise_logits_list,
            sampled_lemma_invariants=sampled_lemmas,
        )

    def compute_losses(
        self,
        batched_graph: BatchedHypergraph,
        targets: BatchedTargets,
        temperature: float = 1.0,
    ) -> TaskLosses:
        """
        Compute multi-task loss terms:
        1. L_policy_ce: Cross-Entropy for tactic action.
        2. L_premise_nce: InfoNCE contrastive ranking loss for premise selection.
        3. L_value_mse: Combined MSE for closure probability and distance-to-QED.
        4. L_flow_matching: Optimal Transport CFM loss for lemma abduction.
        """
        # Forward pass
        out = self.forward(
            batched_graph=batched_graph,
            open_goal_mask=targets.open_goal_mask,
        )

        num_graphs = batched_graph.num_graphs
        device = out.node_embeddings.device

        # 1. Policy Action Loss (Cross-Entropy)
        if targets.target_actions is not None:
            policy_loss = F.cross_entropy(out.action_logits, targets.target_actions.to(device))
            acc = (out.action_logits.argmax(dim=-1) == targets.target_actions.to(device)).float().mean().item()
        else:
            policy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            acc = 0.0

        # 2. Premise Selection Loss (InfoNCE / Contrastive Softmax per sub-graph)
        premise_losses = []
        premise_top1_hits = 0
        total_graphs_with_prems = 0

        node_ptr = batched_graph.graph_node_ptr
        if isinstance(node_ptr, torch.Tensor):
            ptr = node_ptr.cpu().tolist()
        else:
            ptr = list(node_ptr)

        for i in range(num_graphs):
            logits = out.subgraph_premise_logits[i].squeeze(0)  # [N_i]
            if targets.target_premises is not None and targets.target_premises[i] is not None:
                glob_prems = targets.target_premises[i]
                if isinstance(glob_prems, torch.Tensor):
                    glob_prems = glob_prems.to(device)
                else:
                    glob_prems = torch.tensor(glob_prems, device=device, dtype=torch.long)

                if glob_prems.numel() > 0:
                    total_graphs_with_prems += 1
                    # Convert global premise indices back to local sub-graph indices
                    local_prems = glob_prems - ptr[i]
                    # Filter valid local premises within [0, logits.shape[0])
                    valid_mask = (local_prems >= 0) & (local_prems < logits.shape[0])
                    valid_prems = local_prems[valid_mask]

                    if valid_prems.numel() > 0:
                        # InfoNCE: -log(sum(exp(s_pos / tau)) / sum(exp(s_all / tau)))
                        log_denom = torch.logsumexp(logits / temperature, dim=-1)
                        pos_logits = logits[valid_prems] / temperature
                        log_num = torch.logsumexp(pos_logits, dim=-1)
                        sub_loss = log_denom - log_num
                        premise_losses.append(sub_loss)

                        top1_pred = logits.argmax().item()
                        if top1_pred in valid_prems.tolist():
                            premise_top1_hits += 1

        if premise_losses:
            premise_loss = torch.stack(premise_losses).mean()
            premise_acc = premise_top1_hits / max(total_graphs_with_prems, 1)
        else:
            premise_loss = torch.tensor(0.0, device=device, requires_grad=True)
            premise_acc = 0.0

        # 3. Value Loss (Closure probability BCE/MSE + Steps-to-QED Huber/MSE)
        val_closure_loss = torch.tensor(0.0, device=device)
        val_dist_loss = torch.tensor(0.0, device=device)
        has_val = False

        if targets.target_values is not None:
            tgt_v = targets.target_values.to(device).view(-1, 1)
            pred_v = out.closure_probabilities.view(-1, 1)
            val_closure_loss = F.mse_loss(pred_v, tgt_v)
            has_val = True

        if targets.target_distances is not None:
            tgt_d = targets.target_distances.to(device).view(-1, 1)
            pred_d = out.topological_distances.view(-1, 1)
            val_dist_loss = F.smooth_l1_loss(pred_d, tgt_d)
            has_val = True

        if has_val:
            value_loss = val_closure_loss + 0.5 * val_dist_loss
        else:
            value_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 4. Latent Invariant Flow Matching Loss
        if targets.target_latents is not None:
            tgt_latents = targets.target_latents.to(device)
            fm_out: FlowMatchingLossOutput = self.diffusion_head.compute_loss(
                target_invariants=tgt_latents,
                context=out.pooled_context,
            )
            flow_loss = fm_out.loss
        else:
            flow_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = policy_loss + premise_loss + value_loss + flow_loss

        metrics = {
            "policy_acc": acc,
            "premise_top1": premise_acc,
            "value_closure_mse": val_closure_loss.item(),
            "value_dist_loss": val_dist_loss.item(),
            "flow_matching_mse": flow_loss.item(),
        }

        return TaskLosses(
            policy_loss=policy_loss,
            premise_loss=premise_loss,
            value_loss=value_loss,
            flow_loss=flow_loss,
            total_loss=total_loss,
            metrics=metrics,
        )
